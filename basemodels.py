import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, BatchNormalization 
from tensorflow.keras.layers import  Activation, Dropout, TimeDistributed
from tensorflow.keras.layers import LSTM, RepeatVector, Flatten, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import anomalyutils
import utils
import pandas as pd

class AnomalyDetectionModel():
    def __init__(self, capture_info=False) -> None:
        if capture_info not in [False, True]:
            raise ValueError("capture_info should be of type bool either True or False")
        
        self.capture_info = capture_info
        self.model_name = None
        self.model = None
        self.UCL = None
        self.anomaly_col = None

    ## AE
    def autoencoder(self, data):

        input_dots = Input((data.shape[1],))

        x = Dense(5)(input_dots)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)

        x = Dense(4)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        bottleneck = Dense(2, activation='linear')(x)

        x = Dense(4)(bottleneck)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dense(5)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        out = Dense(data.shape[1], activation='linear')(x)

        model = Model(input_dots, out)
        model.compile(optimizer=Adam(0.005), loss='mae', metrics=["mse"])
        
        early_stopping = EarlyStopping(patience=3, verbose=0)
        model.fit(data, data,
                    validation_split=0.2,
                    epochs=40,
                    batch_size=32,
                    verbose=0,
                    shuffle=True,
                    callbacks=[early_stopping]
                )
        return model

    ## Conv_AE
    def conv_ae(self, data):
        model = keras.Sequential(
            [
                layers.Input(shape=(data.shape[1], data.shape[2])),
                layers.Conv1D(
                    filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                layers.Dropout(rate=0.2),
                layers.Conv1D(
                    filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                layers.Conv1DTranspose(
                    filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                layers.Dropout(rate=0.2),
                layers.Conv1DTranspose(
                    filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
            ]
        )
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
        # model.summary()

        history = model.fit(
            data,
            data,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=0)
            ],
        )
        return model#, history
        

    ## LSTM
    def lstm(self, data_x, data_y, name=None):
        name_of_method = ""
        if name is None:
            name_of_method = "lstm.h5"
        else:
            name_of_method = f"lstm_{name}.h5"
        
        n_features = data_x.shape[2]
        
        # model defining
        model = Sequential()
        model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(5, n_features)))
        model.add(LSTM(100, activation='relu'))
        model.add(Dense(n_features))
        model.compile(optimizer='adam', loss='mae', metrics=["mse"])
        
        # callbacks defining
        early_stopping = EarlyStopping(patience=10, verbose=0)
        model_checkpoint = ModelCheckpoint(name_of_method, save_best_only=True, verbose=0, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.0001, verbose=0)
        
        # model fitting
        history = model.fit(data_x, data_y,
                            validation_split=0.2,
                            epochs=25,
                            batch_size=32,
                            verbose=0,
                            shuffle=False,
                            callbacks=[early_stopping, model_checkpoint, reduce_lr])
        return model

    ## LSTM_AE
    def lstm_ae(self, data):
        # model defining
        # define encoder
        inputs = keras.Input(shape=(data.shape[1], data.shape[2]))
        encoded = layers.LSTM(100, activation='relu')(inputs)

        # define reconstruct decoder
        decoded = layers.RepeatVector(data.shape[1])(encoded)
        decoded = layers.LSTM(100, activation='relu', return_sequences=True)(decoded)
        decoded = layers.TimeDistributed(layers.Dense(data.shape[2]))(decoded)

        # tie it together
        model = keras.Model(inputs, decoded)
        encoder = keras.Model(inputs, encoded)

        model.compile(optimizer='adam', loss='mae', metrics=["mse"])
        
        # callbacks defining
        early_stopping = EarlyStopping(patience=5, verbose=0)
    #     reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.0001, verbose=0)
        
        # model fitting
        history = model.fit(data, data,
                            validation_split=0.1,
                            epochs=100,
                            batch_size=32,
                            verbose=0,
                            shuffle=False,
                            callbacks=[early_stopping]#, reduce_lr]
                            )
        return model

    ## LSMTM_VAE
    def create_lstm_vae(self, input_dim, 
        timesteps, 
        batch_size, 
        intermediate_dim, 
        latent_dim,
        epsilon_std):

        """
        Creates an LSTM Variational Autoencoder (VAE). Returns VAE, Encoder, Generator. 
        # Arguments
            input_dim: int.
            timesteps: int, input timestep dimension.
            batch_size: int.
            intermediate_dim: int, output shape of LSTM. 
            latent_dim: int, latent z-layer shape. 
            epsilon_std: float, z-layer sigma.
        # References
            - [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
            - [Generating sentences from a continuous space](https://arxiv.org/abs/1511.06349)
        """
        x = Input(shape=(timesteps, input_dim,))

        # LSTM encoding
        h = LSTM(intermediate_dim)(x)

        # VAE Z layer
        z_mean = Dense(latent_dim)(h)
        z_log_sigma = Dense(latent_dim)(h)
        
        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                    mean=0., stddev=epsilon_std)
            return z_mean + z_log_sigma * epsilon

        # note that "output_shape" isn't necessary with the TensorFlow backend
        # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
        
        # decoded LSTM layer
        decoder_h = LSTM(intermediate_dim, return_sequences=True)
        decoder_mean = LSTM(input_dim, return_sequences=True)

        h_decoded = RepeatVector(timesteps)(z)
        h_decoded = decoder_h(h_decoded)

        # decoded layer
        x_decoded_mean = decoder_mean(h_decoded)
        
        # end-to-end autoencoder
        vae = Model(x, x_decoded_mean)

        # encoder, from inputs to latent space
        encoder = Model(x, z_mean)

        # generator, from latent space to reconstructed inputs
        decoder_input = Input(shape=(latent_dim,))

        _h_decoded = RepeatVector(timesteps)(decoder_input)
        _h_decoded = decoder_h(_h_decoded)

        _x_decoded_mean = decoder_mean(_h_decoded)
        generator = Model(decoder_input, _x_decoded_mean)

        vae.compile(optimizer='rmsprop', loss="mae")
        
        return vae, encoder, generator

    def lstm_vae(self, data):
        input_dim = data.shape[-1] # 13
        timesteps = data.shape[1] # 3
        BATCH_SIZE = 1
        
        model, enc, gen = self.create_lstm_vae(input_dim, 
            timesteps=timesteps, 
            batch_size=BATCH_SIZE, 
            intermediate_dim=32,
            latent_dim=100,
            epsilon_std=1.)

        history = model.fit(
            data,
            data,
            epochs=20,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=0)
            ],
        )
        return model
    
    def fit(self, df_data, model_name, task_name):
        """define task_name for checkpoints of lstm"""
        self.model_name = model_name

        data = df_data.to_numpy()
        if self.model_name.upper() == "AE":
            self.model = self.autoencoder(data)
            predictions_ae = anomalyutils.get_ae_predicts(self.model, data)
            residuals_autoencoder = anomalyutils.get_ae_residuals(data, predictions_ae)
            self.UCL = residuals_autoencoder.quantile(0.99)
        elif self.model_name.upper() == "CONV_AE":
            X_conv_ae = utils.create_sequences(data, 60)
            self.model = self.conv_ae(X_conv_ae)
            predictions_conv_ae = anomalyutils.get_conv_ae_predicts(self.model, X_conv_ae)
            residuals_conv_ae = anomalyutils.get_conv_ae_residuals(X_conv_ae, predictions_conv_ae)
            self.UCL = residuals_conv_ae.quantile(0.999)
        elif self.model_name.upper() == "LSTM":
            X_lstm, y_lstm = utils.split_sequences(data, 5)
            self.model = self.lstm(X_lstm, y_lstm, f"{task_name}")
            self.model.load_weights(f"{task_name}.h5")
            predictions_lstm = anomalyutils.get_lstm_predicts(self.model, X_lstm)
            residuals_lstm = anomalyutils.get_lstm_residuals(y_lstm, predictions_lstm)
            self.UCL = residuals_lstm.quantile(0.99)
        elif self.model_name.upper() == "LSTM_AE":
            X_lstm_ae = utils.create_sequences(data, 10)
            self.model = self.lstm_ae(X_lstm_ae)
            predictions_lstm_ae = anomalyutils.get_lstm_ae_predicts(self.model, X_lstm_ae)
            residuals_lstm_ae = anomalyutils.get_lstm_ae_residuals(X_lstm_ae, predictions_lstm_ae)
            self.UCL = residuals_lstm_ae.quantile(0.99)
        elif self.model_name.upper() == "LSTM_VAE":
            X_lstm_vae = utils.create_sequences(data, 5)
            self.model = self.lstm_vae(X_lstm_vae)
            predictions_lstm_vae = anomalyutils.get_lstm_vae_predicts(self.model, X_lstm_vae)
            residuals_lstm_vae = anomalyutils.get_lstm_vae_residuals(X_lstm_vae, predictions_lstm_vae)
            self.UCL = residuals_lstm_vae.quantile(0.999)
        else:
            raise NotImplemented(f"{self.model_name} Not implemnted yet!")
        
    def transform(self, df_data, infoWriter=None):
        if self.model_name.upper() == "AE":
            data = df_data.to_numpy()
            predictions_ae = anomalyutils.get_ae_predicts(self.model, data)
            residuals_autoencoder = anomalyutils.get_ae_residuals(data, predictions_ae)
            df_final = pd.DataFrame(pd.Series(residuals_autoencoder.values, index=df_data.index).fillna(0)).rename(columns={0:f"scores"})
            df_final["predicted_anomaly"] = (df_final["scores"] > (3/2)*self.UCL).astype(int)
            if self.capture_info:
                infoWriter.scores_ucls_anomalies = df_final
            
        elif self.model_name.upper() == "CONV_AE":
            data = df_data.to_numpy()
            X_conv_ae = utils.create_sequences(data, 60)
            predictions_conv_ae = anomalyutils.get_conv_ae_predicts(self.model, X_conv_ae)
            residuals_conv_ae = anomalyutils.get_conv_ae_residuals(X_conv_ae, predictions_conv_ae)
            df_final = utils.get_actual_scores_for_windows(residuals_conv_ae, df_data, X_conv_ae, 60, self.UCL, "scores", "predicted_anomaly")
            
            if self.capture_info:
                infoWriter.scores_ucls_anomalies = df_final

        elif self.model_name.upper() == "LSTM":
            X_all_rotated = df_data.to_numpy()
            X_lstm, y_lstm = utils.split_sequences(X_all_rotated, 5)
            predictions_lstm = anomalyutils.get_lstm_predicts(self.model, X_lstm)
            residuals_lstm = anomalyutils.get_lstm_residuals(y_lstm, predictions_lstm)
            prediction_labels_lstm = pd.DataFrame(pd.Series(residuals_lstm.values, index=df_data[5:].index).fillna(0)).rename(columns={0:f"scores"})
            df_to_append = pd.DataFrame(pd.Series(0, index=df_data[:5].index).fillna(0)).rename(columns={0:f"scores"})
            df_final = pd.concat([df_to_append, prediction_labels_lstm], ignore_index=False)
            df_final["predicted_anomaly"] = (df_final["scores"] > (3/2) * self.UCL).astype(int)
            if self.capture_info:
                infoWriter.scores_ucls_anomalies = df_final

        elif self.model_name.upper() == "LSTM_AE":
            X_all_rotated = df_data.to_numpy()
            X_lstm_ae = utils.create_sequences(X_all_rotated, 10)
            predictions_lstm_ae = anomalyutils.get_lstm_ae_predicts(self.model, X_lstm_ae)
            residuals_lstm_ae = anomalyutils.get_lstm_ae_residuals(X_lstm_ae, predictions_lstm_ae)
            df_final = utils.get_actual_scores_for_windows(residuals_lstm_ae, df_data, X_lstm_ae, 10, self.UCL, "scores", "predicted_anomaly")
            
            if self.capture_info:
                infoWriter.scores_ucls_anomalies = df_final
            
        elif self.model_name.upper() == "LSTM_VAE":
            X_all_rotated = df_data.to_numpy()
            X_lstm_vae = utils.create_sequences(X_all_rotated, 5)
            predictions_lstm_vae = anomalyutils.get_lstm_vae_predicts(self.model, X_lstm_vae)
            residuals_lstm_vae = anomalyutils.get_lstm_vae_residuals(X_lstm_vae, predictions_lstm_vae)
            df_final = utils.get_actual_scores_for_windows(residuals_lstm_vae, df_data, X_lstm_vae, 5, self.UCL, "scores", "predicted_anomaly")
            
            if self.capture_info:
                infoWriter.scores_ucls_anomalies = df_final
            
        else:
            raise NotImplemented(f"{self.model_name} Not implemnted yet!")
        
        return df_final
