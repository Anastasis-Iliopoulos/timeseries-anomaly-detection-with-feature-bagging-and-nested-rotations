from pathlib import Path
from sklearn.preprocessing import StandardScaler
import utils
import basemodels
import anomalyutils
import pandas as pd


def ensemble_autoencoder(df, df_name, task_name):
    X_train = df[:400].drop(['anomaly', 'changepoint'], axis=1)
    StSc = StandardScaler()
    StSc.fit(X_train)
    X_ae = StSc.transform(X_train)
    autoencoder_model = basemodels.autoencoder(X_ae)
    predictions_ae = anomalyutils.get_ae_predicts(autoencoder_model, X_ae)
    residuals_autoencoder = anomalyutils.get_ae_residuals(X_ae, predictions_ae)
    UCL_autoencoder = residuals_autoencoder.quantile(0.99)
    X_all = df.drop(['anomaly','changepoint'], axis=1)
    X_ae = StSc.transform(X_all)
    predictions_ae = anomalyutils.get_ae_predicts(autoencoder_model, X_ae)
    residuals_autoencoder = anomalyutils.get_ae_residuals(X_ae, predictions_ae)

    prediction_labels_autoencoder = pd.DataFrame(pd.Series(residuals_autoencoder.values, index=df.index).fillna(0)).rename(columns={0:f"anomaly_by_autoencoder_fbagging_task_{task_name}_score"})
    prediction_labels_autoencoder[f"anomaly_by_autoencoder_fbagging_task_{task_name}_ucl"] = 3/2*UCL_autoencoder

    return prediction_labels_autoencoder


def ensemble_conv_ae(df, df_name, task_name):
    X_train = df[:400].drop(['anomaly', 'changepoint'], axis=1)
    StSc = StandardScaler()
    StSc.fit(X_train)
    X_conv_ae = utils.create_sequences(StSc.transform(X_train), 60)
    conv_ae_model = basemodels.conv_ae(X_conv_ae)
    predictions_conv_ae = anomalyutils.get_conv_ae_predicts(conv_ae_model, X_conv_ae)
    residuals_conv_ae = anomalyutils.get_conv_ae_residuals(X_conv_ae, predictions_conv_ae)
    UCL_conv_ae = residuals_conv_ae.quantile(0.999)
    X_all = df.drop(['anomaly','changepoint'], axis=1)
    X_conv_ae = utils.create_sequences(StSc.transform(X_all), 60)
    predictions_conv_ae = anomalyutils.get_conv_ae_predicts(conv_ae_model, X_conv_ae)
    residuals_conv_ae = anomalyutils.get_conv_ae_residuals(X_conv_ae, predictions_conv_ae)
    
    
    df_final = utils.get_actual_scores_for_windows(residuals_conv_ae, df, X_conv_ae, 60, UCL_conv_ae, f"anomaly_by_conv_ae_fbagging_task_{task_name}_score", f"anomaly_by_conv_ae_fbagging_task_{task_name}_ucl")
        
    return df_final


def ensemble_lstm(df, df_name, task_name):
    X_train = df[:400].drop(['anomaly', 'changepoint'], axis=1)
    StSc = StandardScaler()
    StSc.fit(X_train)
    X_lstm, y_lstm = utils.split_sequences(StSc.transform(X_train), 5)
    lstm_model = basemodels.lstm(X_lstm, y_lstm, f"{task_name}")
    lstm_model.load_weights(f"lstm_{task_name}.h5")
    predictions_lstm = anomalyutils.get_lstm_predicts(lstm_model, X_lstm)
    residuals_lstm = anomalyutils.get_lstm_residuals(y_lstm, predictions_lstm)
    UCL_lstm = residuals_lstm.quantile(0.99)
    X_all = df.drop(['anomaly','changepoint'], axis=1)
    X_lstm, y_lstm = utils.split_sequences(StSc.transform(X_all), 5)
    predictions_lstm = anomalyutils.get_lstm_predicts(lstm_model, X_lstm)
    residuals_lstm = anomalyutils.get_lstm_residuals(y_lstm, predictions_lstm)

    prediction_labels_lstm = pd.DataFrame(pd.Series(residuals_lstm.values, index=df[5:].index).fillna(0)).rename(columns={0:f"anomaly_by_lstm_fbagging_task_{task_name}_score"})
    df_to_append = pd.DataFrame(pd.Series(0, index=df[:5].index).fillna(0)).rename(columns={0:f"anomaly_by_lstm_fbagging_task_{task_name}_score"})
    df_final = pd.concat([df_to_append, prediction_labels_lstm], ignore_index=False)
    df_final[f"anomaly_by_lstm_fbagging_task_{task_name}_ucl"] = 3/2*UCL_lstm
    
    return df_final


def ensemble_lstm_ae(df, df_name, task_name):
    X_train = df[:400].drop(['anomaly', 'changepoint'], axis=1)
    StSc = StandardScaler()
    StSc.fit(X_train)
    X_lstm_ae = utils.create_sequences(StSc.transform(X_train), 10)
    lstm_ae_model = basemodels.lstm_ae(X_lstm_ae)
    predictions_lstm_ae = anomalyutils.get_lstm_ae_predicts(lstm_ae_model, X_lstm_ae)
    residuals_lstm_ae = anomalyutils.get_lstm_ae_residuals(X_lstm_ae, predictions_lstm_ae)
    UCL_lstm_ae = residuals_lstm_ae.quantile(0.99)
    X_all = df.drop(['anomaly','changepoint'], axis=1)
    X_lstm_ae = utils.create_sequences(StSc.transform(X_all), 10)
    predictions_lstm_ae = anomalyutils.get_lstm_ae_predicts(lstm_ae_model, X_lstm_ae)
    residuals_lstm_ae = anomalyutils.get_lstm_ae_residuals(X_lstm_ae, predictions_lstm_ae)
    
    df_final = utils.get_actual_scores_for_windows(residuals_lstm_ae, df, X_lstm_ae, 10, UCL_lstm_ae, f"anomaly_by_lstm_ae_fbagging_task_{task_name}_score", f"anomaly_by_lstm_ae_fbagging_task_{task_name}_ucl")
    return df_final


def ensemble_lstm_vae(df, df_name, task_name):
    X_train = df[:400].drop(['anomaly', 'changepoint'], axis=1)
    StSc = StandardScaler()
    StSc.fit(X_train)
    X_lstm_vae = utils.create_sequences(StSc.transform(X_train), 5)
    lstm_vae_model = basemodels.lstm_vae(X_lstm_vae)
    predictions_lstm_vae = anomalyutils.get_lstm_vae_predicts(lstm_vae_model, X_lstm_vae)
    residuals_lstm_vae = anomalyutils.get_lstm_vae_residuals(X_lstm_vae, predictions_lstm_vae)
    UCL_lstm_vae = residuals_lstm_vae.quantile(0.999)
    X_all = df.drop(['anomaly','changepoint'], axis=1)
    X_lstm_vae = utils.create_sequences(StSc.transform(X_all), 5)
    predictions_lstm_vae = anomalyutils.get_lstm_vae_predicts(lstm_vae_model, X_lstm_vae)
    residuals_lstm_vae = anomalyutils.get_lstm_vae_residuals(X_lstm_vae, predictions_lstm_vae)

    df_final = utils.get_actual_scores_for_windows(residuals_lstm_vae, df, X_lstm_vae, 5, UCL_lstm_vae, f"anomaly_by_lstm_vae_fbagging_task_{task_name}_score", f"anomaly_by_lstm_vae_fbagging_task_{task_name}_ucl")
    return df_final


def create_folder_if_not_exist(SAVE_PATH):
    parts_SAVE_PATH = SAVE_PATH.split("/")
    SAVE_FOLDER = "/".join(parts_SAVE_PATH[0:-1])
    # Dont do anything with name
    SAVE_NAME = parts_SAVE_PATH[-1]

    Path(SAVE_FOLDER).mkdir(parents=True, exist_ok=True)