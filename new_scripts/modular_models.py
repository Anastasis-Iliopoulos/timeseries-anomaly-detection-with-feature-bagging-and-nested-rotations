from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
import utils
import basemodels
import anomalyutils
import pandas as pd
import rotation_utils
import bagging_utils
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

autoencoder = PipelineModel()
autoencoder.apply_standard_scaling(())
autoencoder.apply_nested_rotations((K, fraction))
autoencoder.apply_feature_bagging(())
autoencoder.apply_model(())


class PipelineModel():
    """Autoencoder based model"""
    def __init__(self, model_name):
        self.pipeline_actions = []
        self.pipeline_args = []
        self.model_name = model_name
        self.model = None
        self.UCL = -np.inf
    
    def pipeline_fit(self, data):
        # raise ValueError("NOT IMPLEMENTED YET")
        for action, args in zip(self.pipeline_actions, self.pipeline_args):
            if len(args)==0:
                action.fit(data)    
            else:
                action.fit(data, *args)
    
    def pipeline_transform(self, data):
        # raise ValueError("NOT IMPLEMENTED YET")
        tmp_data = data.copy()
        for action, args in self.pipeline_actions:
            tmp_data = action.transform(tmp_data)
        
        return tmp_data
            
    def apply_standard_scaling(self, tuple_args):
        StSc = StandardScaler()
        self.pipeline_actions.append(StSc)
        self.pipeline_args.append(tuple_args)
    
    def apply_nested_rotations(self, tuple_args):
        Rot = rotation_utils.Rotator()
        self.pipeline_actions.append(Rot)
        self.pipeline_args.append(tuple_args)
    
    def apply_feature_bagging(self, tuple_args):
        FBag = bagging_utils.FeatureBagger()
        self.pipeline_actions.append(FBag)
        self.pipeline_args.append(tuple_args)
    
    def apply_model(self, data):
        if self.model_name.upper() == "AUTOENCODER":
            autoencoder_model = basemodels.autoencoder(data)
            predictions_ae = anomalyutils.get_ae_predicts(autoencoder_model, data)
            residuals_autoencoder = anomalyutils.get_ae_residuals(data, predictions_ae)
            self.UCL = residuals_autoencoder.quantile(0.99)
        elif self.model_name.upper() == "AUTOENCODER":
            autoencoder_model = basemodels.autoencoder(data)
            predictions_ae = anomalyutils.get_ae_predicts(autoencoder_model, data)
            residuals_autoencoder = anomalyutils.get_ae_residuals(data, predictions_ae)
            self.UCL = residuals_autoencoder.quantile(0.99)
        elif self.model_name.upper() == "AUTOENCODER":
            autoencoder_model = basemodels.autoencoder(data)
            predictions_ae = anomalyutils.get_ae_predicts(autoencoder_model, data)
            residuals_autoencoder = anomalyutils.get_ae_residuals(data, predictions_ae)
            self.UCL = residuals_autoencoder.quantile(0.99)
        elif self.model_name.upper() == "AUTOENCODER":
            autoencoder_model = basemodels.autoencoder(data)
            predictions_ae = anomalyutils.get_ae_predicts(autoencoder_model, data)
            residuals_autoencoder = anomalyutils.get_ae_residuals(data, predictions_ae)
            self.UCL = residuals_autoencoder.quantile(0.99)
        elif self.model_name.upper() == "AUTOENCODER":
            autoencoder_model = basemodels.autoencoder(data)
            predictions_ae = anomalyutils.get_ae_predicts(autoencoder_model, data)
            residuals_autoencoder = anomalyutils.get_ae_residuals(data, predictions_ae)
            self.UCL = residuals_autoencoder.quantile(0.99)
        else:
            raise NotImplemented(f"{self.model_name} not implemented yet")


    X_train = df[:400].drop(['anomaly', 'changepoint'], axis=1)
    StSc = StandardScaler()
    StSc.fit(X_train)
    X_ae_normalized = StSc.transform(X_train)
    Rot = rotation_utils.Rotator()
    Rot.fit(X_ae_normalized, K, fraction)
    X_ae_df = pd.DataFrame(Rot.transform(X_ae_normalized), columns=X_train.columns.to_list())
    X_ae = X_ae_df[ds].to_numpy()
    autoencoder_model = basemodels.autoencoder(X_ae)
    predictions_ae = anomalyutils.get_ae_predicts(autoencoder_model, X_ae)
    residuals_autoencoder = anomalyutils.get_ae_residuals(X_ae, predictions_ae)
    UCL_autoencoder = residuals_autoencoder.quantile(0.99)
    X_all = df.drop(['anomaly','changepoint'], axis=1)
    X_ae_normalized = StSc.transform(X_all)
    X_ae_df = pd.DataFrame(Rot.transform(X_ae_normalized), columns=X_all.columns.to_list())
    X_ae = X_ae_df[ds].to_numpy()
    predictions_ae = anomalyutils.get_ae_predicts(autoencoder_model, X_ae)
    residuals_autoencoder = anomalyutils.get_ae_residuals(X_ae, predictions_ae)
    prediction_labels_autoencoder = pd.DataFrame(pd.Series(residuals_autoencoder.values, index=df.index).fillna(0)).rename(columns={0:f"anomaly_by_autoencoder_rotation_task_{task_name}_score"})
    prediction_labels_autoencoder[f"anomaly_by_autoencoder_rotation_task_{task_name}_ucl"] = 3/2*UCL_autoencoder
    return prediction_labels_autoencoder    