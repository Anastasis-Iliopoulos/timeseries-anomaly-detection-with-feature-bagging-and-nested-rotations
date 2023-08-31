
from tabnanny import verbose
import pandas as pd
import numpy as np

def get_ae_residuals(data_true, data_predicted):
    return pd.DataFrame(data_true - data_predicted).abs().sum(axis=1)

def get_conv_ae_residuals(data_true, data_predicted):
    return pd.Series(np.sum(np.mean(np.abs(data_true - data_predicted), axis=1), axis=1))

def get_conv_ae_residuals_mse(data_true, data_predicted):
    return pd.Series(np.sum(np.mean(np.power(np.abs(data_true - data_predicted),2), axis=1), axis=1))

def get_lstm_residuals(data_true, data_predicted):
    return pd.DataFrame(data_true - data_predicted).abs().sum(axis=1)

def get_lstm_ae_residuals(data_true, data_predicted):
    return pd.Series(np.sum(np.mean(np.abs(data_true - data_predicted), axis=1), axis=1))

def get_lstm_vae_residuals(data_true, data_predicted):
    return pd.Series(np.sum(np.mean(np.abs(data_true - data_predicted), axis=1), axis=1))


def get_ae_predicts(model, data):
    return model.predict(data,verbose=1)

def get_conv_ae_predicts(model, data, chunk_step=None):
    if chunk_step is None:
        return model.predict(data,verbose=1)
    if chunk_step<=0:
        return model.predict(data,verbose=1)
     
    predictions_list = []
    if (chunk_step is not None) and (chunk_step>0):
        for i in range(0,len(data),chunk_step):
            predictions_list.append(model.predict(data[i:i+chunk_step],verbose=1))
    predictions = np.concatenate(predictions_list, axis=0)
    return predictions

def get_lstm_predicts(model, data):
    return model.predict(data,verbose=1)

def get_lstm_ae_predicts(model, data):
    return model.predict(data,verbose=1)

def get_lstm_vae_predicts(model, data):
    return model.predict(data,verbose=1)