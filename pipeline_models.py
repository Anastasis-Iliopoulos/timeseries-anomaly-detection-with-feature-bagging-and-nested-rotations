from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
import utils
import basemodels
import anomalyutils
import pandas as pd
import scaling_utils
import rotation_utils
import bagging_utils
import info_utils
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

class PipelineModel():
    """Autoencoder based model"""
    def __init__(self, capture_info, infoWriter_args):
        if capture_info not in [False, True]:
            raise ValueError("capture_info should be of type bool either True or False")
        
        self.pipeline_actions = []
        self.pipeline_args = []
        self.model = None
        self.capture_info = capture_info
    
    def pipeline_fit(self, df_data):
        tmp_data = df_data.copy(deep=True)
        for action, args in zip(self.pipeline_actions, self.pipeline_args):
            if len(args)==0:
                action.fit(tmp_data)
                tmp_data = action.transform(tmp_data)    
            else:
                action.fit(tmp_data, *args)
                tmp_data = action.transform(tmp_data)
    
    def pipeline_transform(self, df_data):
        tmp_data = df_data.copy(deep=True)
        info_writer = None 
        if self.capture_info:
            info_writer = info_utils.InfoWriter(*self.infoWriter_args)
        for action in self.pipeline_actions:
            tmp_data = action.transform(tmp_data, info_writer)
        if self.capture_info:
            info_writer.write_info()
        return tmp_data
            
    def apply_standard_scaling(self, tuple_args):
        StSc = scaling_utils.STDScaler()
        if self.capture_info:
            StSc = scaling_utils.STDScaler(True)
        self.pipeline_actions.append(StSc)
        self.pipeline_args.append(tuple_args)
    
    def apply_nested_rotations(self, tuple_args):
        Rot = rotation_utils.Rotator()
        if self.capture_info:
            Rot = rotation_utils.Rotator(True)
        self.pipeline_actions.append(Rot)
        self.pipeline_args.append(tuple_args)
    
    def apply_feature_bagging(self, tuple_args):
        FBag = bagging_utils.FeatureBagger()
        if self.capture_info:
            FBag = bagging_utils.FeatureBagger(True)
        self.pipeline_actions.append(FBag)
        self.pipeline_args.append(tuple_args)
    
    def apply_model(self, tuple_args):
        self.model = basemodels.AnomalyDetectionModel()
        if self.capture_info:
            self.model = basemodels.AnomalyDetectionModel(True)
        self.pipeline_actions.append(self.model)
        self.pipeline_args.append(tuple_args)
        