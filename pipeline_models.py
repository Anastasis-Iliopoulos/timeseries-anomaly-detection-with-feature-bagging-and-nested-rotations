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
    def __init__(self):
        self.pipeline_actions = []
        self.pipeline_args = []
        self.model = None
    
    def pipeline_fit(self, data):
        # raise ValueError("NOT IMPLEMENTED YET")
        for action, args in zip(self.pipeline_actions, self.pipeline_args):
            if len(args)==0:
                action.fit(data)    
            else:
                action.fit(data, *args)
    
    def pipeline_transform(self, data):
        # raise ValueError("NOT IMPLEMENTED YET")
        for action in self.pipeline_actions:
            tmp_data = action.transform(data)
        
        return tmp_data
            
    def apply_standard_scaling(self, tuple_args):
        StSc = scaling_utils.STDScaler()
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
    
    def apply_model(self, tuple_args):
        self.model = basemodels.AnomalyDetectionModel()
        self.pipeline_actions.append(self.model)
        self.pipeline_args.append(tuple_args)
        