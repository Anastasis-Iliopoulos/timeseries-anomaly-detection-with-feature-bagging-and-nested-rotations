import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class STDScaler():
    """FeatureBagger"""
    def __init__(self, capture_info=False):
        if capture_info not in [False, True]:
            raise ValueError("capture_info should be of type bool either True or False")
        
        self.capture_info = capture_info
        self.StSc = StandardScaler()

    def fit(self, df_data):
        self.StSc.fit(df_data)
        return self

    def transform(self, df_data, infoWriter=None):
        transformed_data = self.StSc.transform(df_data)
        transformed_data = pd.DataFrame(transformed_data, columns=df_data.columns, index=df_data.index)
        
        if (self.capture_info) and (infoWriter is not None):
            infoWriter.scvar = self.StSc.mean_
            infoWriter.scmean = self.StSc.scale_
            infoWriter.scscale = self.StSc.var_
            infoWriter.scalings = transformed_data
        return transformed_data