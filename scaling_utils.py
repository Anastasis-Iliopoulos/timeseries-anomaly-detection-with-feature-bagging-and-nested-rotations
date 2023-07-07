import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class STDScaler():
    """FeatureBagger"""
    def __init__(self):
        self.StSc = StandardScaler()

    def fit(self, df_data):
        self.StSc.fit(df_data)
        return self

    def transform(self, df_data):
        transformed_data = self.StSc.transform(df_data)
        transformed_data = pd.DataFrame(transformed_data, columns=df_data.columns, index=df_data.index)
        return transformed_data