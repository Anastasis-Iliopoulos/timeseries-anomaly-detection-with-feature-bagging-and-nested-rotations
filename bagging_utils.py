import numpy as np
import pandas as pd

class FeatureBagger():
    """FeatureBagger"""
    def __init__(self, capture_info=False):
        if capture_info not in [False, True]:
            raise ValueError("capture_info should be of type bool either True or False")

        self.capture_info = capture_info
        self.if_fitted = False
        self.subset = []

    def fit(self, data):
        possible_values = [int(i) for i in range(int(np.floor(data.shape[1]/2)), data.shape[1], 1)]
        number_of_features = np.random.choice(possible_values, 1, replace=False)[0]
        self.subset = sorted(np.random.choice(data.shape[1], number_of_features, replace=False).tolist())
        return self

    def transform(self, df_data, infoWriter = None):
        data = df_data.to_numpy()
        transformed_data = data[:,self.subset]
        transformed_data = pd.DataFrame(transformed_data, columns=[df_data.columns.tolist()[i] for i in self.subset], index=df_data.index)
        
        if self.capture_info:
            infoWriter.fbsubset = self.subset
        
        return transformed_data