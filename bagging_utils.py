import numpy as np
import pandas as pd

class FeatureBagger():
    """FeatureBagger"""
    def __init__(self):
        self.if_fitted = False
        self.subset = []

    def fit(self, data):
        possible_values = [int(i) for i in range(int(np.floor(data.shape[1]/2)), data.shape[1], 1)]
        number_of_features = np.random.choice(possible_values, 1, replace=False)[0]
        self.subset = sorted(np.random.choice(data.shape[1], number_of_features, replace=False).tolist())
        return self

    def transform(self, df_data):
        data = df_data.to_numpy()
        transformed_data = data[:,self.subset]
        transformed_data = pd.DataFrame(transformed_data, columns=[df_data.columns.tolist()[i] for i in self.subset], index=df_data.index)
        return transformed_data