import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

class Rotator():
    """Rotator"""
    def __init__(self, capture_info=False):
        if capture_info not in [False, True]:
            raise ValueError("capture_info should be of type bool either True or False")
        
        self.capture_info = capture_info
        self.if_fitted = False
        self.subsets = []
        self.rotation_matricies = []
        self.varK=None
        self.varfraction=None

# StSc = StandardScaler()
# StSc.fit(X_train)
# X_ae_normalized = StSc.transform(X_train)

    def reverse_permutation(self, list_of_subs):
        permutation = []
        for subset in list_of_subs:
            for i in subset:
                permutation.append(i)
        permutation_indexing = {}
        for i in permutation:
            permutation_indexing[permutation[i]] = i
        reverse_permutation_list = []
        for i in range(len(permutation_indexing)):
            reverse_permutation_list.append(permutation_indexing[i])

        return reverse_permutation_list

    def get_partitioned_data(self, data, number_of_subsets):
        number_of_features = data.shape[1]
        if number_of_features//number_of_subsets < 2:
            raise ValueError(f"Cannot partition {number_of_features} features into {number_of_subsets} subsets. Number_of_subsets/Number_of_features should be greated than 2.")
        random_permutation_feature_list = np.random.choice(number_of_features, number_of_features, replace=False).tolist()
        K = number_of_subsets
        N = int(len(random_permutation_feature_list)//K)
        subs = []
        for i in range(K):
            a_set = []
            for j in range(i*N,i*N+N):
                a_set.append(random_permutation_feature_list[j])
            subs.append(a_set)
        for i in range(K*N,len(random_permutation_feature_list)):
            subs[i-(K*N-1)-1].append(random_permutation_feature_list[i])

        partitioned_data = []
        for partition in subs:
            partitioned_data.append(np.array([data[:,i] for i in partition]).T)

        return partitioned_data, subs

    def get_random_data(self, partition, fraction):
        random_data = np.take(partition, sorted(np.random.choice(int(partition.shape[0]), size=round(fraction*partition.shape[0]), replace=False).tolist()), axis=0)
        return random_data

    def get_rotation_matrix(self, data):
        pca = PCA()
        pca.fit(data)
        rotation_matrix = pca.components_
        return rotation_matrix

    def fit(self, df_data, K=2, fraction=0.75):
        self.varK=K
        self.varfraction=fraction
        data = df_data.to_numpy()
        a_partition, permutation = self.get_partitioned_data(data, K)
        self.subsets = permutation
        for partition in a_partition:
            random_data = self.get_random_data(partition, fraction)
            rotation_matrix = self.get_rotation_matrix(random_data)
            self.rotation_matricies.append(rotation_matrix)
        return self

    def transform(self, df_data, infoWriter=None):
        data = df_data.to_numpy()
        transformed_partitions = []
        for sub, rotation_matrix in zip(self.subsets, self.rotation_matricies):
            partition = np.array([data[:,i] for i in sub]).T
            transformed_partitions.append(np.dot(partition, rotation_matrix.T))

        transformed_data_unordered = np.concatenate(transformed_partitions, axis=1)
        reverse_perm = self.reverse_permutation(self.subsets)
        transformed_data = transformed_data_unordered[:,reverse_perm]
        transformed_data = pd.DataFrame(transformed_data, columns=df_data.columns.tolist(), index=df_data.index)
        
        if (self.capture_info) and (infoWriter is not None):
            infoWriter.nrpartitions = self.subsets
            infoWriter.nrmatrices = self.rotation_matricies
            infoWriter.nrK = self.varK
            infoWriter.nrfranction = self.varfraction
            infoWriter.nested_rotations = transformed_data
        
        return transformed_data
