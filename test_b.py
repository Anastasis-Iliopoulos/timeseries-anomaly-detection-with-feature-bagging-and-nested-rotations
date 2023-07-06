import numpy as np

np.random.seed(0)

def get_list(a_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]):
    return list(np.random.choice(a_list, size=3, replace=False))