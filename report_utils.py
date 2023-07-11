import os
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm

class DFconstants():
    def __init__(self) -> None:
        self.parquet_suffix = 'parquet.gzip'
        self.NESTED_ROTATIONS = 'nested_rotations' 
        self.OTHER_INFO = 'other_info'
        self.PARTITION_SUFFIX = 'partition_'
        self.SCALINGS = 'scalings'
        self.SCORES_AND_ANOMALIES = 'scores_and_anomalies'
    def __str__(self) -> str:
        return f"""parquet_suffix = '.parquet.gzip'
NESTED_ROTATIONS = 'nested_rotations' 
OTHER_INFO = 'other_info'
PARTITION_SUFFIX = 'partition_'
SCALINGS = 'scalings'
SCORES_AND_ANOMALIES = 'scores_and_anomalies'"""
    def __repr__(self) -> str:
         return f"""parquet_suffix = '.parquet.gzip'
NESTED_ROTATIONS = 'nested_rotations' 
OTHER_INFO = 'other_info'
PARTITION_SUFFIX = 'partition_'
SCALINGS = 'scalings'
SCORES_AND_ANOMALIES = 'scores_and_anomalies'"""
         
             
def warning_formatter(msg, category, filename, lineno,   line=None):
        return f"WARNING: {msg}"

warnings.formatwarning = warning_formatter

def get_family_scores_and_anomalies(top_level_folder, model_family):
    warnings.warn('Index should not be repeated. Check your files')
    print('Index should not be repeated. Check your files')

    if top_level_folder is None:
        raise ValueError(f"Missing top_level_folder={top_level_folder} is invalid")
    elif top_level_folder == "":
        raise ValueError(f"Missing top_level_folder={top_level_folder}")
    
    while top_level_folder[-1] == "/":
        top_level_folder = top_level_folder[:-1]
    
    dfconstants = DFconstants()
    
    model_tasks = os.listdir(f"{top_level_folder}/{model_family}")

    model_task_dataframes = []
    for model_task_ind, model_task in tqdm(enumerate(model_tasks), desc="Model"):
        dataset_folders = os.listdir(f"{top_level_folder}/{model_family}/{model_task}")
        df_datasets = []
        for dataset in tqdm(dataset_folders, desc="Dataset"):
            scores_and_anomalies = f"{top_level_folder}/{model_family}/{model_task}/{dataset}/{dfconstants.SCORES_AND_ANOMALIES}.{dfconstants.parquet_suffix}"
            df_datasets.append(pd.read_parquet(scores_and_anomalies))
        model_task_df = pd.concat(df_datasets, ignore_index=False)
        model_task_df = model_task_df.rename(columns={"scores": f"scores_{model_task_ind}"})
        model_task_df = model_task_df.rename(columns={"predicted_anomaly": f"predicted_anomaly_{model_task_ind}"})
        model_task_dataframes.append(model_task_df)
    final_df = pd.concat(model_task_dataframes, ignore_index=False,axis=1)
    return final_df

def family_majority_voting(top_level_folder, model_family):
    df = get_family_scores_and_anomalies(top_level_folder, model_family)
    anomaly_cols = [col for col in df.columns if "anomaly" in col]
    df = df[anomaly_cols]
    for col in df.columns:
        df[col] = np.where(df[col] == 1 , 1, -1)
    df['predicted_anomaly_majority_voting'] = df.sum(axis=1)
    df['predicted_anomaly_majority_voting'] = np.where(df['predicted_anomaly_majority_voting'] < 1 , 0, 1)
    for col in df.columns:
        df[col] = np.where(df[col] == 1 , 1, 0)
    return df