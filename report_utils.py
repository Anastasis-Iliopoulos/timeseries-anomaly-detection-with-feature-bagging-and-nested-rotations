import os
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score
import math

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

def plot_single_confusion_matrix(df, y_predicted_col, y_actual_col, family_model_name, extra_info=None, save_image_path=None):
    sns.set(font_scale=1)
    
    px = 1/plt.rcParams['figure.dpi']

    fig = plt.figure()
    fig.patch.set_facecolor((1,1,1,1))
    fig.patch.set_alpha(1.0)
    
    roc_number = roc_auc_score(df[y_actual_col], df[y_predicted_col])
    F1 = f1_score(df[y_actual_col], df[y_predicted_col])

    custom_title = f"{str(family_model_name)}\nAUC: {str(roc_number)}\nF1 Score: {str(F1)}"
    if extra_info is not None:
        custom_title = custom_title + "\n" + extra_info

    confusion_matrix = pd.crosstab(df[y_actual_col], df[y_predicted_col], rownames=['Actual'], colnames=['Predicted'])
    sns.set(font_scale=3.0)
    ax = sns.heatmap(confusion_matrix, cmap="Blues", cbar_kws={'shrink': 0.6}, annot=True, fmt='g')
    # ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_title(custom_title, fontdict={'size':'21'})
    ax.set_xlabel('Predicted', fontdict={'size':'21'})
    ax.set_ylabel('Actual', fontdict={'size':'21'})
    ax.tick_params(axis='both', which='major', labelsize=21)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=21)
    
    if save_image_path is not None:
        image_path = str(save_image_path)
        # remove white space
        plt.savefig(image_path, bbox_inches='tight')
    plt.show()

    sns.set(font_scale=1)

    return 1

def get_x_y(n):
   x = -1
   y = -1
   if int(math.sqrt(n)) == math.sqrt(n):
      x = int(math.sqrt(n))+1
      y = int(math.sqrt(n))+1
      return x, y
   
   m = n
   while int(math.sqrt(m)) == math.sqrt(m):
      m-=1

   x = int(math.sqrt(m))+1
   y = int(math.sqrt(m))+1
   return x, y

def plot_N_confusion_matrix(df, list_of_predicted_cols, y_actual_col, list_of_titles, family_model_name, extra_info=None, save_image_path=None):
    px = 1/plt.rcParams['figure.dpi']

    fig = plt.figure(figsize=(1920*px, 1080*px))
    # fig = plt.figure(figsize = (24,8))
    fig.patch.set_facecolor((1,1,1,1))
    fig.patch.set_alpha(1.0)

    x, y = get_x_y(len(list_of_predicted_cols))

    # fig.subplots_adjust(top=0.88)
    fig.subplots_adjust(hspace=1.2, wspace=0.5)

    for i,col,sub_title in enumerate(zip(list_of_predicted_cols,list_of_titles)):
        ax = fig.add_subplot(x, y, i+1)

        roc_number = roc_auc_score(df[y_actual_col], df[col])
        F1 = f1_score(df[y_actual_col], df[col])

        custom_title = f"{str(sub_title)}\nAUC: {str(roc_number)}\nF1 Score: {str(F1)}"
        
        confusion_matrix = pd.crosstab(df[y_actual_col], df[col], rownames=['Actual'], colnames=['Predicted'])
        g = sns.heatmap(confusion_matrix, annot=True, fmt='g', ax = ax).set_title(custom_title)

    super_title = f"{str(family_model_name)}"
    if extra_info is not None:
        super_title = super_title + "\n" + extra_info

    fig.suptitle(super_title) # or plt.suptitle(super_title)
    fig.tight_layout()
    if save_image_path is not None:
        image_path = str(save_image_path)
        # remove white space
        plt.savefig(image_path, bbox_inches='tight')
    
    plt.show()