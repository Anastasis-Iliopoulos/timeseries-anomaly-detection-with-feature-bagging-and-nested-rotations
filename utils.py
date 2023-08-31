import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import roc_auc_score, f1_score
import itertools
from IPython.display import display

def get_files_and_names():
    # benchmark files checking
    all_files=[]
    all_files_name = []
    for root, dirs, files in os.walk("./data/"):
        for file in files:
            if file.endswith(".csv"):
                all_files.append(os.path.join(root, file))
                all_files_name.append(root.split("/")[-1] + "_" + file.replace(".", "_"))
    return all_files, all_files_name


def get_anomaly_data_and_names():
    all_files, all_files_name = get_files_and_names()
    # datasets with anomalies loading
    list_of_df = [pd.read_csv(file, 
                            sep=';', 
                            index_col='datetime', 
                            parse_dates=True) for file in all_files if 'anomaly-free' not in file]
    list_of_names = [file for file in all_files_name if 'anomaly-free' not in file]
    return list_of_df, list_of_names

def get_anomaly_free_data_and_names():
    all_files, all_files_name = get_files_and_names()
    anomaly_free_df = pd.read_csv([file for file in all_files if 'anomaly-free' in file][0], 
                                sep=';', 
                                index_col='datetime', 
                                parse_dates=True)
    list_of_names = [file for file in all_files_name if 'anomaly-free' in file]
    return anomaly_free_df, list_of_names

def get_files():
    # benchmark files checking
    all_files=[]
    for root, dirs, files in os.walk("./data/"):
        for file in files:
            if file.endswith(".csv"):
                all_files.append(os.path.join(root, file))

    return all_files


def get_anomaly_data():
    all_files = get_files()
    # datasets with anomalies loading
    list_of_df = [pd.read_csv(file, 
                            sep=';', 
                            index_col='datetime', 
                            parse_dates=True) for file in all_files if 'anomaly-free' not in file]

    return list_of_df

def get_anomaly_free_data():
    all_files = get_files()
    anomaly_free_df = pd.read_csv([file for file in all_files if 'anomaly-free' in file][0], 
                                sep=';', 
                                index_col='datetime', 
                                parse_dates=True)
    return anomaly_free_df

def create_sequences(values, time_steps):
        output = []
        for i in range(len(values) - time_steps + 1):
            output.append(values[i : (i + time_steps)])
        return np.stack(output)

def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def get_anomalies_labels_from_continuous_windows(original_X_data, residuals, N_STEPS, UCL, df_index):
    anomalous_data = residuals > (3/2 * UCL)
    anomalous_data_indices = []
    for data_idx in range(N_STEPS - 1, len(original_X_data) - N_STEPS + 1):
        if np.all(anomalous_data[data_idx - N_STEPS + 1 : data_idx]):
            anomalous_data_indices.append(data_idx)
    
    prediction = pd.Series(data=0, index=df_index)
    prediction.iloc[anomalous_data_indices] = 1
    return prediction

def get_scores_from_residuals(original_X_data, residuals, N_STEPS, df_index):

    prediction = pd.Series(data=0, index=df_index)

    for data_idx in range(N_STEPS - 1, len(original_X_data) - N_STEPS + 1):
        score = residuals[data_idx]
        prediction.iloc[data_idx] = score

    return prediction

def get_actual_scores_for_windows_v4(residuals, df, N_STEPS, UCL, name_of_score_col, name_of_anomaly_col, ucl_multiplier):
    anomalous_data = pd.DataFrame(pd.Series(residuals.values, index=df[N_STEPS-1:].index).fillna(0)).rename(columns={0:f"cur_score"})
    data_to_append = pd.DataFrame(pd.Series(pd.NA, index=df[:N_STEPS-1].index)).rename(columns={0:f"cur_score"})
    all_data = pd.concat([data_to_append, anomalous_data], axis=0)
    all_data["previous_scores"] = all_data.shift(1)
    all_data[name_of_score_col] = all_data["previous_scores"].rolling(N_STEPS-1).min()
    all_data = all_data.fillna(0)
    all_data[name_of_anomaly_col] = (all_data[name_of_score_col] > ((ucl_multiplier)*UCL)).astype(int)
    final_df = all_data[[name_of_score_col, name_of_anomaly_col]]
    return final_df

def get_actual_scores_for_windows(residuals, df, X_df, N_STEPS, UCL, name_of_score_col, name_of_anomaly_col, ucl_multiplier):
    anomalous_data = pd.DataFrame(pd.Series(residuals.values, index=df[N_STEPS-1:].index).fillna(0)).rename(columns={0:f"cur_score"})

    for ind in range(1,N_STEPS):
        anomalous_data[f"cur_score_{ind}"] = anomalous_data["cur_score"].shift(ind)

    anomalous_data = anomalous_data.reset_index(drop=True)
    anomalous_data = anomalous_data.iloc[:len(X_df) - N_STEPS + 1]

    data_to_append = pd.DataFrame(pd.Series(0, index=df.index).fillna(0)).rename(columns={0:f"cur_score_init"}).reset_index(drop=True)
    all_data = data_to_append.join(anomalous_data)
    all_data = all_data.drop(["cur_score_init", "cur_score"], axis=1)
    all_data["min_score"] = all_data.min(axis=1, skipna=False).fillna(0)
    final_df = pd.DataFrame(pd.Series(all_data["min_score"].values, index=df.index).fillna(0)).rename(columns={0: name_of_score_col})
    final_df[name_of_anomaly_col] = (final_df[name_of_score_col] > ((ucl_multiplier)*UCL)).astype(int)
    final_df = final_df[[name_of_score_col, name_of_anomaly_col]]
    return final_df

def get_actual_scores_for_windows_2(residuals, df, X_df, N_STEPS, name_of_score_col):
    anomalous_data = pd.DataFrame(pd.Series(residuals.values, index=df[N_STEPS-1:].index).fillna(0)).rename(columns={0:f"cur_score"})

    for ind in range(1,N_STEPS):
        anomalous_data[f"cur_score_{ind}"] = anomalous_data["cur_score"].shift(ind)

    anomalous_data = anomalous_data.reset_index(drop=True)
    anomalous_data = anomalous_data.iloc[:len(X_df) - N_STEPS + 1]

    data_to_append = pd.DataFrame(pd.Series(0, index=df.index).fillna(0)).rename(columns={0:f"cur_score_init"}).reset_index(drop=True)
    all_data = data_to_append.join(anomalous_data)
    all_data = all_data.drop(["cur_score_init", "cur_score"], axis=1)
    all_data["min_score"] = all_data.min(axis=1, skipna=False).fillna(0)
    all_data_with_anomalies = pd.DataFrame(pd.Series(all_data["min_score"].values, index=df.index).fillna(0)).rename(columns={0: name_of_score_col})
    final_df = all_data_with_anomalies[[name_of_score_col]]
    return final_df

def get_scores_scoring_with_current_window(original_X_data, residuals, N_STEPS, df_index):
    
    prediction = pd.Series(data=0, index=df_index)
    
    for data_idx in range(N_STEPS - 1, len(original_X_data)):
        score = residuals[data_idx]
        prediction.iloc[data_idx] = score

    return prediction

def get_scores_scoring_with_mean_steps_windows(original_X_data, residuals, N_STEPS, df_index):
    
    prediction = pd.Series(data=0, index=df_index)

    for data_idx in range(N_STEPS - 1, len(original_X_data) - N_STEPS + 1):
        score = np.mean(residuals[data_idx - N_STEPS + 1 : data_idx])
        prediction.iloc[data_idx] = score
    
    return prediction

def plot_single_confusion_matrix_forthesis(df, y_actual, y_predicted, save_image=None):
    px = 1/plt.rcParams['figure.dpi']

    fig = plt.figure()
    fig.patch.set_facecolor((1,1,1,1))
    fig.patch.set_alpha(1.0)
    
    roc_number = roc_auc_score(df[y_actual], df[y_predicted])

    model_title = y_predicted.replace("anomaly_by_","")
    if model_title == "autoencoder":
        model_title = "Autoencoder\n "
    elif model_title == "conv_ae":
        model_title = "Convolutional Autoencoder\n "
    elif model_title == "lstm":
        model_title = "LSTM\n "
    elif model_title == "lstm_ae":
        model_title = "LSTM Autoencoder\n "
    elif model_title == "lstm_vae":
        model_title = "LSTM Variational Autoencoder\n "
    elif model_title == "logistic_regression":
        model_title = "Logistic Regression\n "
    elif model_title == "ensemble_ensemble_autoencoder":
        model_title = "Feature Bagging \nAutoencoder\n "
    elif model_title == "ensemble_ensemble_conv_ae":
        model_title = "Feature Bagging \nConvolutional Autoencoder\n "
    elif model_title == "ensemble_ensemble_lstm":
        model_title = "Feature Bagging \nLSTM\n "
    elif model_title == "ensemble_ensemble_lstm_ae":
        model_title = "Feature Bagging \nLSTM Autoencoder\n "
    elif model_title == "ensemble_ensemble_lstm_vae":
        model_title = "Feature Bagging LSTM \nVariational Autoencoder\n " 
    elif model_title == "ensemble_ensemble_rotation_autoencoder":
        model_title = "Feature Bagging with Rotation \nAutoencoder\n " 
    elif model_title == "ensemble_ensemble_rotation_conv_ae":
        model_title = "Feature Bagging with Rotation \nConvolutional Autoencoder\n " 
    elif model_title == "ensemble_ensemble_rotation_lstm":
        model_title = "Feature Bagging with Rotation \nLSTM\n " 
    elif model_title == "ensemble_ensemble_rotation_lstm_ae":
        model_title = "Feature Bagging with Rotation \nLSTM Autoencoder\n " 
    elif model_title == "ensemble_ensemble_rotation_lstm_vae":
        model_title = "Feature Bagging with Rotation \nVariational Autoencoder\n " 
    elif model_title == "voting":
        model_title = "Majority Voting\n " 
    F1 = f1_score(df[y_actual], df[y_predicted])
    custom_title = f"{model_title}\nAUC: {roc_number}\nF1 Score: {F1}"
    confusion_matrix = pd.crosstab(df[y_actual], df[y_predicted], rownames=['Actual'], colnames=['Predicted'])
    sns.set(font_scale=3.0)
    ax = sns.heatmap(confusion_matrix, cmap="Blues", cbar_kws={'shrink': 0.6}, annot=True, fmt='g')
    # ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_title(model_title, fontdict={'size':'21'})
    ax.set_xlabel('Predicted', fontdict={'size':'21'})
    ax.set_ylabel('Actual', fontdict={'size':'21'})
    ax.tick_params(axis='both', which='major', labelsize=21)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=21)
    
    if save_image is not None:
        image_path = str(save_image)
        # remove white space
        plt.savefig(image_path, bbox_inches='tight')
    plt.show()

def print_latex_code(df):
    rows = []
    for i, row in df.iterrows():
        
        row_string = f'        {row["Model Title"]} & {row["F1 Score"]} & {row["AUC"]} \\\\\n        \\hline'
        rows.append(row_string)
    latex_string = f"""
\\begin{{table}}[htbp!]
    \\centering
    \\begin{{tabular}}{{|l|c|c|}}
        \\hline
        Model Title & F1 Score & AUC \\\\
        \\hline\n
"""
    for tmp_row_string in rows:
        latex_string = latex_string+tmp_row_string+"\n"

    ending_string=f"""    \\end{{tabular}}
    \\label{{tab:statistics}}
    \\caption{{Table of Statistics}}
\\end{{table}}"""

    latex_string = latex_string+ending_string
    print(latex_string)
    return latex_string


def print_metrics_forthesis_and_dfs(path_to_dfs, order_by_column):
    df = pd.read_csv(path_to_dfs, sep=',')
    cols = [cl for cl in df.columns.to_list() if "anomaly_by_" in cl]

    data_ae = []
    data_conv_ae = []
    data_lstm = []
    data_lstm_ae = []
    data_lstm_vae = []
    data_voting = []
    data_regressor = []
    for cl in cols:
        model_title, F1, roc_number = print_metrics_forthesis(df, "anomaly", cl)
        if ("Autoencoder" in model_title) and (("LSTM" not in model_title) and ("Convolutional" not in model_title)):
            data_ae.append((model_title, F1, roc_number))
            if "Feature" not in model_title:
                data_voting.append((model_title, F1, roc_number))
                data_regressor.append((model_title, F1, roc_number))
        elif ("LSTM" in model_title) and ("Autoencoder" not in model_title):
            data_lstm.append((model_title, F1, roc_number))
            if "Feature" not in model_title:
                data_voting.append((model_title, F1, roc_number))
                data_regressor.append((model_title, F1, roc_number))
        elif "LSTM Autoencoder" in model_title:
            data_lstm_ae.append((model_title, F1, roc_number))
            if "Feature" not in model_title:
                data_voting.append((model_title, F1, roc_number))
                data_regressor.append((model_title, F1, roc_number))
        elif "LSTM Variational Autoencoder" in model_title:
            data_lstm_vae.append((model_title, F1, roc_number))
            if "Feature" not in model_title:
                data_voting.append((model_title, F1, roc_number))
                data_regressor.append((model_title, F1, roc_number))
        elif "Convolutional" in model_title:
            data_conv_ae.append((model_title, F1, roc_number))
            if "Feature" not in model_title:
                data_voting.append((model_title, F1, roc_number))
                data_regressor.append((model_title, F1, roc_number))
        elif "Voting" in model_title:
            data_voting.append((model_title, F1, roc_number))
        elif "Logistic Regression" in model_title:
            data_regressor.append((model_title, F1, roc_number))

    df_ae = pd.DataFrame(data_ae, columns=["Model Title", "F1 Score", "AUC"])
    df_conv_ae = pd.DataFrame(data_conv_ae, columns=["Model Title", "F1 Score", "AUC"])
    df_lstm = pd.DataFrame(data_lstm, columns=["Model Title", "F1 Score", "AUC"])
    df_lstm_ae = pd.DataFrame(data_lstm_ae, columns=["Model Title", "F1 Score", "AUC"])
    df_lstm_vae = pd.DataFrame(data_lstm_vae, columns=["Model Title", "F1 Score", "AUC"])
    df_voting = pd.DataFrame(data_voting, columns=["Model Title", "F1 Score", "AUC"])
    df_regressor = pd.DataFrame(data_regressor, columns=["Model Title", "F1 Score", "AUC"])

    ord_by_col = None
    if order_by_column=="f1":
        ord_by_col = "F1 Score"
    elif order_by_column=="auc":
        ord_by_col = "AUC"

    df_ae_final_ordered = df_ae.sort_values(by=[ord_by_col], ascending=False)
    df_conv_ae_final_ordered = df_conv_ae.sort_values(by=[ord_by_col], ascending=False)
    df_lstm_final_ordered = df_lstm.sort_values(by=[ord_by_col], ascending=False)
    df_lstm_ae_final_ordered = df_lstm_ae.sort_values(by=[ord_by_col], ascending=False)
    df_lstm_vae_final_ordered = df_lstm_vae.sort_values(by=[ord_by_col], ascending=False)
    df_voting_final_ordered = df_voting.sort_values(by=[ord_by_col], ascending=False)
    df_regressor_final_ordered = df_regressor.sort_values(by=[ord_by_col], ascending=False)

    display(df_ae_final_ordered)
    display(df_conv_ae_final_ordered)
    display(df_lstm_final_ordered)
    display(df_lstm_ae_final_ordered)
    display(df_lstm_vae_final_ordered)
    display(df_voting_final_ordered)
    display(df_regressor_final_ordered)

    print_latex_code(df_ae_final_ordered)
    print_latex_code(df_conv_ae_final_ordered)
    print_latex_code(df_lstm_final_ordered)
    print_latex_code(df_lstm_ae_final_ordered)
    print_latex_code(df_lstm_vae_final_ordered)
    print_latex_code(df_voting_final_ordered)
    print_latex_code(df_regressor_final_ordered)


def print_metrics_forthesis(df, y_actual, y_predicted):
    
    model_title = y_predicted.replace("anomaly_by_","")
    if model_title == "autoencoder":
        model_title = "Autoencoder"
    elif model_title == "conv_ae":
        model_title = "Convolutional Autoencoder"
    elif model_title == "lstm":
        model_title = "LSTM"
    elif model_title == "lstm_ae":
        model_title = "LSTM Autoencoder"
    elif model_title == "lstm_vae":
        model_title = "LSTM Variational Autoencoder"
    elif model_title == "logistic_regression":
        model_title = "Logistic Regression"
    elif model_title == "ensemble_ensemble_autoencoder":
        model_title = "Feature Bagging Autoencoder"
    elif model_title == "ensemble_ensemble_conv_ae":
        model_title = "Feature Bagging Convolutional Autoencoder"
    elif model_title == "ensemble_ensemble_lstm":
        model_title = "Feature Bagging LSTM"
    elif model_title == "ensemble_ensemble_lstm_ae":
        model_title = "Feature Bagging LSTM Autoencoder"
    elif model_title == "ensemble_ensemble_lstm_vae":
        model_title = "Feature Bagging LSTM Variational Autoencoder" 
    elif model_title == "ensemble_ensemble_rotation_autoencoder":
        model_title = "Feature Bagging with Rotation Autoencoder" 
    elif model_title == "ensemble_ensemble_rotation_conv_ae":
        model_title = "Feature Bagging with Rotation Convolutional Autoencoder" 
    elif model_title == "ensemble_ensemble_rotation_lstm":
        model_title = "Feature Bagging with Rotation LSTM" 
    elif model_title == "ensemble_ensemble_rotation_lstm_ae":
        model_title = "Feature Bagging with Rotation LSTM Autoencoder" 
    elif model_title == "ensemble_ensemble_rotation_lstm_vae":
        model_title = "Feature Bagging with Rotation LSTM Variational Autoencoder" 
    elif model_title == "voting":
        model_title = "Majority Voting" 
    F1 = f1_score(df[y_actual], df[y_predicted])
    roc_number = roc_auc_score(df[y_actual], df[y_predicted])
    
    print(f"{model_title} {F1} {roc_number}")

    return model_title, F1, roc_number

def plot_single_confusion_matrix(df, y_actual, y_predicted, save_image=None):
    px = 1/plt.rcParams['figure.dpi']

    fig = plt.figure()
    fig.patch.set_facecolor((1,1,1,1))
    fig.patch.set_alpha(1.0)
    
    roc_number = roc_auc_score(df[y_actual], df[y_predicted])

    model_title = y_predicted.replace("anomaly_by_","")
    F1 = f1_score(df[y_actual], df[y_predicted])
    custom_title = f"{model_title}\nAUC: {roc_number}\nF1 Score: {F1}"
    confusion_matrix = pd.crosstab(df[y_actual], df[y_predicted], rownames=['Actual'], colnames=['Predicted'])
    ax = sns.heatmap(confusion_matrix, annot=True, fmt='g').set_title(custom_title)
    
    if save_image is not None:
        image_path = str(save_image)
        # remove white space
        plt.savefig(image_path, bbox_inches='tight')
    plt.show()

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

def plot_confusion_matrix_all(df, y_actual, *cols, save_image=None):
    px = 1/plt.rcParams['figure.dpi']

    fig = plt.figure(figsize=(1920*px, 1080*px))
    # fig = plt.figure(figsize = (24,8))
    fig.patch.set_facecolor((1,1,1,1))
    fig.patch.set_alpha(1.0)

    x, y = get_x_y(len(cols))


    fig.subplots_adjust(hspace=1.2, wspace=0.5)

    for i,col in enumerate(cols):
        ax = fig.add_subplot(x, y, i+1)

        roc_number = roc_auc_score(df[y_actual], df[col])

        model_title = col.replace("anomaly_by_","")
        F1 = f1_score(df[y_actual], df[col])
        custom_title = f"{model_title}\nAUC: {roc_number}\nF1 Score: {F1}"
        confusion_matrix = pd.crosstab(df[y_actual], df[col], rownames=['Actual'], colnames=['Predicted'])
        g = sns.heatmap(confusion_matrix, annot=True, fmt='g', ax = ax).set_title(custom_title)

    fig.tight_layout()
    if save_image is not None:
        image_path = str(save_image)
        # remove white space
        plt.savefig(image_path, bbox_inches='tight')
    
    plt.show()

def get_metric_scores(df, y_actual, *cols, save_image=None):

    data = []
    for i,col in enumerate(cols):

        roc_number = roc_auc_score(df[y_actual], df[col])

        model_title = col.replace("anomaly_by_","")
        model_name = model_title.replace("ensemble_ensemble_","")
        F1 = f1_score(df[y_actual], df[col])
    
        data.append((model_name,F1,roc_number))
    
    df = pd.DataFrame(data, columns=["model_name", "f1_score", "auc"])

    return df


def get_bagg_features(T=-1):
    seq = ['Accelerometer1RMS', 'Accelerometer2RMS', 'Current', 'Pressure', 'Temperature', 'Thermocouple', 'Voltage', 'Volume Flow RateRMS']
    comb_size_4 = list(itertools.combinations(seq,4))
    comb_size_5 = list(itertools.combinations(seq,5))
    comb_size_6 = list(itertools.combinations(seq,6))
    comb_size_7 = list(itertools.combinations(seq,7))
    size_N = 0
    if ((T<=0) or (T>162)):
        return [seq]
    elif ((T>0) or (T<=162)):
        all_lists = []
        for i in range(0,T):
            choose_comb = []
            if len(comb_size_4)!=0:
                choose_comb.append(4)
            if len(comb_size_5)!=0:
                choose_comb.append(5)
            if len(comb_size_6)!=0:
                choose_comb.append(6)
            if len(comb_size_7)!=0:
                choose_comb.append(7)
            
            size_N = random.choice(choose_comb)
            
            tmp_list_ind = None
            if size_N==4:
                tmp_list_ind = random.randint(0, len(comb_size_4)-1)
                all_lists.append(comb_size_4.pop(tmp_list_ind))
            if size_N==5:
                tmp_list_ind = random.randint(0, len(comb_size_5)-1)
                all_lists.append(comb_size_5.pop(tmp_list_ind))
            if size_N==6:
                tmp_list_ind = random.randint(0, len(comb_size_6)-1)
                all_lists.append(comb_size_6.pop(tmp_list_ind))
            if size_N==7:
                tmp_list_ind = random.randint(0, len(comb_size_7)-1)
                all_lists.append(comb_size_7.pop(tmp_list_ind))
        return all_lists
    else:
        return None

def get_bagg_features_random(T, random_value):
    if random_value:
        seq = ['Accelerometer1RMS', 'Accelerometer2RMS', 'Current', 'Pressure', 'Temperature', 'Thermocouple', 'Voltage', 'Volume Flow RateRMS']
        comb_size_4 = list(itertools.combinations(seq,4))
        comb_size_5 = list(itertools.combinations(seq,5))
        comb_size_6 = list(itertools.combinations(seq,6))
        comb_size_7 = list(itertools.combinations(seq,7))
        comb_all = []
        comb_all.append(seq)
        for cb in [comb_size_4, comb_size_5, comb_size_6, comb_size_7]:
            for i in cb:
                comb_all.append(i)
        size_N = 0
        if T<=0:
            return [seq]
        elif T>0:
            all_lists = []
            for i in range(0,T):
                choose_comb = [4,5,6,7]
                size_N = random.choice(choose_comb)
                
                tmp_list_ind = None
                if size_N==4:
                    tmp_list_ind = random.randint(0, len(comb_size_4)-1)
                    all_lists.append(comb_size_4[tmp_list_ind])
                if size_N==5:
                    tmp_list_ind = random.randint(0, len(comb_size_5)-1)
                    all_lists.append(comb_size_5[tmp_list_ind])
                if size_N==6:
                    tmp_list_ind = random.randint(0, len(comb_size_6)-1)
                    all_lists.append(comb_size_6[tmp_list_ind])
                if size_N==7:
                    tmp_list_ind = random.randint(0, len(comb_size_7)-1)
                    all_lists.append(comb_size_7[tmp_list_ind])
            return all_lists
        else:
            return None
    else:
        seq = ['Accelerometer1RMS', 'Accelerometer2RMS', 'Current', 'Pressure', 'Temperature', 'Thermocouple', 'Voltage', 'Volume Flow RateRMS']
        comb_size_4 = list(itertools.combinations(seq,4))
        comb_size_5 = list(itertools.combinations(seq,5))
        comb_size_6 = list(itertools.combinations(seq,6))
        comb_size_7 = list(itertools.combinations(seq,7))
        size_N = 0
        if ((T<=0) or (T>162)):
            return [seq]
        elif ((T>0) or (T<=162)):
            all_lists = []
            for i in range(0,T):
                choose_comb = []
                if len(comb_size_4)!=0:
                    choose_comb.append(4)
                if len(comb_size_5)!=0:
                    choose_comb.append(5)
                if len(comb_size_6)!=0:
                    choose_comb.append(6)
                if len(comb_size_7)!=0:
                    choose_comb.append(7)
                
                size_N = random.choice(choose_comb)
                
                tmp_list_ind = None
                if size_N==4:
                    tmp_list_ind = random.randint(0, len(comb_size_4)-1)
                    all_lists.append(comb_size_4.pop(tmp_list_ind))
                if size_N==5:
                    tmp_list_ind = random.randint(0, len(comb_size_5)-1)
                    all_lists.append(comb_size_5.pop(tmp_list_ind))
                if size_N==6:
                    tmp_list_ind = random.randint(0, len(comb_size_6)-1)
                    all_lists.append(comb_size_6.pop(tmp_list_ind))
                if size_N==7:
                    tmp_list_ind = random.randint(0, len(comb_size_7)-1)
                    all_lists.append(comb_size_7.pop(tmp_list_ind))
            return all_lists
        else:
            return None

def check_to_stop(df, col_to_test, f1_limit, roc_limit):
    df["anomaly_test"] = np.where(pd.notnull(df[col_to_test]), df[col_to_test], df["anomaly"])

    if f1_limit is not None:
        F1 = f1_score(df["anomaly"], df["anomaly_test"])
        if F1<f1_limit:
            roc_number = roc_auc_score(df["anomaly"], df["anomaly_test"])
            reason="F1"
            return True, F1, roc_number, reason

    if roc_limit is not None:
        roc_number = roc_auc_score(df["anomaly"], df["anomaly_test"])
        if roc_number<roc_limit:
            F1 = f1_score(df["anomaly"], df["anomaly_test"])
            reason = "ROC"
            return True, F1, roc_number, reason
    F1 = f1_score(df["anomaly"], df["anomaly_test"])
    roc_number = roc_auc_score(df["anomaly"], df["anomaly_test"])
    return False, F1, roc_number, None


class FileWriter():
    def __init__(self):
        self.SAVE_FOLDER = "./filewriter_tmp"
        self.mode = "a"
    def set_filewriter(self, SAVE_FOLDER="./filewriter_tmp", mode="a"):
        self.SAVE_FOLDER = SAVE_FOLDER
        self.mode = mode

    def write(self, data):
        with open(self.SAVE_FOLDER, self.mode) as f:
            f.write(data)
            f.write('\n')

    def write_batch(self, data):
        with open(self.SAVE_FOLDER, self.mode) as f:
            f.writelines(data)
            f.write('\n')

class t_estimator:
    def __init__(self) -> None:
        self.total_number_of_iterations = np.Infinity
        self.time_of_iteration = []
        self.readable_time = False

    def set_total_number_of_iterations(self, n):
        self.total_number_of_iterations = n
    def add_iteration_time(self, secs):
        self.time_of_iteration.append(secs)
    

    def __str__(self) -> str:
        if len(self.time_of_iteration)<=0 or self.total_number_of_iterations==np.Infinity:
            return f"Finished: {len(self.time_of_iteration)}/{self.total_number_of_iterations} - Time Estimation: {np.Infinity}"
        else:
            if self.readable_time:
                return f"Finished: {len(self.time_of_iteration)}/{self.total_number_of_iterations} - Total: {np.sum(self.time_of_iteration):.2f} secs - ET: {np.sum(self.time_of_iteration)/len(self.time_of_iteration):.2f} sec/it - Remaining Time: {(np.sum(self.time_of_iteration)/len(self.time_of_iteration))*(self.total_number_of_iterations-len(self.time_of_iteration)):.2f} secs"
            else:
                total_time_hour = int((np.sum(self.time_of_iteration))//3600)
                total_time_min = int(((np.sum(self.time_of_iteration))%3600)//60)
                total_time_sec = int(((np.sum(self.time_of_iteration))%3600)%60)

                time_hour = int((np.sum(self.time_of_iteration)//len(self.time_of_iteration))//3600)
                time_min = int(((np.sum(self.time_of_iteration)//len(self.time_of_iteration))%3600)//60)
                time_sec = int(((np.sum(self.time_of_iteration)//len(self.time_of_iteration))%3600)%60)

                rem_time_hour = int(((np.sum(self.time_of_iteration)/len(self.time_of_iteration))*(self.total_number_of_iterations-len(self.time_of_iteration)))//3600)
                rem_time_min = int((((np.sum(self.time_of_iteration)/len(self.time_of_iteration))*(self.total_number_of_iterations-len(self.time_of_iteration)))%3600)//60)
                rem_time_sec = int((((np.sum(self.time_of_iteration)/len(self.time_of_iteration))*(self.total_number_of_iterations-len(self.time_of_iteration)))%3600)%60)

                return f"Finished: {len(self.time_of_iteration)}/{self.total_number_of_iterations} - Total: {np.sum(self.time_of_iteration):.2f} secs - {total_time_hour} H : {total_time_min} m : {total_time_sec:.2f} s ET: {np.sum(self.time_of_iteration)/len(self.time_of_iteration):.2f} sec/it - {time_hour} H : {time_min} m : {time_sec:.2f} s /it - Remaining Time: {(np.sum(self.time_of_iteration)/len(self.time_of_iteration))*(self.total_number_of_iterations-len(self.time_of_iteration)):.2f} secs - {rem_time_hour} H : {rem_time_min} m : {rem_time_sec:.2f} s"

