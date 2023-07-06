import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score
import os
# pd.options.mode.chained_assignment = None

def get_anomalies_from_list(models_list):
    tmp_list = []
    for model, name in models_list:
        pairs = []
        for cl in model.columns.to_list():
            if "score" in cl:
                tmp = cl[:-6]
                pairs.append((tmp, tmp+"_score", tmp+"_ucl"))
        for pair in pairs:
            model[pair[0]] = (model[pair[1]] > model[pair[2]]).astype("int")
        tmp_list.append((model[["anomaly"] + [pair[0] for pair in pairs]], name))
    return tmp_list

def get_ensemble_models_from_list(models_list):
    tmp_list = []
    for model, name in models_list:
        for cl in model.columns.to_list():
            if "anomaly_by_" in cl:
                model[cl] = model.apply(lambda x: -1 if x[cl]==0 else 1, axis=1)

        model["predicted"] = model.drop("anomaly", axis=1).apply(np.sum, axis=1)
        model["predicted"] = model.apply(lambda x: 1 if x["predicted"]>0 else 0, axis=1)
        tmp_list.append((model[["anomaly", "predicted"]], name))
    return tmp_list

def get_anomalies(filename):
    df = pd.read_csv(f"{filename}")
    cols = df.columns.to_list()
    basic = [cl for cl in cols if "basic" in cl]
    rotation = [cl for cl in cols if "rotation" in cl]
    fbagging = [cl for cl in cols if "fbagging" in cl]

    ae_basic = df[["anomaly"]+[cl for cl in basic if "anomaly_by_autoencoder_basic" in cl]]
    conv_ae_basic = df[["anomaly"]+[cl for cl in basic if "anomaly_by_conv_ae_basic" in cl]]
    lstm_basic = df[["anomaly"]+[cl for cl in basic if "anomaly_by_lstm_basic" in cl]]
    lstm_ae_basic = df[["anomaly"]+[cl for cl in basic if "anomaly_by_lstm_ae_basic" in cl]]
    lstm_vae_basic = df[["anomaly"]+[cl for cl in basic if "anomaly_by_lstm_vae_basic" in cl]]
    basic_models_list = [(ae_basic, "autoencoder_basic"), (conv_ae_basic, "conv_ae_basic"), (lstm_basic, "LSTM_basic"), 
                        (lstm_ae_basic, "LSTM_autoencoder_basic"), (lstm_vae_basic, "LSTM_VAE_basic")]

    ae_fbagging = df[["anomaly"]+[cl for cl in fbagging if "anomaly_by_autoencoder_fbagging" in cl]]
    conv_ae_fbagging = df[["anomaly"]+[cl for cl in fbagging if "anomaly_by_conv_ae_fbagging" in cl]]
    lstm_fbagging = df[["anomaly"]+[cl for cl in fbagging if "anomaly_by_lstm_fbagging" in cl]]
    lstm_ae_fbagging = df[["anomaly"]+[cl for cl in fbagging if "anomaly_by_lstm_ae_fbagging" in cl]]
    lstm_vae_fbagging = df[["anomaly"]+[cl for cl in fbagging if "anomaly_by_lstm_vae_fbagging" in cl]]
    fbagging_models_list = [(ae_fbagging, "autoencoder_Feature_Bagging"), (conv_ae_fbagging, "conv_ae_Feature_Bagging"), 
                            (lstm_fbagging, "LSTM_Feature_Bagging"), (lstm_ae_fbagging, "LSTM_autoencoder_Feature_Bagging"), 
                            (lstm_vae_fbagging, "LSTM_VAE_Feature_Bagging")]

    ae_rotation = df[["anomaly"]+[cl for cl in rotation if "anomaly_by_autoencoder_rotation" in cl]]
    conv_ae_rotation = df[["anomaly"]+[cl for cl in rotation if "anomaly_by_conv_ae_rotation" in cl]]
    lstm_rotation = df[["anomaly"]+[cl for cl in rotation if "anomaly_by_lstm_rotation" in cl]]
    lstm_ae_rotation = df[["anomaly"]+[cl for cl in rotation if "anomaly_by_lstm_ae_rotation" in cl]]
    lstm_vae_rotation = df[["anomaly"]+[cl for cl in rotation if "anomaly_by_lstm_vae_rotation" in cl]]
    rotation_models_list = [(ae_rotation, "autoencoder_FBR"), (conv_ae_rotation, "conv_ae_FBR"), (lstm_rotation, "LSTM_FBR"), 
                            (lstm_ae_rotation, "LSTM_autoencoder_FBR"), (lstm_vae_rotation, "LSTM_VAE_FBR")]

    basic_models_list = get_anomalies_from_list(basic_models_list)
    fbagging_models_list = get_anomalies_from_list(fbagging_models_list)
    rotation_models_list = get_anomalies_from_list(rotation_models_list)

    basic_models_list = get_ensemble_models_from_list(basic_models_list)
    fbagging_models_list = get_ensemble_models_from_list(fbagging_models_list)
    rotation_models_list = get_ensemble_models_from_list(rotation_models_list)

    return basic_models_list, fbagging_models_list, rotation_models_list

def plot_confusion_matrix_all(basic_models_list, fbagging_models_list, rotation_models_list, save_folder=None, save_image_name=None):
    px = 1/plt.rcParams['figure.dpi']

    fig = plt.figure(figsize=(1920*px, 1080*px))
    # fig = plt.figure(figsize = (24,8))
    fig.patch.set_facecolor((1,1,1,1))
    fig.patch.set_alpha(1.0)

    df_list = basic_models_list + fbagging_models_list + rotation_models_list


    fig.subplots_adjust(hspace=1.2, wspace=0.5)

    for i, tmp_df in enumerate(df_list):
        df = tmp_df[0]
        df_name = tmp_df[1]
        ax = fig.add_subplot(3, 5, i+1)

        roc_number = roc_auc_score(df["anomaly"], df["predicted"])
        F1 = f1_score(df["anomaly"], df["predicted"])

        model_title = df_name.replace("_"," ")
        
        custom_title = f"{model_title}\nF1: {F1}\nAUC: {roc_number}"
        confusion_matrix = pd.crosstab(df["anomaly"], df["predicted"], rownames=['Actual'], colnames=['Predicted'])
        g = sns.heatmap(confusion_matrix, annot=True, fmt='g', ax = ax).set_title(custom_title)

    fig.tight_layout()
    if (save_image_name is not None) and (save_folder is not None):
        image_path = str(save_image_name)
        # remove white space
        plt.savefig(f"{save_folder}/{image_path}.png", bbox_inches='tight')
    
    plt.show()

def plot_confusion_matrix_fast_folders():
    all_files=[]
    for root, dirs, files in os.walk("./"):
        for file in files:
            if file.endswith(".csv"):
                if "df_values_scores" in file:
                    all_files.append((root, os.path.join(root, file)))

    for folder, filename in all_files:
        basic_models_list, fbagging_models_list, rotation_models_list = get_anomalies(filename)
        plot_confusion_matrix_all(basic_models_list, fbagging_models_list, rotation_models_list, 
                                save_folder=folder, save_image_name="df_values_plot")