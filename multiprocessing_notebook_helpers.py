from tqdm import tqdm
import pipeline_models
import numpy as np
import os
import random
import tensorflow as tf

def set_random(seed_value):
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)

def run_pipeline_conv_ae(task_label, list_of_df, seed):
    set_random(seed)
    for  tmp_file_ind, df in tqdm(enumerate(list_of_df), str(task_label)):
        
        tmp_model_obj = pipeline_models.PipelineModel(capture_info=True, infoWriter_args=["tmp_res", "family_conv_ae", f"model_name_{str(task_label)}", f"data_name_{str(tmp_file_ind).rjust(3,'0')}"])
        
        tmp_model_obj.apply_standard_scaling(())
        tmp_model_obj.apply_nested_rotations((2, 0.75))
        tmp_model_obj.apply_feature_bagging(())
        tmp_model_obj.apply_model(("CONV_AE", f"model_name_{str(task_label)}"))
        tmp_model_obj.pipeline_fit(df[:400].drop(['anomaly', 'changepoint'], axis=1))
        final_df = tmp_model_obj.pipeline_transform(df.drop(['anomaly', 'changepoint'], axis=1))
    return 1