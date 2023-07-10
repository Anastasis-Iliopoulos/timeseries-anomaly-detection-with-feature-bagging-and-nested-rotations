from pathlib import Path
import pandas as pd

class InfoWriter():
    def __init__(self, save_folder, family_model_name, model_name, data_name):
        if (save_folder is None) or (family_model_name is None) or (model_name is None):
            raise ValueError("arguments should not be None!")
        
        self.SAVE_FOLDER = f"./{save_folder}/{family_model_name}/{model_name}/{data_name}"
        Path(self.SAVE_FOLDER).mkdir(parents=True, exist_ok=True)

        self.family_model_name = family_model_name
        self.model_name = model_name
        self.data_name = data_name

        # DataFrame
        self.scores_and_anomalies = None
        # DataFrame
        self.nested_rotations = None
        # DataFrame
        self.scalings = None
        
        # For each partition/rotationMatrix Create DataFrame
        self.nrpartitions = None
        self.nrmatrices = None
        
        # Crate DataFrame with these values as rows
        self.scvar = None
        self.scmean = None
        self.scscale = None
        self.fbsubset = None
        self.nrK = None
        self.nrfranction = None
        self.UCL = None
        self.timers = None
        self.total_time = None
        self.steps_applied = None


    def write_info(self):
        self.scores_and_anomalies.to_parquet(f"./{self.SAVE_FOLDER}/scores_and_anomalies.parquet.gzip", compression='gzip')
        self.nested_rotations.to_parquet(f"./{self.SAVE_FOLDER}/nested_rotations.parquet.gzip", compression='gzip')
        self.scalings.to_parquet(f"./{self.SAVE_FOLDER}/scalings.parquet.gzip", compression='gzip')

        for ind, (partition_numbers, rotation_matrix) in enumerate(zip(self.nrpartitions, self.nrmatrices)):
            partition = [str(i) for i in partition_numbers]
            df = pd.DataFrame(rotation_matrix, columns=partition)
            df.to_parquet(f"./{self.SAVE_FOLDER}/partition_{ind}.parquet.gzip", compression='gzip')

        df = pd.DataFrame([
            ("scaler_var", self.scvar.tolist()),
            ("scaler_mean", self.scmean.tolist()),
            ("scaler_scale", self.scscale.tolist()),
            ("feature_bagging_subset", self.fbsubset),
            ("nested_rotations_K", [self.nrK]),
            ("nested_rotations_franction", [self.nrfranction]),
            ("timers", self.timers),
            ("total_time", self.total_time)
        ], columns=["info_key", "info_value"])

        df.to_parquet(f"./{self.SAVE_FOLDER}/other_info.parquet.gzip", compression='gzip')
