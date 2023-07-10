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
        self.scores_ucls_anomalies = None
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
        self.timers = None
        self.total_time = None
        


    def write_info(self):
        self.scores_ucls_anomalies.to_parquet(f"./{self.SAVE_FOLDER}/scores_ucls_anomalies.parquet.gzip", compression='gzip')
        self.nested_rotations.to_parquet(f"./{self.SAVE_FOLDER}/nested_rotations.parquet.gzip", compression='gzip')
        self.scalings.to_parquet(f"./{self.SAVE_FOLDER}/scalings.parquet.gzip", compression='gzip')

        for ind, tmp_t in enumerate(zip(self.nrpartitions, self.nrmatrices)):
            partition = tmp_t[0] 
            rotation_matrix = tmp_t[1]
            df = pd.DataFrame(rotation_matrix, columns=partition)
            df.to_parquet(f"./{self.SAVE_FOLDER}/partition_{ind}.parquet.gzip", compression='gzip')

        df = pd.DataFrame([
            ("scaler_var", self.scvar)
            ("scaler_mean", self.scmean)
            ("scaler_scale", self.scscale)
            ("feature_bagging_subset", self.fbsubset)
            ("nested_rotations_K", self.nrK)
            ("nested_rotations_franction", self.nrfranction)
            ("timers", self.timers)
            ("total_time", self.total_time)
        ], columns=["info_key", "info_value"])

        df.to_parquet(f"./{self.SAVE_FOLDER}/other_info.parquet.gzip", compression='gzip')



# create Folders if not exist
Path(SAVE_FOLDER).mkdir(parents=True, exist_ok=True)

start = time.perf_counter()
end = time.perf_counter()

start_of_the_whole_process = time.perf_counter()

Path(SAVE_FOLDER).mkdir(parents=True, exist_ok=True)
X_conv_ae_window = 60
final_aggr_pd = None
final_aggr_pd_general = None
start_of_loop = time.perf_counter()
times_of_ds = []
ESTIMATORS = n_est
LOGS_SAVE_FOLDER = SAVE_FOLDER
LOGS_FILENAME = "logs_info.log"

log = logging.getLogger()
for hdlr in log.handlers[:]:
    log.removeHandler(hdlr)

import logging
logging.basicConfig(level=logging.INFO, filename=f"{LOGS_SAVE_FOLDER}/{LOGS_FILENAME}",
                    format="CREATED_AT: %(asctime)s - MESSAGE: %(message)s")


logging.info(f"'Params: num_of_cores={num_of_cores}, SAVE_FOLDER={SAVE_FOLDER}, n_est={n_est}, K={K}, fraction={fraction}'")