
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

logging.basicConfig(level=logging.INFO, filename=f"{LOGS_SAVE_FOLDER}/{LOGS_FILENAME}",
                    format="CREATED_AT: %(asctime)s - MESSAGE: %(message)s")


logging.info(f"'Params: num_of_cores={num_of_cores}, SAVE_FOLDER={SAVE_FOLDER}, n_est={n_est}, K={K}, fraction={fraction}'")