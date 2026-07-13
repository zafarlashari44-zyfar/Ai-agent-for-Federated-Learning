from pathlib import Path

# ==========================
# PROJECT PATHS
# ==========================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
RESULTS_DIR = BASE_DIR / "results"
ASSETS_DIR = BASE_DIR / "assets"

# ==========================
# DATASET
# ==========================

MITBIH_TRAIN = DATA_DIR / "mitbih_train.csv"
MITBIH_TEST = DATA_DIR / "mitbih_test.csv"

PTB_NORMAL = DATA_DIR / "ptbdb_normal.csv"
PTB_ABNORMAL = DATA_DIR / "ptbdb_abnormal.csv"

# ==========================
# OUTPUT FILES
# ==========================

MODEL_FILE = MODELS_DIR / "federated_model.pth"

JSON_REPORT = REPORTS_DIR / "ehr_report.json"

PDF_REPORT = REPORTS_DIR / "ehr_report.pdf"

LOG_FILE = RESULTS_DIR / "agent.log"

# ==========================
# MODEL PARAMETERS
# ==========================

NUM_CLIENTS = 5

NUM_CLASSES = 5

INPUT_FEATURES = 187

DEVICE = "cpu"