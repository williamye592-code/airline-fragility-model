from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MODELS_DIR = OUTPUTS_DIR / "models"
TABLES_DIR = OUTPUTS_DIR / "tables"
PLOTS_DIR = OUTPUTS_DIR / "plots"

for path in [OUTPUTS_DIR, MODELS_DIR, TABLES_DIR, PLOTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

RAW_CSV_PATH = RAW_DATA_DIR / "aa_sequences.csv"


DATE_COL = "SEQ_SCHD_START_DT"
START_TIME_COL = "SEQ_START"
TARGET_SOURCE_COL = "SPOILAGE"
TARGET_COL = "target_spoilage"

CATEGORICAL_COLS = [
    "FLEET",
    "BASE",
    "DIVISION",
    "FLIGHT_PATTERN",
]

NUMERICAL_COLS = [
    "TOTAL_BLOCKED_HRS",
    "SEQ_CAL_DAYS",
    "SEQ_DUTY_DAYS",
    "SEQ_TTL_FLTTIME",
    "MIN_FLTTIME_PER_LEG",
    "MAX_LEGS_PER_DAY",
    "SEQ_TTL_LEGS",
    "LAYOVER",
]

ENGINEERED_NUM_COLS = [
    "seq_month",
    "seq_dayofweek",
    "seq_start_hour",
    "is_weekend",
]

FEATURE_COLS = CATEGORICAL_COLS + NUMERICAL_COLS + ENGINEERED_NUM_COLS


TEST_SIZE_FRACTION = 0.2
RANDOM_STATE = 42
POSITIVE_LABELS = {"PARTIALLY SPOILED", "SPOILED"} 