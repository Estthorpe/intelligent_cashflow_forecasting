# config/config.py
from pathlib import Path
import os

# ----------------------------
# Project Root
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]

# ----------------------------
# Data Paths
# ----------------------------
CLEAN_DATA_PATH = str(ROOT / "data" / "processed" / "clean_ap_ar.csv")
DATA_PROCESSED = Path(CLEAN_DATA_PATH)  # keep both string and Path versions

# ----------------------------
# Forecasting Parameters
# ----------------------------
FORECAST_HORIZON_DAYS = 60
TRAIN_END = "2019-12-31"
VAL_END = "2020-03-31"

# ----------------------------
# Output Directories
# ----------------------------
FORECAST_OUTPUT_DIR = str(ROOT / "outputs" / "forecasts")
OUTPUT_FORECASTS_DIR = Path(FORECAST_OUTPUT_DIR)  # keep Path version
OUTPUT_ALERTS_DIR = ROOT / "outputs" / "alerts"

# Create directories if they don't exist
os.makedirs(FORECAST_OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_ALERTS_DIR, exist_ok=True)

# ----------------------------
# Anomaly Detection Settings
# ----------------------------
ZSCORE_THRESHOLD = 3.0
ISOF_CONTAM = 0.01
RANDOM_STATE = 42
