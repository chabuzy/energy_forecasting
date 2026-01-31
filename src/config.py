from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Raw files
RAW_DEMAND_DIR = RAW_DIR
RAW_TEMPERATURE_FILE = RAW_DIR / "temperature_monthly_clean.csv"

# Processed files
DEMAND_ALL_YEARS = PROCESSED_DIR / "demand_all_years.csv"
TEMPERATURE_HALF_HOURLY = PROCESSED_DIR / "temperature_half_hourly.csv"
DEMAND_TEMP_MERGED = PROCESSED_DIR / "demand_temperature_half_hourly.csv"
