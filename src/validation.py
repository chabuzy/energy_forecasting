# src/validation.py
import pandas as pd
import logging

def validate_processed_files(temp_path, merged_path):
    # Validate temperature file
    temp_df = pd.read_csv(temp_path, parse_dates=["datetime"])
    logging.info(f"Temperature file OK: {len(temp_df)} rows.")

    if temp_df.isna().any().any():
        logging.warning("Temperature file contains missing values.")

    # Validate merged file
    merged_df = pd.read_csv(merged_path, parse_dates=["timestamp"])
    logging.info(f"Merged file OK: {len(merged_df)} rows.")

    if merged_df.isna().any().any():
        logging.warning("Merged file contains missing values.")

    if merged_df["timestamp"].duplicated().any():
        logging.warning("Merged file contains duplicate timestamps.")
