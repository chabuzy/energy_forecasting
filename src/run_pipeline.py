# src/run_pipeline.py
import sys
from pathlib import Path
import logging

from validation import validate_processed_files

sys.path.append(str(Path.cwd() / "src"))

from config import (
    RAW_TEMPERATURE_FILE,
    TEMPERATURE_HALF_HOURLY,
    DEMAND_TEMP_MERGED,
)
from temperature_processing import process_temperature
from merge_temperature import merge_temperature_with_demand

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def main():
    logging.info("Starting pipeline run.")

    # Step 1: Process temperature
    logging.info(f"Processing temperature from {RAW_TEMPERATURE_FILE} to {TEMPERATURE_HALF_HOURLY}")
    process_temperature(RAW_TEMPERATURE_FILE, TEMPERATURE_HALF_HOURLY)
    logging.info("Temperature processing completed.")

    # Step 2: Merge with demand
    logging.info(f"Merging demand with temperature into {DEMAND_TEMP_MERGED}")
    merge_temperature_with_demand()
    logging.info("Merging completed.")

    # Step 3: Validate outputs
    logging.info("Validating outputs...")
    validate_processed_files(TEMPERATURE_HALF_HOURLY, DEMAND_TEMP_MERGED)

    logging.info("Pipeline run finished successfully.")

if __name__ == "__main__":
    main()
