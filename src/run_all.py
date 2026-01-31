import logging
from pathlib import Path
import sys

sys.path.append(str(Path.cwd() / "src"))

from config import RAW_DEMAND_DIR, DEMAND_ALL_YEARS
from combine_demand_files import combine_demand_files
from prepare_temperature import main as prepare_temperature_main
from run_pipeline import main as run_pipeline_main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def main():
    logging.info("=== MASTER PIPELINE START ===")

    logging.info("Combining demand files...")
    combine_demand_files(RAW_DEMAND_DIR, DEMAND_ALL_YEARS)
    logging.info("Demand files combined.")

    logging.info("Preparing temperature CSV...")
    prepare_temperature_main()
    logging.info("Temperature CSV prepared.")

    logging.info("Running main pipeline...")
    run_pipeline_main()
    logging.info("Main pipeline completed.")

    logging.info("=== MASTER PIPELINE FINISHED ===")

if __name__ == "__main__":
    main()
