# src/merge_temperature.py
import pandas as pd
from config import (
    DEMAND_ALL_YEARS,
    TEMPERATURE_HALF_HOURLY,
    DEMAND_TEMP_MERGED
)

def merge_temperature_with_demand():
    # Load demand
    demand = pd.read_csv(DEMAND_ALL_YEARS, parse_dates=["timestamp"])
    demand = demand.rename(columns={"ENGLAND_WALES_DEMAND": "demand"})

    # Load temperature
    temp = pd.read_csv(TEMPERATURE_HALF_HOURLY, parse_dates=["datetime"])
    temp = temp.rename(columns={"datetime": "timestamp"})

    # Merge
    merged = pd.merge(demand, temp, on="timestamp", how="left")

    merged.to_csv(DEMAND_TEMP_MERGED, index=False)
    print(f"Merged dataset saved to {DEMAND_TEMP_MERGED}")
