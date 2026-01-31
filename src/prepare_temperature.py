# src/prepare_temperature.py
import pandas as pd
from pathlib import Path

def main():
    # Load files
    min_df = pd.read_csv("data/raw/Monthly_Min_Temperature_1991_2020.csv")
    max_df = pd.read_csv("data/raw/Monthly_Max_Temperature_1991_2020.csv")

    print(min_df.columns)
    print(max_df.columns)

    # Use first grid cell
    min_row = min_df.iloc[0]
    max_row = max_df.iloc[0]

    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    tmin = [min_row[f"tmin {m}"] for m in months]
    tmax = [max_row[f"tmax {m}"] for m in months]

    tmean = [(lo + hi) / 2 for lo, hi in zip(tmin, tmax)]

    # Create monthly dates (climatology)
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")

    df = pd.DataFrame({
        "date": dates,
        "tmean": tmean
    })

    output_path = Path("data/raw/temperature_monthly_clean.csv")
    df.to_csv(output_path, index=False)

    print("temperature_monthly_clean.csv created successfully.")

if __name__ == "__main__":
    main()
