# src/temperature_processing.py
import pandas as pd

def process_temperature(input_path, output_path):
    # Load monthly temperature
    df = pd.read_csv(input_path, parse_dates=["date"])

    # Expand monthly → daily
    df_daily = df.set_index("date").resample("D").interpolate()

    # Expand daily → half-hourly
    df_half_hourly = df_daily.resample("30min").interpolate()

    # Reset index and rename to datetime
    df_half_hourly = df_half_hourly.reset_index()
    df_half_hourly = df_half_hourly.rename(columns={"date": "datetime"})

    # Save
    df_half_hourly.to_csv(output_path, index=False)
    print(f"Temperature half-hourly saved to {output_path}")
