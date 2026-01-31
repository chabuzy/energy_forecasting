import pandas as pd

def build_modelling_dataset(input_path, output_path):
    # Load merged dataset
    df = pd.read_csv(input_path, parse_dates=["timestamp"])

    # Standardise datetime column
    df = df.rename(columns={"timestamp": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Remove duplicates
    df = df.drop_duplicates(subset=["datetime"])

    # Sort chronologically
    df = df.sort_values("datetime")

    # Ensure demand is numeric
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")

    # Fill missing temperature values (linear interpolation)
    df["tmean"] = df["tmean"].interpolate()

    # Time features
    df["hour"] = df["datetime"].dt.hour # type: ignore
    df["dayofweek"] = df["datetime"].dt.dayofweek # type: ignore
    df["month"] = df["datetime"].dt.month # type: ignore

    # Lag features
    df["demand_lag_1"] = df["demand"].shift(1)
    df["demand_lag_48"] = df["demand"].shift(48)

    # Rolling mean (previous day)
    df["demand_roll_48"] = df["demand"].rolling(48).mean()

    # Drop rows with missing values (from lags/rolling)
    df = df.dropna()

    # Save modelling dataset
    df.to_csv(output_path, index=False)
    print(f"Modelling dataset saved to {output_path}")


if __name__ == "__main__":
    build_modelling_dataset(
        "data/processed/demand_temperature_half_hourly.csv",
        "data/processed/modelling_dataset.csv"
    )
