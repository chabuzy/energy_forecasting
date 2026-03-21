import pandas as pd
from pathlib import Path

def save_forecast(model_name, timestamps, predictions, path="reports/forecasts"):
    """
    Save forecast results for any model (ML, DL, ARIMA, etc.)
    timestamps: array-like of datetime values
    predictions: array-like of forecast values
    """

    Path(path).mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "datetime": timestamps,
        "forecast": predictions
    })

    df.to_csv(f"{path}/{model_name}.csv", index=False)
    print(f"Saved forecast: {path}/{model_name}.csv")
