import pandas as pd
from pathlib import Path

def list_available_models(path="reports/forecasts"):
    """
    List model names based on forecast CSVs in the given folder.
    """
    path = Path(path)
    if not path.exists():
        return []
    files = list(path.glob("*.csv"))
    return [f.stem for f in files]

def load_model_forecast(model_name, path="reports/forecasts"):
    """
    Load forecast for a given model as a DataFrame indexed by datetime.
    """
    p = Path(path) / f"{model_name}.csv"
    df = pd.read_csv(p, parse_dates=["datetime"])
    df = df.set_index("datetime")
    return df
