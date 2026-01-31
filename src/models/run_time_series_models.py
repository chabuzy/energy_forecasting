import sys
from pathlib import Path
import pandas as pd

# Add project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.arima_model import run_arima
from src.models.sarima_model import run_sarima
from src.models.sarimax_model import run_sarimax

# Load dataset
df = pd.read_csv("data/processed/modelling_dataset.csv", parse_dates=["datetime"])
df = df.set_index("datetime")

df = df.asfreq("30min")
df = df.infer_objects(copy=False)
df = df.interpolate()


# Target
y = df["demand"]

# Exogenous
X = df[["tmean"]]

# Split
split = int(len(df) * 0.8)
y_train, y_test = y[:split], y[split:]
X_train, X_test = X[:split], X[split:]

print("Running ARIMA...")
arima_model, arima_forecast = run_arima(y_train, y_test)

print("Running SARIMA...")
sarima_model, sarima_forecast = run_sarima(y_train, y_test)

print("Running SARIMAX...")
sarimax_model, sarimax_forecast = run_sarimax(y_train, y_test, X_train, X_test)


