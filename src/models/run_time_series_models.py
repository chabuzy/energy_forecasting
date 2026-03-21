import pandas as pd

from src.models.arima_model import run_arima
from src.models.sarima_model import run_sarima
from src.models.sarimax_model import run_sarimax
from src.models.naive_model import run_naive

from src.evaluation.model_comparison import save_results
from src.evaluation.save_forecast import save_forecast

def run_time_series_pipeline():
    # Load dataset
    df = pd.read_csv("data/processed/modelling_dataset.csv", parse_dates=["datetime"])
    df = df.set_index("datetime")

    # Target and exogenous
    y = df["demand"]
    X = df[["tmean"]] if "tmean" in df.columns else None

    # Train/test split
    split = int(len(df) * 0.8)
    y_train, y_test = y[:split], y[split:]
    X_train, X_test = (X[:split], X[split:]) if X is not None else (None, None)

    results = {"model": [], "mae": []}

    # ---------------- Naive baseline ----------------
    print("Running Naive baseline...")
    naive_forecast, naive_mae = run_naive(y_train, y_test, lag=48)
    save_forecast("Naive", y_test, naive_forecast)
    results["model"].append("Naive")
    results["mae"].append(naive_mae)

    # ---------------- ARIMA ----------------
    print("Running ARIMA...")
    arima_model, arima_forecast, arima_mae = run_arima(y_train, y_test, return_mae=True)
    save_forecast("ARIMA", y_test, arima_forecast)
    results["model"].append("ARIMA")
    results["mae"].append(arima_mae)

    # ---------------- SARIMA ----------------
    print("Running SARIMA...")
    sarima_model, sarima_forecast, sarima_mae = run_sarima(y_train, y_test, return_mae=True)
    save_forecast("SARIMA", y_test, sarima_forecast)
    results["model"].append("SARIMA")
    results["mae"].append(sarima_mae)

    # ---------------- SARIMAX ----------------
    if X_train is not None:
        print("Running SARIMAX...")
        sarimax_model, sarimax_forecast, sarimax_mae = run_sarimax(
            y_train, y_test, X_train, X_test, return_mae=True
        )
        save_forecast("SARIMAX", y_test, sarimax_forecast)
        results["model"].append("SARIMAX")
        results["mae"].append(sarimax_mae)

    # Save results
    save_results(results, path="reports/model_results_ts.csv")

if __name__ == "__main__":
    run_time_series_pipeline()
