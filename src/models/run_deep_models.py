"""
run_deep_models.py
-------------------
Trains LSTM and TCN deep learning models for energy demand forecasting.

This script:
1. Loads the processed dataset
2. Converts the time series into supervised learning sequences
3. Splits into train/test sets
4. Trains LSTM and TCN models
5. Saves models (.keras format)
6. Saves forecasts to reports/forecasts/
7. Saves MAE results to reports/model_results_deep.csv
"""

import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error

# --- Import model builders + sequence preparation
from src.models.lstm_tcn_model import (
    create_lstm_model,
    create_tcn_model,
    prepare_sequences
)

# --- Forecast saving utilities
from src.evaluation.save_forecast import save_forecast
from src.evaluation.model_comparison import save_results


def run_deep_learning():
    # ============================================================
    # 1. LOAD DATA
    # ============================================================
    df = pd.read_csv("data/processed/modelling_dataset.csv", parse_dates=["datetime"])
    df = df.set_index("datetime")

    # Extract raw demand values
    series = df["demand"].values

    # Extract timestamps (needed for saving forecasts)
    timestamps = df.index.values

    # ============================================================
    # 2. PREPARE SUPERVISED SEQUENCES
    # ============================================================
    # Convert series → sliding windows (X) and next-step targets (y)
    X, y = prepare_sequences(series, window=48)

    # Align timestamps with y (first 48 timestamps are lost)
    timestamps = timestamps[48:]

    # ============================================================
    # 3. TRAIN/TEST SPLIT
    # ============================================================
    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Split timestamps for test set
    ts_train, ts_test = timestamps[:split], timestamps[split:]

    # Prepare results dictionary
    results = {"model": [], "mae": []}

    # Ensure models folder exists
    Path("models").mkdir(exist_ok=True)

    # ============================================================
    # 4. TRAIN LSTM MODEL
    # ============================================================
    print("Running LSTM...")

    lstm_model = create_lstm_model((48, 1))
    lstm_model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1)

    # Predict
    lstm_preds = lstm_model.predict(X_test).flatten()

    # Evaluate
    lstm_mae = mean_absolute_error(y_test, lstm_preds)
    print(f"LSTM MAE: {lstm_mae:.2f}")

    # Save model (modern Keras format)
    lstm_model.save("models/lstm_model.keras")

    # Save forecast CSV
    save_forecast("LSTM", ts_test, lstm_preds)

    # Log results
    results["model"].append("LSTM")
    results["mae"].append(lstm_mae)

    # ============================================================
    # 5. TRAIN TCN MODEL
    # ============================================================
    print("Running TCN...")

    tcn_model = create_tcn_model((48, 1))
    tcn_model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1)

    # Predict
    tcn_preds = tcn_model.predict(X_test).flatten()

    # Evaluate
    tcn_mae = mean_absolute_error(y_test, tcn_preds)
    print(f"TCN MAE: {tcn_mae:.2f}")

    # Save model
    tcn_model.save("models/tcn_model.keras")

    # Save forecast CSV
    save_forecast("TCN", ts_test, tcn_preds)

    # Log results
    results["model"].append("TCN")
    results["mae"].append(tcn_mae)

    # ============================================================
    # 6. SAVE RESULTS
    # ============================================================
    save_results(results, path="reports/model_results_deep.csv")


if __name__ == "__main__":
    run_deep_learning()
