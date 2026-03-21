import pandas as pd
from pathlib import Path
import joblib
import json

from src.features.feature_engineering import build_feature_matrix
from src.models.xgboost_model import run_xgboost
from src.models.lightgbm_model import run_lightgbm
from src.evaluation.model_comparison import save_results
from src.evaluation.save_forecast import save_forecast
from src.utils.metadata import save_model_metadata


def run_ml_pipeline():
    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------
    df = pd.read_csv("data/processed/modelling_dataset.csv", parse_dates=["datetime"])
    df = df.set_index("datetime")

    # ---------------------------------------------------------
    # Build ML feature matrix
    # ---------------------------------------------------------
    df = build_feature_matrix(df)

    # Keep only numeric columns (critical for XGBoost/LightGBM)
    df = df.select_dtypes(include=["number"])

    # ---------------------------------------------------------
    # Target + features
    # ---------------------------------------------------------
    y = df["demand"]
    X = df.drop(columns=["demand"])

    # ---------------------------------------------------------
    # Save feature names for Streamlit consistency
    # ---------------------------------------------------------
    Path("models").mkdir(exist_ok=True)
    with open("models/feature_names.json", "w") as f:
        json.dump(list(X.columns), f)

    # ---------------------------------------------------------
    # Train/test split
    # ---------------------------------------------------------
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    results = {"model": [], "mae": []}

    # ---------------------------------------------------------
    # XGBoost
    # ---------------------------------------------------------
    print("Running XGBoost...")
    xgb_model, xgb_preds, xgb_mae = run_xgboost(X_train, y_train, X_test, y_test)

    # Save forecast
    save_forecast("XGBoost", y_test.index, xgb_preds)

    # Save model
    joblib.dump(xgb_model, "models/xgboost_model.pkl")

    # Save metadata
    save_model_metadata(
        model_name="XGBoost",
        mae=xgb_mae,
        feature_names=list(X.columns),
        hyperparameters=xgb_model.get_params()
    )

    results["model"].append("XGBoost")
    results["mae"].append(xgb_mae)

    # ---------------------------------------------------------
    # LightGBM
    # ---------------------------------------------------------
    print("Running LightGBM...")
    lgbm_model, lgbm_preds, lgbm_mae = run_lightgbm(X_train, y_train, X_test, y_test)

    # Save forecast
    save_forecast("LightGBM", y_test.index, lgbm_preds)

    # Save model
    joblib.dump(lgbm_model, "models/lightgbm_model.pkl")

    # Save metadata
    save_model_metadata(
        model_name="LightGBM",
        mae=lgbm_mae,
        feature_names=list(X.columns),
        hyperparameters=lgbm_model.get_params()
    )

    results["model"].append("LightGBM")
    results["mae"].append(lgbm_mae)

    # ---------------------------------------------------------
    # Save results table
    # ---------------------------------------------------------
    save_results(results, path="reports/model_results_ml.csv")


if __name__ == "__main__":
    run_ml_pipeline()
