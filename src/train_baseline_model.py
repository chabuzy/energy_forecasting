import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

def train_baseline(input_path, n_estimators=300):
    """
    Train a baseline RandomForest model for energy demand forecasting.
    """

    # Load modelling dataset
    df = pd.read_csv(input_path, parse_dates=["datetime"])

    # Feature columns used for forecasting
    feature_cols = [
        "tmean",
        "hour",
        "dayofweek",
        "month",
        "demand_lag_1",
        "demand_lag_48",
        "demand_roll_48",
    ]

    target_col = "demand"

    # Split into features and target
    X = df[feature_cols]
    y = df[target_col]

    # Time-series split (no shuffling)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Baseline RandomForest model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate performance
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Baseline RandomForest MAE: {mae:.2f}")

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("\nFeature Importance:")
    print(importances.sort_values(ascending=False))

    return model


if __name__ == "__main__":
    input_path = Path("data/processed/modelling_dataset.csv")
    train_baseline(input_path)
