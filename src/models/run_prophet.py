import pandas as pd
from src.models.prophet_model import prepare_prophet_df, run_prophet
from src.evaluation.model_comparison import save_results
from src.evaluation.save_forecast import save_forecast

def run_prophet_pipeline():
    # Load dataset
    df = pd.read_csv("data/processed/modelling_dataset.csv", parse_dates=["datetime"])
    df = df.set_index("datetime")

    # Prepare Prophet-style dataframe
    prophet_df = prepare_prophet_df(df)

    # Train/test split
    split = int(len(prophet_df) * 0.8)
    train_df = prophet_df.iloc[:split]
    test_df = prophet_df.iloc[split:]

    print("Running Prophet...")
    model, forecast, mae = run_prophet(train_df, test_df, use_temp=True, add_holidays=True)

    # Align forecast with test_df for saving
    forecast_out = forecast[["ds", "yhat"]].copy()
    forecast_out = forecast_out.set_index("ds")
    test_y = test_df.set_index("ds")["y"]

    save_forecast("Prophet", test_y, forecast_out["yhat"])

    # Save metrics
    save_results(
        {"model": ["Prophet"], "mae": [mae]},
        path="reports/model_results_prophet.csv"
    )

if __name__ == "__main__":
    run_prophet_pipeline()
