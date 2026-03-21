from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from sklearn.metrics import mean_absolute_error
import pandas as pd

def prepare_prophet_df(df):
    out = df.reset_index().rename(columns={"datetime": "ds", "demand": "y"})
    if "tmean" in out.columns:
        out["tmean"] = out["tmean"]
    return out

def run_prophet(train_df, test_df, use_temp=True, add_holidays=True):
    model = Prophet()

    if add_holidays:
        # UK holidays via built-in country support
        model.add_country_holidays(country_name="UK")

    if use_temp and "tmean" in train_df.columns:
        model.add_regressor("tmean")

    model.fit(train_df)
    forecast = model.predict(test_df)
    mae = mean_absolute_error(test_df["y"], forecast["yhat"])
    print(f"Prophet MAE: {mae:.2f}")
    return model, forecast, mae
