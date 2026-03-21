# models/arima_model.py
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

def run_arima(train_series, test_series, order=(5, 1, 0), return_mae=False):
    """
    Fit ARIMA and forecast on the test set.
    Returns (model, forecast) or (model, forecast, mae) if return_mae=True.
    """
    model = ARIMA(train_series, order=order)
    fitted = model.fit()

    forecast = fitted.forecast(steps=len(test_series))

    if return_mae:
        mae = mean_absolute_error(test_series, forecast)
        print(f"ARIMA MAE: {mae:.2f}")
        return fitted, forecast, mae

    return fitted, forecast

