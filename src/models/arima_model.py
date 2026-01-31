# models/arima_model.py
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

def run_arima(train_series, test_series, order=(5,1,0)):
    """
    Fit ARIMA model and forecast on the test set.
    """

    # Fit model
    model = ARIMA(train_series, order=order)
    fitted = model.fit()

    # Forecast
    forecast = fitted.forecast(steps=len(test_series))

    # Evaluate
    mae = mean_absolute_error(test_series, forecast)
    print(f"ARIMA MAE: {mae:.2f}")

    return fitted, forecast
