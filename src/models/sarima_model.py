from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

def run_sarima(train_y, test_y):
    model = SARIMAX(
        train_y,
        order=(0,1,1),
        seasonal_order=(0,1,1,48),
        simple_differencing=True,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted = model.fit(
        method="powell",
        maxiter=5,
        disp=False
    )

    forecast = fitted.forecast(steps=len(test_y))

    mae = mean_absolute_error(test_y, forecast)
    print(f"SARIMA MAE: {mae:.2f}")

    return fitted, forecast



