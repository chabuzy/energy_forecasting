from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

def run_sarimax(train_y, test_y, train_exog, test_exog, return_mae=False):
    """
    Lightweight SARIMAX with exogenous regressor(s).
    """
    model = SARIMAX(
        train_y,
        exog=train_exog,
        order=(0, 1, 1),
        seasonal_order=(0, 1, 1, 48),
        simple_differencing=True,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted = model.fit(method="powell", maxiter=5, disp=False)
    forecast = fitted.forecast(steps=len(test_y), exog=test_exog)

    if return_mae:
        mae = mean_absolute_error(test_y, forecast)
        print(f"SARIMAX MAE: {mae:.2f}")
        return fitted, forecast, mae

    return fitted, forecast


    return fitted, forecast


