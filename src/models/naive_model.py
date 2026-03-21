from sklearn.metrics import mean_absolute_error

def run_naive(train_y, test_y, lag=48):
    """
    Naive baseline: forecast = value lag steps ago (e.g. same time yesterday).
    """
    forecast = test_y.shift(lag)
    forecast = forecast.fillna(method="bfill")
    mae = mean_absolute_error(test_y, forecast)
    print(f"Naive (lag={lag}) MAE: {mae:.2f}")
    return forecast, mae
