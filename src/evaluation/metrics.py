#This module standardises evaluation:
#MAE: average absolute error
#RMSE: penalises large errors
#MAPE: percentage error
#Every model uses this for consistent comparison

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def compute_metrics(y_true, y_pred):
    """
    Computes standard forecasting metrics:
    - MAE
    - RMSE
    - MAPE
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }
