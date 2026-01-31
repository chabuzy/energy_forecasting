import pandas as pd

def add_lag_features(df, column='demand', lags=[1, 48, 96]):
    """
    Add lag features for time series forecasting.
    Default lags:
        1 step  = 30 minutes
        48 step = 1 day
        96 step = 2 days
    """
    for lag in lags:
        df[f'{column}_lag_{lag}'] = df[column].shift(lag)
    return df


def add_rolling_features(df, column='demand', windows=[48, 96, 336]):
    """
    Add rolling mean features.
    Default windows:
        48  = 1 day
        96  = 2 days
        336 = 1 week
    """
    for w in windows:
        df[f'{column}_rollmean_{w}'] = df[column].rolling(w).mean()
    return df


def add_time_features(df):
    """
    Add useful datetime features.
    """
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    return df
