import pandas as pd
import numpy as np

def create_time_features(df):
    """
    Adds calendar-based features:
    - hour of day
    - day of week
    - month
    - ISO week number
    """
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["weekofyear"] = df.index.isocalendar().week.astype(int)
    return df

def create_lag_features(df, target_col="demand", lags=[1, 2, 24, 48]):
    """
    Adds lag features such as:
    - demand 1 step ago
    - demand 2 steps ago
    - demand 24 steps ago (12 hours)
    - demand 48 steps ago (24 hours)
    """
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    return df

def create_rolling_features(df, target_col="demand"):
    """
    Adds rolling window statistics:
    - 24-hour rolling mean
    - 7-day rolling mean
    """
    df["rolling_24h"] = df[target_col].rolling(48).mean()
    df["rolling_7d"] = df[target_col].rolling(48 * 7).mean()
    return df

def build_feature_matrix(df):
    """
    Full feature engineering pipeline:
    - time features
    - lag features
    - rolling features
    - drop rows with NaN (from lagging)
    """
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = df.dropna()
    return df

