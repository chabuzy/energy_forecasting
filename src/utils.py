import pandas as pd

def load_dataset(path):
    """
    Load dataset with timestamp parsing.
    """
    return pd.read_csv(path, parse_dates=['timestamp'])


def ensure_sorted(df):
    """
    Ensure dataset is sorted by timestamp.
    """
    return df.sort_values('timestamp').reset_index(drop=True)


def train_test_split_time_series(df, test_size=0.1):
    """
    Split time series into train and test sets.
    """
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test
