import pandas as pd

def save_results(results, path="reports/model_results.csv"):
    """
    Save a dict like {"model": [...], "mae": [...]} to CSV.
    """
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    print(f"Saved model comparison to {path}")
