import pandas as pd
from pathlib import Path

def aggregate_all_results():
    files = [
        "reports/model_results_ts.csv",
        "reports/model_results_ml.csv",
        "reports/model_results_prophet.csv",
        "reports/model_results_deep.csv",
    ]

    dfs = []
    for f in files:
        p = Path(f)
        if p.exists():
            dfs.append(pd.read_csv(p))

    if not dfs:
        print("No results files found.")
        return

    final = pd.concat(dfs, ignore_index=True)
    final.to_csv("reports/model_results_all.csv", index=False)
    print("Saved combined model comparison table to reports/model_results_all.csv")

if __name__ == "__main__":
    aggregate_all_results()
