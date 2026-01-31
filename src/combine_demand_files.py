import pandas as pd
from pathlib import Path

def combine_demand_files(raw_dir, output_path):
    raw_dir = Path(raw_dir)
    all_files = sorted(raw_dir.glob("demanddata_*.csv"))

    dfs = []
    for file in all_files:
        df = pd.read_csv(file)

        required_cols = ['SETTLEMENT_DATE', 'SETTLEMENT_PERIOD']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"{file.name} is missing required column: {col}")

        df['SETTLEMENT_DATE'] = pd.to_datetime(df['SETTLEMENT_DATE'])

        df['timestamp'] = df['SETTLEMENT_DATE'] + pd.to_timedelta(
            (df['SETTLEMENT_PERIOD'] - 1) * 30, unit='m'
        )

        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)

    combined.to_csv(output_path, index=False)
    print(f"Combined demand saved to {output_path}")
