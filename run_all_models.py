#Full End‑to‑End Pipeline (ALL models)
import subprocess

def run_cmd(cmd):
    print(f"\n=== {cmd} ===")
    subprocess.run(cmd, shell=True, check=True)

def main():
    # 1. Time-series models
    run_cmd("python -m src.models.run_time_series_models")

    # 2. ML models (XGBoost, LightGBM)
    run_cmd("python -m src.models.run_ml_models")

    # 3. Prophet
    run_cmd("python -m src.models.run_prophet")

    # 4. Deep models (LSTM, TCN)
    run_cmd("python -m src.models.run_deep_models")

    # 5. Aggregate results
    run_cmd("python -m src.evaluation.aggregate_results")

if __name__ == "__main__":
    main()
