 # Full End‑to‑End Pipeline (ALL models)runs:
#Time‑series models (Naive, ARIMA, SARIMA, SARIMAX)
#ML models (XGBoost, LightGBM)
#Prophet
#Deep learning (LSTM, TCN)
#Aggregates results
#Prints a clean summary
import subprocess
import sys

def run(cmd):
    print("\n" + "="*80)
    print(f"RUNNING: {cmd}")
    print("="*80)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ ERROR running: {cmd}")
        sys.exit(1)

def main():
    print("\n🚀 Starting FULL forecasting pipeline...\n")

    # 1. Time-series models
    run("python -m src.models.run_time_series_models")

    # 2. Machine learning models
    run("python -m src.models.run_ml_models")

    # 3. Prophet
    run("python -m src.models.run_prophet")

    # 4. Deep learning models
    run("python -m src.models.run_deep_models")

    # 5. Aggregate results
    run("python -m src.evaluation.aggregate_results")

    print("\n🎉 Pipeline complete!")
    print("📊 Combined results saved to: reports/model_results_all.csv")
    print("📈 Forecasts saved to: reports/forecasts/")
    print("📊 View results in Streamlit: streamlit run app.py\n")

if __name__ == "__main__":
    main()
