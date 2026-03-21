# src/run_pipeline.py
import subprocess
import sys

def run(cmd):
    print("\n" + "-"*80)
    print(f"RUNNING: {cmd}")
    print("-"*80)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ ERROR running: {cmd}")
        sys.exit(1)

def main():
    print("\n⚡ Running LIGHTWEIGHT pipeline (Time-Series Only)...\n")

    # Only classical models
    run("python -m src.models.run_time_series_models")

    # Aggregate only TS results
    run("python -m src.evaluation.aggregate_results")

    print("\n✨ Lightweight pipeline complete!")
    print("📊 Results saved to: reports/model_results_all.csv\n")

if __name__ == "__main__":
    main()
