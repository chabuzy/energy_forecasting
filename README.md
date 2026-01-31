# ⚡ ENERGY_FORECASTING

A professional-grade time series forecasting pipeline for UK electricity demand using NESO half-hourly data and Met Office temperature records. Built with Python, SARIMAX, Prophet, and modular forecasting models. Designed for reproducibility, scenario analysis, and portfolio presentation.

---

## 📁 Project Structure

# National Grid Demand & Temperature Forecasting

This project builds a reproducible forecasting pipeline that combines
NESO half-hourly electricity demand with Met Office temperature data
(Sutton Bonington) to model and analyse the impact of weather on demand.

## Project structure

```text
ENERGY_FORECASTING/
├── data/
│   ├── raw/          # Original NESO & Met Office files
│   ├── processed/    # Half-hourly temperature, merged datasets
├── notebooks/
│   ├── prepare_temperature.ipynb
│   ├── merge_datasets.ipynb
│   ├── forecasting_pipeline.ipynb
│   ├── model_comparison.ipynb
│   ├── scenario_analysis.ipynb
│   ├── arima_sarima_sarimax.ipynb
│   ├── baseline_models.ipynb
│   ├── stationarity_test.ipynb
├── src/
│   ├── temperature_processing.py
│   ├── merge_temperature.py
│   ├── feature_engineering.py
│   ├── evaluation.py
│   ├── utils.py
│   ├── models/
│   │   ├── arima_model.py
│   │   ├── sarima_model.py
│   │   ├── sarimax_model.py
├── venv/
├── requirements.txt
└── README.md

Pipeline overview
Temperature processing

src/temperature_processing.py

Converts monthly Sutton Bonington temperature to daily, then half-hourly.

Output: data/processed/temperature_half_hourly.csv

Demand–temperature merge

src/merge_temperature.py

Merges NESO half-hourly demand with half-hourly temperature.

Output: data/processed/demand_temperature_half_hourly.csv

Feature engineering

src/feature_engineering.py

Adds lags, rolling means, and time-based features.

Modelling & evaluation

notebooks/forecasting_pipeline.ipynb

notebooks/model_comparison.ipynb

Uses ARIMA, SARIMA, SARIMAX and baselines.

Metrics: MAE, RMSE, MAPE (from src/evaluation.py).

Scenario analysis

notebooks/scenario_analysis.ipynb

Explores demand under warmer/colder temperature scenarios using SARIMAX.
