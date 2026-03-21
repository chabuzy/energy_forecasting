# ⚡ ENERGY_FORECASTING

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-ff4b4b" />
  <img src="https://img.shields.io/badge/Models-ML%20%7C%20DL%20%7C%20TS-green" />
  <img src="https://img.shields.io/badge/Explainability-SHAP%20%7C%20LIME-orange" />
  <img src="https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

A professional-grade time series forecasting pipeline for UK electricity demand using NESO half-hourly data and Met Office temperature records. Built with Python, SARIMAX, Prophet, and modular forecasting models. Designed for reproducibility, scenario analysis, and portfolio presentation.

---

## 📁 Project Structure

# National Grid Demand & Temperature Forecasting

This project builds a reproducible forecasting pipeline that combines
NESO half-hourly electricity demand with Met Office temperature data
(Sutton Bonington) to model and analyse the impact of weather on demand.

# ⚡ UK Energy Demand Forecasting

End‑to‑end forecasting of UK electricity demand using:
- NESO demand data  
- Met Office temperature data  
- Classical time‑series models (Naive, ARIMA, SARIMA, SARIMAX)  
- Machine learning models (XGBoost, LightGBM)  
- Prophet  
- Deep learning (LSTM / TCN – planned)
- Streamlit dashboard for visualisation

The project is structured as a reproducible pipeline: from raw data → engineered features → multiple models → comparison and visualisation.

---

## 📁 Project structure

```text
energy_forecasting/
│
├── data/
│   ├── raw/                     # Raw NESO + Met Office data
│   ├── interim/                 # Cleaned intermediate files
│   └── processed/               # Modelling dataset(s)
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_comparison.ipynb
│
├── reports/
│   ├── figures/                 # Plots (actual vs forecast, etc.)
│   └── model_results.csv        # Model comparison table
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── clean_temperature.py
│   │   ├── merge_datasets.py
│   │   └── build_modelling_dataset.py
│   │
│   ├── features/
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── arima_model.py
│   │   ├── sarima_model.py
│   │   ├── sarimax_model.py
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── prophet_model.py
│   │   └── lstm_tcn_model.py
│   │
│   └── evaluation/
│       ├── metrics.py
│       └── model_comparison.py
│
├── .gitignore
├── README.md
└── requirements.txt

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

flowchart TD

    A[Raw Data<br>NESO + Met Office] --> B[Data Processing<br>Cleaning, Merging, Resampling]
    B --> C[Feature Engineering<br>Lags, Rolling, Weather, Calendar]
    C --> D[Train/Test Split]

    D --> E1[Classical Models<br>ARIMA, SARIMA, SARIMAX]
    D --> E2[Machine Learning<br>XGBoost, LightGBM]
    D --> E3[Deep Learning<br>LSTM, TCN]
    D --> E4[Prophet]

    E1 --> F[Forecast Registry]
    E2 --> F
    E3 --> F
    E4 --> F

    F --> G[Model Comparison<br>MAE Ranking]
    G --> H[Best Model Selector]

    H --> I[Explainability Hub<br>SHAP, LIME, PDP, Interactions]

    I --> J[Streamlit Dashboard<br>10+ Pages]

    J --> K[Deployment<br>Docker + GitHub Actions]
