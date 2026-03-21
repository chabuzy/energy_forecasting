import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from src.evaluation.model_registry import list_available_models, load_model_forecast

st.title("🔍 Time-Series Diagnostics")

df = pd.read_csv("data/processed/modelling_dataset.csv", parse_dates=["datetime"]).set_index("datetime")

models = list_available_models("reports/forecasts")
model_name = st.selectbox("Select model", models)

forecast_df = load_model_forecast(model_name, "reports/forecasts")
residuals = forecast_df["actual"] - forecast_df["forecast"]

# Residual plot
st.subheader("Residual Plot")
fig, ax = plt.subplots(figsize=(12, 4))
residuals.plot(ax=ax)
st.pyplot(fig)

# ACF & PACF
st.subheader("ACF & PACF")
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(residuals, ax=ax[0])
plot_pacf(residuals, ax=ax[1])
st.pyplot(fig)

# ADF Test
st.subheader("ADF Stationarity Test")
adf = adfuller(df["demand"])
st.write(f"ADF Statistic: {adf[0]}")
st.write(f"p-value: {adf[1]}")
