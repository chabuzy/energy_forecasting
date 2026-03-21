import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluation.model_registry import list_available_models, load_model_forecast

st.set_page_config(page_title="UK Energy Forecasting", layout="wide")

st.title("UK Energy Demand Forecasting Dashboard")

# Load base demand data
df = pd.read_csv("data/processed/modelling_dataset.csv", parse_dates=["datetime"]).set_index("datetime")

# Load model comparison table if available
results = None
try:
    results = pd.read_csv("reports/model_results_all.csv")
except FileNotFoundError:
    pass

# Sidebar controls
st.sidebar.header("Controls")

models = list_available_models("reports/forecasts")
if not models:
    st.error("No forecast files found in reports/forecasts. Run your pipeline first.")
    st.stop()

selected_models = st.sidebar.multiselect("Select models", models, default=models[:1])
horizon = st.sidebar.slider("Forecast horizon (last N points)", 100, 2000, 500)

# Main plot
st.subheader("Actual vs Forecast")

fig, ax = plt.subplots(figsize=(12, 4))
df["demand"].iloc[-horizon:].plot(ax=ax, label="Actual", color="black")

for m in selected_models:
    forecast_df = load_model_forecast(m, "reports/forecasts")
    forecast_df["forecast"].iloc[-horizon:].plot(ax=ax, label=m)

ax.legend()
ax.set_title(f"Actual vs Forecast (last {horizon} points)")
st.pyplot(fig)

# Model comparison table
if results is not None:
    st.subheader("Model Comparison (MAE)")
    st.dataframe(results.sort_values("mae"), use_container_width=True)
