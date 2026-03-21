import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluation.model_registry import list_available_models, load_model_forecast

st.title("📈 Forecast Viewer")

df = pd.read_csv("data/processed/modelling_dataset.csv", parse_dates=["datetime"]).set_index("datetime")

models = list_available_models("reports/forecasts")
if not models:
    st.error("No forecast files found. Run your pipeline first.")
    st.stop()

selected_models = st.multiselect("Select models", models, default=models[:1])
horizon = st.slider("Forecast horizon (last N points)", 100, 2000, 500)

fig, ax = plt.subplots(figsize=(14, 5))
df["demand"].iloc[-horizon:].plot(ax=ax, label="Actual", color="black")

for m in selected_models:
    f = load_model_forecast(m, "reports/forecasts")
    f["forecast"].iloc[-horizon:].plot(ax=ax, label=m)

ax.legend()
ax.set_title(f"Actual vs Forecast (last {horizon} points)")
st.pyplot(fig)
