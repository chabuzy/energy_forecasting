import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluation.model_registry import load_model_forecast
from src.utils.metadata import load_model_metadata

st.title("🏆 Best Model Selector")

# ---------------------------------------------------------
# Load combined model results
# ---------------------------------------------------------
try:
    results = pd.read_csv("reports/model_results_all.csv")
except FileNotFoundError:
    st.error("model_results_all.csv not found. Run your full pipeline first.")
    st.stop()

if results.empty:
    st.error("No model results found in model_results_all.csv.")
    st.stop()

st.subheader("All Models (MAE)")
st.dataframe(results.sort_values("mae"), use_container_width=True)

# ---------------------------------------------------------
# Select best model by MAE
# ---------------------------------------------------------
best_row = results.sort_values("mae").iloc[0]
best_model_name = best_row["model"]
best_mae = best_row["mae"]

st.markdown(f"### 🥇 Best Model: **{best_model_name}**")
st.write(f"**MAE:** {best_mae:.2f}")

# ---------------------------------------------------------
# Try to load metadata (if last trained model matches)
# ---------------------------------------------------------
metadata = load_model_metadata()
if metadata and metadata.get("model_name") == best_model_name:
    st.subheader("Model Metadata")
    st.write(f"**Training Date:** {metadata['training_date']}")
    st.write(f"**Number of Features:** {len(metadata['feature_names'])}")
    with st.expander("Hyperparameters"):
        st.json(metadata["hyperparameters"])
else:
    st.info("No matching metadata found for this best model (or metadata is from a different model).")

# ---------------------------------------------------------
# Plot forecast vs actual if forecast file exists
# ---------------------------------------------------------
st.subheader("Forecast vs Actual")

try:
    forecast_df = load_model_forecast(best_model_name, "reports/forecasts")
except FileNotFoundError:
    st.error(f"No forecast file found for {best_model_name} in reports/forecasts.")
    st.stop()

horizon = st.slider("Show last N points", 100, min(len(forecast_df), 2000), 500)

fig, ax = plt.subplots(figsize=(12, 5))
forecast_df["actual"].iloc[-horizon:].plot(ax=ax, label="Actual", color="black")
forecast_df["forecast"].iloc[-horizon:].plot(ax=ax, label="Forecast", color="tab:blue")
ax.set_title(f"{best_model_name} — Actual vs Forecast (last {horizon} points)")
ax.legend()
st.pyplot(fig)
