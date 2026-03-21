import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Model Comparison")

try:
    results = pd.read_csv("reports/model_results_all.csv")
except FileNotFoundError:
    st.error("model_results_all.csv not found. Run your pipeline first.")
    st.stop()

st.subheader("Model Performance (MAE)")
st.dataframe(results.sort_values("mae"), use_container_width=True)

fig, ax = plt.subplots(figsize=(10, 4))
results.sort_values("mae").set_index("model")["mae"].plot(kind="bar", ax=ax)
ax.set_ylabel("MAE")
ax.set_title("Model Comparison (Lower is Better)")
st.pyplot(fig)
