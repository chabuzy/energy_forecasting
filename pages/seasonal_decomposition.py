import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

st.title("📆 Seasonal Decomposition")

df = pd.read_csv("data/processed/modelling_dataset.csv", parse_dates=["datetime"]).set_index("datetime")

model_type = st.selectbox("Model type", ["additive", "multiplicative"])

decomp = seasonal_decompose(df["demand"], model=model_type, period=48)

fig = decomp.plot()
fig.set_size_inches(12, 8)
st.pyplot(fig)
