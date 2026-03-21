import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🔥 Heatmaps & Monthly/Quarterly Trends")

df = pd.read_csv("data/processed/modelling_dataset.csv", parse_dates=["datetime"])
df["month"] = df["datetime"].dt.month
df["hour"] = df["datetime"].dt.hour
df["quarter"] = df["datetime"].dt.quarter

# Monthly average
st.subheader("Monthly Average Demand")
monthly = df.groupby("month")["demand"].mean()
st.line_chart(monthly)

# Quarterly average
st.subheader("Quarterly Average Demand")
quarterly = df.groupby("quarter")["demand"].mean()
st.line_chart(quarterly)

# Heatmap (hour × month)
st.subheader("Heatmap: Hour vs Month")
pivot = df.pivot_table(values="demand", index="hour", columns="month", aggfunc="mean")

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(pivot, cmap="viridis", ax=ax)
st.pyplot(fig)
