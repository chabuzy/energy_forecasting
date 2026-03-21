import streamlit as st
from src.utils.metadata import load_model_metadata
from src.utils.feature_checker import (
    load_saved_feature_names,
    check_feature_consistency
)
import pandas as pd

st.title("📄 Model Metadata & Consistency Checker")

metadata = load_model_metadata()

if metadata is None:
    st.error("No model metadata found. Train your ML models first.")
    st.stop()

# Display metadata
st.subheader("Model Information")
st.json(metadata)

# Feature consistency check
st.subheader("Feature Engineering Consistency Check")

saved_features = metadata["feature_names"]

# Load current dataset
df = pd.read_csv("data/processed/modelling_dataset.csv", parse_dates=["datetime"])
df = df.set_index("datetime")
df = df.select_dtypes(include=["number"])
current_features = list(df.drop(columns=["demand"]).columns)

check_feature_consistency(current_features, saved_features)
