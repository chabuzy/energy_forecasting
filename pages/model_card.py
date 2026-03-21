import streamlit as st
from src.utils.metadata import load_model_metadata

st.title("📘 Model Card")

metadata = load_model_metadata()

if metadata is None:
    st.error("No metadata found. Train a model first.")
    st.stop()

st.markdown(f"""
# Model Card: {metadata['model_name']} (v{metadata['version']})

## 📌 Overview
This model predicts UK electricity demand using engineered features and machine learning.

## 🧪 Performance
- **MAE:** {metadata['mae']}
- **Training Date:** {metadata['training_date']}

## 🧠 Model Type
- {metadata['model_name']}

## 🔧 Hyperparameters
""")

with st.expander("View Hyperparameters"):
    st.json(metadata["hyperparameters"])

st.markdown("""
## 📊 Intended Use
- Short‑term electricity demand forecasting  
- Operational planning  
- Energy analytics dashboards  

## ⚠ Limitations
- Sensitive to feature drift  
- Requires retraining when new patterns emerge  

## 📁 Dataset
- UK electricity demand  
- Weather features  
- Time‑based features  

## 👤 Author
Chibuzor Okonkwo  
""")
