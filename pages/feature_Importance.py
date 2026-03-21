import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import json

st.title("🧠 Feature Importance (XGBoost & LightGBM)")

# ---------------------------------------------------------
# Load available models
# ---------------------------------------------------------
model_files = {
    "XGBoost": "models/xgboost_model.pkl",
    "LightGBM": "models/lightgbm_model.pkl"
}

available = [m for m, f in model_files.items() if Path(f).exists()]

if not available:
    st.error("No saved ML models found. Train XGBoost/LightGBM first.")
    st.stop()

model_name = st.selectbox("Select model", available)
model = joblib.load(model_files[model_name])

# ---------------------------------------------------------
# Load feature names saved during training
# ---------------------------------------------------------
with open("models/feature_names.json", "r") as f:
    feature_names = json.load(f)

importances = model.feature_importances_

# ---------------------------------------------------------
# Safety check
# ---------------------------------------------------------
if len(feature_names) != len(importances):
    st.error(
        f"Feature mismatch: model has {len(importances)} importances "
        f"but feature_names.json contains {len(feature_names)} features."
    )
    st.stop()

# ---------------------------------------------------------
# Build importance DataFrame
# ---------------------------------------------------------
imp_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

# ---------------------------------------------------------
# Display table
# ---------------------------------------------------------
st.subheader("Feature Importance Table")
st.dataframe(imp_df, use_container_width=True)

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=imp_df, x="importance", y="feature", ax=ax)
ax.set_title(f"Feature Importance — {model_name}")
st.pyplot(fig)
