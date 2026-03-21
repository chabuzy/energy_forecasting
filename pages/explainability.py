import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from pathlib import Path
import json

st.title("🧠 Model Explainability (SHAP + Permutation Importance)")

# ---------------------------------------------------------
# Load available ML models
# ---------------------------------------------------------
model_files = {
    "XGBoost": "models/xgboost_model.pkl",
    "LightGBM": "models/lightgbm_model.pkl"
}

available = [m for m, f in model_files.items() if Path(f).exists()]

if not available:
    st.error("No ML models found. Train XGBoost/LightGBM first.")
    st.stop()

model_name = st.selectbox("Select model", available)
model = joblib.load(model_files[model_name])

# ---------------------------------------------------------
# Load feature names
# ---------------------------------------------------------
with open("models/feature_names.json", "r") as f:
    feature_names = json.load(f)

# ---------------------------------------------------------
# Load dataset for explainability
# ---------------------------------------------------------
df = pd.read_csv("data/processed/modelling_dataset.csv", parse_dates=["datetime"])
df = df.set_index("datetime")
df = df.select_dtypes(include=["number"])

X = df.drop(columns=["demand"])
y = df["demand"]

# Align features with saved names
X = X[feature_names]

# ---------------------------------------------------------
# SHAP Explainability
# ---------------------------------------------------------
st.subheader("SHAP Summary Plot")

try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    st.pyplot(fig)

except Exception as e:
    st.warning(f"SHAP could not be computed for {model_name}: {e}")

# ---------------------------------------------------------
# Permutation Importance
# ---------------------------------------------------------
st.subheader("Permutation Importance")

try:
    perm = permutation_importance(model, X, y, n_repeats=10, random_state=42)

    perm_df = pd.DataFrame({
        "feature": feature_names,
        "importance": perm.importances_mean
    }).sort_values("importance", ascending=False)

    st.dataframe(perm_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(perm_df["feature"], perm_df["importance"])
    ax.set_title("Permutation Importance")
    ax.invert_yaxis()
    st.pyplot(fig)

except Exception as e:
    st.warning(f"Permutation importance could not be computed: {e}")

# ---------------------------------------------------------
# Combined Comparison
# ---------------------------------------------------------
st.subheader("Feature Importance Comparison")

try:
    # Tree-based importance
    tree_imp = model.feature_importances_

    comparison_df = pd.DataFrame({
        "feature": feature_names,
        "tree_importance": tree_imp,
        "permutation_importance": perm.importances_mean
    }).sort_values("tree_importance", ascending=False)

    st.dataframe(comparison_df, use_container_width=True)

except Exception:
    st.info("Comparison unavailable — missing SHAP or permutation importance.")


# ---------------------------------------------------------
# SHAP Dependence Plot
# ---------------------------------------------------------
st.subheader("SHAP Dependence Plot")

feature_to_plot = st.selectbox("Select feature for dependence plot", feature_names)

try:
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.dependence_plot(
        feature_to_plot,
        shap_values,
        X,
        feature_names=feature_names,
        show=False
    )
    st.pyplot(fig)

except Exception as e:
    st.warning(f"Dependence plot unavailable: {e}")

# ---------------------------------------------------------
# Partial Dependence Plot (PDP)
# ---------------------------------------------------------
from sklearn.inspection import PartialDependenceDisplay

st.subheader("Partial Dependence Plot (PDP)")

pdp_feature = st.selectbox("Select feature for PDP", feature_names, key="pdp")

try:
    fig, ax = plt.subplots(figsize=(10, 6))
    PartialDependenceDisplay.from_estimator(
        model,
        X,
        [pdp_feature],
        ax=ax
    )
    st.pyplot(fig)

except Exception as e:
    st.warning(f"PDP could not be computed: {e}")

# ---------------------------------------------------------
# SHAP Interaction Values
# ---------------------------------------------------------
st.subheader("SHAP Interaction Values")

try:
    shap_interaction = explainer.shap_interaction_values(X)

    feature_i = st.selectbox("Feature 1", feature_names, key="f1")
    feature_j = st.selectbox("Feature 2", feature_names, key="f2")

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.dependence_plot(
        (feature_names.index(feature_i), feature_names.index(feature_j)),
        shap_interaction,
        X,
        feature_names=feature_names,
        show=False
    )
    st.pyplot(fig)

except Exception as e:
    st.warning(f"Interaction values unavailable: {e}")


# ---------------------------------------------------------
# SHAP Force Plot (Local Explanation)
# ---------------------------------------------------------
st.subheader("SHAP Force Plot")

idx = st.slider("Select row index", 0, len(X)-1, 100)

try:
    shap_value_single = shap_values[idx]

    fig = shap.force_plot(
        explainer.expected_value,
        shap_value_single,
        X.iloc[idx],
        matplotlib=True,
        show=False
    )
    st.pyplot(fig)

except Exception as e:
    st.warning(f"Force plot unavailable: {e}")
