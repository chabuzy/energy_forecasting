import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from lime.lime_tabular import LimeTabularExplainer

st.title("🧠 Explainability Hub (SHAP, LIME, PDP)")

# ---------------------------------------------------------
# Load models
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
# Load feature names + data
# ---------------------------------------------------------
with open("models/feature_names.json", "r") as f:
    feature_names = json.load(f)

df = pd.read_csv("data/processed/modelling_dataset.csv", parse_dates=["datetime"])
df = df.set_index("datetime")
df = df.select_dtypes(include=["number"])
X = df.drop(columns=["demand"])
X = X[feature_names]
y = df["demand"]

# ---------------------------------------------------------
# Prepare SHAP + LIME
# ---------------------------------------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

lime_explainer = LimeTabularExplainer(
    training_data=X.values,
    feature_names=feature_names,
    mode="regression"
)

# ---------------------------------------------------------
# Navigation within explainability
# ---------------------------------------------------------
mode = st.radio(
    "Select explainability view",
    ["Global (SHAP / Permutation / PDP)",
     "Local (SHAP Force / Waterfall / LIME)",
     "Interactions (SHAP Dependence / Interaction)",
     "Robustness (LIME Stability / SHAP–LIME Agreement)"]
)

# =========================================================
# 1. GLOBAL
# =========================================================
if mode.startswith("Global"):
    st.subheader("SHAP Summary Plot")

    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP summary unavailable: {e}")

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
        st.warning(f"Permutation importance unavailable: {e}")

    st.subheader("Partial Dependence Plot (PDP)")
    pdp_feature = st.selectbox("Select feature for PDP", feature_names, key="pdp")
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        PartialDependenceDisplay.from_estimator(model, X, [pdp_feature], ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"PDP unavailable: {e}")

# =========================================================
# 2. LOCAL
# =========================================================
elif mode.startswith("Local"):
    idx = st.slider("Row index for local explanation", 0, len(X)-1, 100)

    st.subheader("SHAP Force Plot")
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

    st.subheader("SHAP Waterfall Plot")
    try:
        fig = shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value,
            shap_values[idx],
            feature_names=feature_names,
            max_display=15,
            show=False
        )
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Waterfall plot unavailable: {e}")

    st.subheader("LIME Local Explanation")
    try:
        exp = lime_explainer.explain_instance(X.iloc[idx].values, model.predict)
        fig = exp.as_pyplot_figure()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"LIME explanation unavailable: {e}")

# =========================================================
# 3. INTERACTIONS
# =========================================================
elif mode.startswith("Interactions"):
    st.subheader("SHAP Dependence Plot")

    feature_to_plot = st.selectbox("Feature for dependence plot", feature_names)
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

    st.subheader("SHAP Interaction Values")
    try:
        shap_interaction = explainer.shap_interaction_values(X)
        f1 = st.selectbox("Feature 1", feature_names, key="f1")
        f2 = st.selectbox("Feature 2", feature_names, key="f2")

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.dependence_plot(
            (feature_names.index(f1), feature_names.index(f2)),
            shap_interaction,
            X,
            feature_names=feature_names,
            show=False
        )
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Interaction values unavailable: {e}")

# =========================================================
# 4. ROBUSTNESS
# =========================================================
else:
    st.subheader("LIME Stability Check")

    idx = st.slider("Row for stability test", 0, len(X)-1, 200)
    runs = st.slider("Number of runs", 3, 20, 5)

    weights = []
    for _ in range(runs):
        exp = lime_explainer.explain_instance(X.iloc[idx].values, model.predict)
        weights.append(dict(exp.as_list()))

    st.write("Raw LIME weights across runs:")
    st.json(weights)

    st.subheader("SHAP–LIME Agreement Score")

    N = st.slider("Top N features", 3, 15, 5)

    # SHAP ranking
    shap_rank = np.argsort(np.abs(shap_values).mean(axis=0))[-N:]
    shap_top = [feature_names[i] for i in shap_rank]

    # LIME ranking (single instance)
    exp = lime_explainer.explain_instance(X.iloc[idx].values, model.predict)
    lime_top = [f for f, _ in exp.as_list()[:N]]

    agreement = len(set(shap_top) & set(lime_top)) / N

    st.write(f"**Agreement Score:** {agreement:.2f}")
    st.write("SHAP Top Features:", shap_top)
    st.write("LIME Top Features:", lime_top)
