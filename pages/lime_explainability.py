import streamlit as st
import pandas as pd
import joblib
import json
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

st.title("🔍 LIME Local Explainability")

# Load model
model_files = {
    "XGBoost": "models/xgboost_model.pkl",
    "LightGBM": "models/lightgbm_model.pkl"
}

model_name = st.selectbox("Select model", list(model_files.keys()))
model = joblib.load(model_files[model_name])

# Load feature names
with open("models/feature_names.json", "r") as f:
    feature_names = json.load(f)

# Load dataset
df = pd.read_csv("data/processed/modelling_dataset.csv", parse_dates=["datetime"])
df = df.set_index("datetime")
df = df.select_dtypes(include=["number"])
X = df.drop(columns=["demand"])
X = X[feature_names]
y = df["demand"]

# LIME explainer
explainer = LimeTabularExplainer(
    training_data=X.values,
    feature_names=feature_names,
    mode="regression"
)

# Select instance
idx = st.slider("Select row index to explain", 0, len(X)-1, 100)
instance = X.iloc[idx]

exp = explainer.explain_instance(instance.values, model.predict)

st.subheader("LIME Explanation")
fig = exp.as_pyplot_figure()
st.pyplot(fig)

# ---------------------------------------------------------
# Compare two LIME explanations
# ---------------------------------------------------------
st.subheader("Compare Two Predictions")

idx1 = st.number_input("Row 1", min_value=0, max_value=len(X)-1, value=50)
idx2 = st.number_input("Row 2", min_value=0, max_value=len(X)-1, value=100)

exp1 = explainer.explain_instance(X.iloc[idx1].values, model.predict)
exp2 = explainer.explain_instance(X.iloc[idx2].values, model.predict)

col1, col2 = st.columns(2)

with col1:
    st.write(f"Explanation for row {idx1}")
    st.pyplot(exp1.as_pyplot_figure())

with col2:
    st.write(f"Explanation for row {idx2}")
    st.pyplot(exp2.as_pyplot_figure())

# ---------------------------------------------------------
# LIME Stability Check
# ---------------------------------------------------------
st.subheader("LIME Stability Check")

idx = st.slider("Row for stability test", 0, len(X)-1, 200)
runs = st.slider("Number of runs", 3, 20, 5)

weights = []

for _ in range(runs):
    exp = explainer.explain_instance(X.iloc[idx].values, model.predict)
    weights.append(dict(exp.as_list()))

st.write("### Stability Across Runs")
st.json(weights)

# ---------------------------------------------------------
# SHAP–LIME Agreement Score
# ---------------------------------------------------------
st.subheader("SHAP–LIME Agreement Score")

N = st.slider("Top N features", 3, 15, 5)

# SHAP ranking
shap_rank = np.argsort(np.abs(shap_values).mean(axis=0))[-N:]
shap_top = [feature_names[i] for i in shap_rank]

# LIME ranking
exp = explainer.explain_instance(X.iloc[0].values, model.predict)
lime_top = [f for f, _ in exp.as_list()[:N]]

agreement = len(set(shap_top) & set(lime_top)) / N

st.write(f"Agreement Score: **{agreement:.2f}**")
st.write("SHAP Top Features:", shap_top)
st.write("LIME Top Features:", lime_top)

# ---------------------------------------------------------
# SHAP Waterfall Plot
# ---------------------------------------------------------
st.subheader("SHAP Waterfall Plot")

idx = st.slider("Row for waterfall", 0, len(X)-1, 150)

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
