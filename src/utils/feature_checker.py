import json
import streamlit as st

def load_saved_feature_names(path="models/feature_names.json"):
    """Load the feature names saved during training."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("❌ feature_names.json not found. Train your ML models first.")
        return None


def check_feature_consistency(current_features, saved_features):
    """
    Compare current feature list with saved training features.
    Returns True if consistent, False otherwise.
    """

    if saved_features is None:
        return False

    if list(current_features) != list(saved_features):
        st.error("❌ Feature mismatch detected!")
        st.write("### Expected features (from training):")
        st.write(saved_features)

        st.write("### Current features (from dataset):")
        st.write(list(current_features))

        st.warning("""
        Your dataset or feature engineering pipeline has changed.
        Retrain your ML models to update feature_names.json.
        """)
        return False

    st.success("✔ Feature engineering is consistent with training.")
    return True
