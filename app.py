"""
Building Energy Efficiency Predictor
COM763 Advanced Machine Learning

Loads two trained Gradient Boosting pipelines (heating and cooling) and predicts
building energy loads from eight design features. Inputs are locked to the values
present in the training data.
"""

import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------------------------
# Load the saved models (a dict: {"heating": pipeline, "cooling": pipeline})
# ----------------------------------------------------------------------
@st.cache_resource
def load_models():
    return joblib.load("model.pkl")

models = load_models()

# Exact values observed in the dataset, so users cannot enter impossible inputs
VALUES = {
    "X1": [0.62, 0.64, 0.66, 0.69, 0.71, 0.74, 0.76, 0.79, 0.82, 0.86, 0.90, 0.98],
    "X2": [514.5, 563.5, 588.0, 612.5, 637.0, 661.5, 686.0, 710.5, 735.0, 759.5, 784.0, 808.5],
    "X3": [245.0, 269.5, 294.0, 318.5, 343.0, 367.5, 416.5],
    "X4": [110.25, 122.5, 147.0, 220.5],
    "X5": [3.5, 7.0],
    "X6": [2, 3, 4, 5],
    "X7": [0.0, 0.1, 0.25, 0.4],
    "X8": [0, 1, 2, 3, 4, 5],
}

LABELS = {
    "X1": "Relative compactness",
    "X2": "Surface area",
    "X3": "Wall area",
    "X4": "Roof area",
    "X5": "Overall height",
    "X6": "Orientation (dataset code)",
    "X7": "Glazing area",
    "X8": "Glazing area distribution (dataset code)",
}

# Column order the pipelines were trained on
FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]

# ----------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------
st.title("Building Energy Efficiency Predictor")
st.caption(
    "Estimates the heating and cooling load of a residential building from its design. "
    "A decision-support prototype, not a substitute for professional engineering assessment."
)

st.subheader("Building design")
col1, col2 = st.columns(2)

choices = {}
for i, feat in enumerate(FEATURES):
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        # index=len//2 gives a sensible mid-range default
        default = len(VALUES[feat]) // 2
        choices[feat] = st.selectbox(LABELS[feat], VALUES[feat], index=default)

# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
if st.button("Predict energy loads", type="primary"):
    row = pd.DataFrame([{f: choices[f] for f in FEATURES}])

    heating = models["heating"].predict(row)[0]
    cooling = models["cooling"].predict(row)[0]

    st.subheader("Predicted loads")
    m1, m2 = st.columns(2)
    m1.metric("Heating load", f"{heating:.2f}")
    m2.metric("Cooling load", f"{cooling:.2f}")
    st.caption("Values are in the same units as the dataset targets. Confirm the exact unit "
               "against the dataset documentation before quoting it.")

    st.info(
        "This model interpolates within the range of building shapes it was trained on. "
        "Some individually valid selections can combine into a shape that never appeared in "
        "training, and predictions for such combinations are less reliable. Heating load is "
        "predicted more accurately than cooling load."
    )
