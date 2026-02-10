import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Breast Cancer Prediction",
    layout="centered"
)

st.title("Breast Cancer Prediction App")
st.write(
    "Please answer the questions below as accurately as possible. "
    "If you are unsure about any field, you may leave the default values."
)
st.warning("This tool is for educational purposes only and is not a medical diagnosis.")

with st.expander("How to use this tool"):
    st.markdown(
        """
- **Age**: Enter your age in years.
- **Tumor size**: Enter tumor size in millimeters (e.g. 2.5 cm = 25 mm).
- **Lymph nodes**: Enter how many lymph nodes were examined and how many were positive.
- **Hormone receptors**: Select Yes / No if known, otherwise choose Not sure.
"""
    )

# =========================
# Load model
# =========================
MODEL_PATH = "Models/breast_cancer_lr_v1.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Ensure the model exists in the Models folder.")
    st.stop()

model = load_model()

if not hasattr(model, "feature_names_"):
    st.error("Model is missing feature_names_. Re-export the model with embedded feature names.")
    st.stop()

feature_columns = list(model.feature_names_)

# =========================
# Helper functions
# =========================
def make_empty_input(columns):
    return pd.DataFrame([np.zeros(len(columns))], columns=columns)

def set_if_exists(df, col, value):
    if col in df.columns:
        df.at[0, col] = value

def yes_no_unknown(label, help_text=""):
    choice = st.radio(label, ["Not sure", "Yes", "No"], horizontal=True, help=help_text)
    if choice == "Yes":
        return 1
    if choice == "No":
        return 0
    return None

# =========================
# Session defaults
# =========================
if "age" not in st.session_state:
    st.session_state["age"] = 50
if "tumor_size" not in st.session_state:
    st.session_state["tumor_size"] = 20
if "nodes_examined" not in st.session_state:
    st.session_state["nodes_examined"] = 0
if "nodes_positive" not in st.session_state:
    st.session_state["nodes_positive"] = 0

# =========================
# Sample data button
# =========================
col1, col2 = st.columns([1, 2])
with col1:
    if st.button("Fill sample values"):
        st.session_state["age"] = 55
        st.session_state["tumor_size"] = 25
        st.session_state["nodes_examined"] = 12
        st.session_state["nodes_positive"] = 3
        st.rerun()

with col2:
    st.caption("This helps test the app without needing medical knowledge.")

# =========================
# Build input row
# =========================
user_input = make_empty_input(feature_columns)

with st.form("prediction_form"):

    st.header("Personal Information")

    age = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=120,
        value=int(st.session_state["age"]),
        step=1
    )
    set_if_exists(user_input, "Age", float(age))

    st.header("Tumor and Lymph Node Information")

    tumor_size = st.number_input(
        "Tumor size (mm)",
        min_value=0,
        max_value=300,
        value=int(st.session_state["tumor_size"]),
        step=1
    )
    set_if_exists(user_input, "Tumor Size", float(tumor_size))

    nodes_examined = st.number_input(
        "Lymph nodes examined",
        min_value=0,
        max_value=200,
        value=int(st.session_state["nodes_examined"]),
        step=1
    )

    nodes_positive = st.number_input(
        "Lymph nodes positive",
        min_value=0,
        max_value=200,
        value=int(st.session_state["nodes_positive"]),
        step=1
    )

    if nodes_positive > nodes_examined:
        st.error("Positive nodes cannot be greater than nodes examined.")
        st.stop()

    node_ratio = (nodes_positive / nodes_examined) if nodes_examined > 0 else 0.0
    set_if_exists(user_input, "Node_Ratio", node_ratio)

    st.caption(f"Computed lymph node positivity ratio: {node_ratio:.3f}")

    st.header("Hormone Receptor Status")

    er = yes_no_unknown(
        "Estrogen receptor positive?",
        help_text="Select Yes or No if known, otherwise choose Not sure."
    )
    if er is not None:
        set_if_exists(user_input, "Estrogen Status_Positive", er)

    pr = yes_no_unknown(
        "Progesterone receptor positive?",
        help_text="Select Yes or No if known, otherwise choose Not sure."
    )
    if pr is not None:
        set_if_exists(user_input, "Progesterone Status_Positive", pr)

    submitted = st.form_submit_button("Predict")

# =========================
# Prediction
# =========================
if submitted:
    try:
        prediction = model.predict(user_input)[0]
        st.success(f"Prediction result: {prediction}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(user_input)[0]
            classes = model.classes_
            st.write("Prediction probabilities:")
            st.dataframe(pd.DataFrame([probabilities], columns=classes))

        with st.expander("Show model input (debug view)"):
            st.dataframe(user_input)

    except Exception as e:
        st.error("Prediction failed due to an internal error.")
        st.exception(e)

st.caption(
    "Logistic Regression model. "
    "Interface simplified for usability. "
    "Advanced clinical features remain at baseline internally."
)
