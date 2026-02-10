import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺", layout="centered")
st.title("🩺 Breast Cancer Prediction App")
st.write("Enter key clinical details. Categorical fields are simplified into dropdowns (dummy columns handled internally).")

# -------------------------
# Load ONE model (feature names embedded)
# -------------------------
MODEL_PATH = "Models/breast_cancer_lr_v1.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Make sure you exported the model into the Models/ folder.")
    st.stop()

model = load_model()

# Feature columns are embedded into the model during export
if not hasattr(model, "feature_names_"):
    st.error("Model does not contain embedded feature names (feature_names_). Re-export your model embedding X.columns.")
    st.stop()

feature_columns = list(model.feature_names_)

# -------------------------
# Helpers
# -------------------------
def make_empty_input(columns):
    return pd.DataFrame([np.zeros(len(columns))], columns=columns)

def set_binary(df, col_name, value01):
    if col_name in df.columns:
        df.at[0, col_name] = int(value01)

def one_hot_cols(prefix):
    return [c for c in feature_columns if c.startswith(prefix + "_")]

def build_options(prefix, baseline_label="(baseline)"):
    """
    Because you used drop_first=True, one category is missing.
    We add a baseline option to represent "all zeros" for that group.
    """
    cols = one_hot_cols(prefix)
    opts = [c.replace(prefix + "_", "") for c in cols]
    return cols, [baseline_label] + opts

def set_one_hot(df, prefix, chosen_value, baseline_label="(baseline)"):
    cols = one_hot_cols(prefix)

    # clear group
    for c in cols:
        df.at[0, c] = 0

    # baseline => all zeros
    if chosen_value == baseline_label:
        return

    target = f"{prefix}_{chosen_value}"
    if target in df.columns:
        df.at[0, target] = 1

def diff_cols():
    return [c for c in feature_columns if c.startswith("differentiate_")]

def build_diff_options(baseline_label="(baseline)"):
    cols = diff_cols()
    opts = [c.replace("differentiate_", "") for c in cols]
    return cols, [baseline_label] + opts

def set_diff(df, chosen_value, baseline_label="(baseline)"):
    cols = diff_cols()
    for c in cols:
        df.at[0, c] = 0
    if chosen_value == baseline_label:
        return
    target = f"differentiate_{chosen_value}"
    if target in df.columns:
        df.at[0, target] = 1

# -------------------------
# UI
# -------------------------
user_input = make_empty_input(feature_columns)

with st.form("predict_form"):
    st.subheader("Numeric fields")

    if "Age" in user_input.columns:
        user_input.at[0, "Age"] = st.number_input("Age", min_value=0.0, step=1.0, value=0.0)

    if "Tumor Size" in user_input.columns:
        user_input.at[0, "Tumor Size"] = st.number_input("Tumor Size", min_value=0.0, step=1.0, value=0.0)

    # ✅ New merged feature
    if "Node_Ratio" in user_input.columns:
        user_input.at[0, "Node_Ratio"] = st.slider(
            "Lymph Node Positivity Ratio (Positive / Examined)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            help="Higher means a larger proportion of examined lymph nodes were cancer-positive."
        )

    st.subheader("Categorical fields (dropdowns)")

    # Marital Status
    m_cols, m_opts = build_options("Marital Status", baseline_label="(baseline)")
    if m_cols:
        chosen = st.selectbox("Marital Status", m_opts, index=0)
        set_one_hot(user_input, "Marital Status", chosen, baseline_label="(baseline)")

    # T Stage
    t_cols, t_opts = build_options("T Stage", baseline_label="(baseline)")
    if t_cols:
        chosen = st.selectbox("T Stage", t_opts, index=0)
        set_one_hot(user_input, "T Stage", chosen, baseline_label="(baseline)")

    # N Stage
    n_cols, n_opts = build_options("N Stage", baseline_label="(baseline)")
    if n_cols:
        chosen = st.selectbox("N Stage", n_opts, index=0)
        set_one_hot(user_input, "N Stage", chosen, baseline_label="(baseline)")

    # 6th Stage
    s_cols, s_opts = build_options("6th Stage", baseline_label="(baseline)")
    if s_cols:
        chosen = st.selectbox("6th Stage", s_opts, index=0)
        set_one_hot(user_input, "6th Stage", chosen, baseline_label="(baseline)")

    # Differentiation
    d_cols, d_opts = build_diff_options(baseline_label="(baseline)")
    if d_cols:
        chosen = st.selectbox("Differentiation", d_opts, index=0)
        set_diff(user_input, chosen, baseline_label="(baseline)")

    # Grade
    grade_cols = [c for c in feature_columns if c.startswith("Grade_")]
    if grade_cols:
        grade_opts = [c.replace("Grade_", "") for c in grade_cols]
        grade_opts = ["(baseline)"] + grade_opts
        chosen = st.selectbox("Grade", grade_opts, index=0)

        for c in grade_cols:
            user_input.at[0, c] = 0

        if chosen != "(baseline)":
            target = f"Grade_{chosen}"
            if target in user_input.columns:
                user_input.at[0, target] = 1

    st.subheader("Hormone receptor status")

    if "Estrogen Status_Positive" in user_input.columns:
        set_binary(user_input, "Estrogen Status_Positive",
                   st.selectbox("Estrogen Status Positive?", [0, 1], index=0))

    if "Progesterone Status_Positive" in user_input.columns:
        set_binary(user_input, "Progesterone Status_Positive",
                   st.selectbox("Progesterone Status Positive?", [0, 1], index=0))

    submitted = st.form_submit_button("✅ Predict")

# -------------------------
# Predict
# -------------------------
if submitted:
    try:
        pred = model.predict(user_input)[0]
        st.success(f"Prediction: **{pred}**")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(user_input)[0]

            # get class labels
            if hasattr(model, "classes_"):
                classes = model.classes_
            else:
                classes = [f"class_{i}" for i in range(len(proba))]

            st.write("Prediction probabilities:")
            st.dataframe(pd.DataFrame([proba], columns=classes))

        with st.expander("Show full model input (debug)"):
            st.dataframe(user_input)

    except Exception as e:
        st.error("Prediction failed (likely feature mismatch).")
        st.exception(e)

st.caption("Model: Logistic Regression | Race removed + Node_Ratio engineered feature | single model file")
