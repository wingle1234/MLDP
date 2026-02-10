import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Breast Cancer Prediction App")
st.write(
    "This tool estimates outcome risk based on clinical information. "
    "If you don’t know a field, you can leave it as default."
)

st.info("⚠️ Educational use only — not a medical diagnosis.")

with st.expander("How to use this (for normal users)"):
    st.markdown(
        """
- **Age**: enter in years (e.g., 55)  
- **Tumor size**: enter in **mm** (example: 2.5 cm = 25 mm)  
- **Lymph nodes**: enter **examined** and **positive** counts (the app computes a positivity ratio internally)  
- **Advanced fields** (stage/grade) are optional — only fill if you have the doctor’s report  
"""
    )

# -------------------------
# Load ONE model (feature names embedded)
# -------------------------
MODEL_PATH = "Models/breast_cancer_lr_v1.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Ensure the model file exists in Models/ and is committed to GitHub.")
    st.stop()

model = load_model()

if not hasattr(model, "feature_names_"):
    st.error(
        "Model missing feature_names_. Re-export the model and attach X.columns as model.feature_names_."
    )
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

def build_options(prefix, baseline_label="Not sure / leave default"):
    """
    You used drop_first=True, so one category is missing.
    Baseline label represents "all zeros" for that one-hot group.
    """
    cols = one_hot_cols(prefix)
    opts = [c.replace(prefix + "_", "") for c in cols]
    return cols, [baseline_label] + opts

def set_one_hot(df, prefix, chosen_value, baseline_label="Not sure / leave default"):
    cols = one_hot_cols(prefix)
    for c in cols:
        df.at[0, c] = 0
    if chosen_value == baseline_label:
        return
    target = f"{prefix}_{chosen_value}"
    if target in df.columns:
        df.at[0, target] = 1

def diff_cols():
    return [c for c in feature_columns if c.startswith("differentiate_")]

def build_diff_options(baseline_label="Not sure / leave default"):
    cols = diff_cols()
    opts = [c.replace("differentiate_", "") for c in cols]
    return cols, [baseline_label] + opts

def set_diff(df, chosen_value, baseline_label="Not sure / leave default"):
    cols = diff_cols()
    for c in cols:
        df.at[0, c] = 0
    if chosen_value == baseline_label:
        return
    target = f"differentiate_{chosen_value}"
    if target in df.columns:
        df.at[0, target] = 1

def yn_select(label, help_text=""):
    choice = st.radio(label, ["Not sure", "Yes", "No"], horizontal=True, help=help_text)
    if choice == "Yes":
        return 1
    if choice == "No":
        return 0
    return None

# -------------------------
# Session defaults (friend-proof)
# -------------------------
if "age_val" not in st.session_state:
    st.session_state["age_val"] = 50
if "tumor_mm" not in st.session_state:
    st.session_state["tumor_mm"] = 20
if "nodes_examined" not in st.session_state:
    st.session_state["nodes_examined"] = 0
if "nodes_positive" not in st.session_state:
    st.session_state["nodes_positive"] = 0

# Optional: store advanced default selections
for key in ["marital", "tstage", "nstage", "stage6", "diff", "grade"]:
    if key not in st.session_state:
        st.session_state[key] = "Not sure / leave default"

# -------------------------
# Demo button (helps lecturers / friends)
# -------------------------
colA, colB = st.columns([1, 2])
with colA:
    if st.button("✨ Fill sample values"):
        st.session_state["age_val"] = 55
        st.session_state["tumor_mm"] = 25
        st.session_state["nodes_examined"] = 12
        st.session_state["nodes_positive"] = 3
        st.rerun()
with colB:
    st.caption("Use this to quickly test the app without knowing the medical details.")

# -------------------------
# Build input row
# -------------------------
user_input = make_empty_input(feature_columns)

# -------------------------
# UI (friendly)
# -------------------------
with st.form("predict_form"):

    st.header("🧍 Personal Information")

    # Age
    if "Age" in user_input.columns:
        age = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=120,
            value=int(st.session_state["age_val"]),
            step=1,
            help="Example: 55"
        )
        user_input.at[0, "Age"] = float(age)

    st.header("📏 Tumor & Lymph Node Details")

    # Tumor Size
    if "Tumor Size" in user_input.columns:
        tumor_mm = st.number_input(
            "Tumor size (mm)",
            min_value=0,
            max_value=300,
            value=int(st.session_state["tumor_mm"]),
            step=1,
            help="Enter in millimeters. Example: 2.5 cm = 25 mm"
        )
        user_input.at[0, "Tumor Size"] = float(tumor_mm)

    # Node ratio (computed)
    if "Node_Ratio" in user_input.columns:
        nodes_examined = st.number_input(
            "Lymph nodes examined",
            min_value=0,
            max_value=200,
            value=int(st.session_state["nodes_examined"]),
            step=1,
            help="Total number of lymph nodes examined."
        )
        nodes_positive = st.number_input(
            "Lymph nodes positive",
            min_value=0,
            max_value=200,
            value=int(st.session_state["nodes_positive"]),
            step=1,
            help="Number of examined nodes that were cancer-positive."
        )

        if nodes_positive > nodes_examined:
            st.error("Positive nodes cannot be more than nodes examined.")
            st.stop()

        node_ratio = (nodes_positive / nodes_examined) if nodes_examined > 0 else 0.0
        user_input.at[0, "Node_Ratio"] = float(node_ratio)
        st.caption(f"Computed positivity ratio = {node_ratio:.3f}")

    st.header("🧬 Hormone Receptor Status")

    # Hormone fields: show Yes/No/Not sure, but model expects 0/1.
    # If user picks Not sure, we keep the default (0) to match baseline behavior.
    if "Estrogen Status_Positive" in user_input.columns:
        er = yn_select(
            "Estrogen receptor positive?",
            help_text="If not sure, leave as Not sure."
        )
        if er is not None:
            set_binary(user_input, "Estrogen Status_Positive", er)

    if "Progesterone Status_Positive" in user_input.columns:
        pr = yn_select(
            "Progesterone receptor positive?",
            help_text="If not sure, leave as Not sure."
        )
        if pr is not None:
            set_binary(user_input, "Progesterone Status_Positive", pr)

    # Advanced fields
    with st.expander("Advanced clinical details (optional — only if you have the report)"):
        st.subheader("Cancer staging / grading (optional)")

        # Marital Status (usually not critical; keep optional)
        m_cols, m_opts = build_options("Marital Status")
        if m_cols:
            chosen = st.selectbox(
                "Marital status",
                m_opts,
                index=0,
                help="If unsure, leave default."
            )
            set_one_hot(user_input, "Marital Status", chosen)

        # T Stage
        t_cols, t_opts = build_options("T Stage")
        if t_cols:
            chosen = st.selectbox(
                "Tumor stage (T stage)",
                t_opts,
                index=0,
                help="Found in the clinical report (e.g., T1, T2...). If unsure, leave default."
            )
            set_one_hot(user_input, "T Stage", chosen)

        # N Stage
        n_cols, n_opts = build_options("N Stage")
        if n_cols:
            chosen = st.selectbox(
                "Node stage (N stage)",
                n_opts,
                index=0,
                help="Found in the clinical report (e.g., N0, N1...). If unsure, leave default."
            )
            set_one_hot(user_input, "N Stage", chosen)

        # 6th Stage
        s_cols, s_opts = build_options("6th Stage")
        if s_cols:
            chosen = st.selectbox(
                "Overall stage (6th stage)",
                s_opts,
                index=0,
                help="Found in clinical report (e.g., I, II, III). If unsure, leave default."
            )
            set_one_hot(user_input, "6th Stage", chosen)

        # Differentiation
        d_cols, d_opts = build_diff_options()
        if d_cols:
            chosen = st.selectbox(
                "Differentiation",
                d_opts,
                index=0,
                help="If unsure, leave default."
            )
            set_diff(user_input, chosen)

        # Grade
        grade_cols = [c for c in feature_columns if c.startswith("Grade_")]
        if grade_cols:
            grade_opts = ["Not sure / leave default"] + [c.replace("Grade_", "") for c in grade_cols]
            chosen = st.selectbox(
                "Grade",
                grade_opts,
                index=0,
                help="If unsure, leave default."
            )

            # clear group
            for c in grade_cols:
                user_input.at[0, c] = 0

            # set one if chosen
            if chosen != "Not sure / leave default":
                target = f"Grade_{chosen}"
                if target in user_input.columns:
                    user_input.at[0, target] = 1

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
            classes = model.classes_ if hasattr(model, "classes_") else [f"class_{i}" for i in range(len(proba))]
            st.write("Prediction probabilities:")
            st.dataframe(pd.DataFrame([proba], columns=classes))

        with st.expander("Show full model input (debug)"):
            st.dataframe(user_input)

    except Exception as e:
        st.error("Prediction failed (likely feature mismatch).")
        st.exception(e)

st.caption("Model: Logistic Regression | Race removed + Node_Ratio engineered feature | single model file")
