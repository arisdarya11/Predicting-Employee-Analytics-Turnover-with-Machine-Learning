import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
import os

# ===================== SAFE FILE LOADER =====================
def safe_load(path):
    """Load model safely from /mnt/data or local repo."""
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except:
            with open(path, "rb") as f:
                return pickle.load(f)
    else:
        st.error(f"❌ File not found: {path}")
        st.stop()

# ===================== LOAD MODELS =====================
model = safe_load("xgb_attrition_model.pkl")
scaler = safe_load("scaler.pkl")
encoder = safe_load("encoder.pkl")

# ===================== CONFIG =====================
st.set_page_config(
    page_title="Employee Turnover Prediction",
    page_icon="📉",
    layout="wide"
)

# ===================== HEADER IMAGE =====================
if os.path.exists("turnover-adalah.jpg"):
    st.image("turnover-adalah.jpg", use_column_width=True)
else:
    st.warning("⚠️ Gambar tidak ditemukan: turnover-adalah.jpg")

st.markdown("<h1 style='text-align:center;'>📉 Employee Turnover Prediction Dashboard</h1>", unsafe_allow_html=True)
st.write("---")

# ===================== INPUT FORM =====================
st.subheader("🧩 Employee Profile Input")

col1, col2 = st.columns(2)

with col1:
    satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
    last_evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.5)
    number_project = st.number_input("Number of Projects", 1, 10, 3)
    average_montly_hours = st.number_input("Average Monthly Hours", 50, 350, 160)

with col2:
    time_spend_company = st.number_input("Years at Company", 1, 20, 3)
    work_accident = st.selectbox("Work Accident?", ["No", "Yes"])
    promotion_last_5years = st.selectbox("Promotion in Last 5 Years?", ["No", "Yes"])
    salary = st.selectbox("Salary Level", ["low", "medium", "high"])

# Convert Yes/No → numeric
work_accident = 1 if work_accident == "Yes" else 0
promotion_last_5years = 1 if promotion_last_5years == "Yes" else 0

# Encode salary
salary_encoded = encoder.transform([salary])[0]

# ===================== BUILD INPUT DF =====================
model_columns = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "salary",
    "Work_accident",
    "promotion_last_5years"
]

input_df = pd.DataFrame([[
    satisfaction_level,
    last_evaluation,
    number_project,
    average_montly_hours,
    time_spend_company,
    salary_encoded,
    work_accident,
    promotion_last_5years
]], columns=model_columns)

scaled_input = scaler.transform(input_df)

# ===================== PREDICT BUTTON =====================
predict_btn = st.button("🔮 Predict Turnover Risk")

if predict_btn:
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    # =============== BEAUTIFUL RESULT CARD ===============
    st.subheader("🎯 Prediction Result")

    if prediction == 1:
        st.markdown("""
        <div style='background-color:#ffdddd;padding:20px;border-radius:10px;border-left:10px solid red'>
        <h2 style='color:red;'>⚠️ High Risk of Leaving</h2>
        <p>Employee memiliki kemungkinan tinggi untuk <b>resign</b>.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background-color:#ddffdd;padding:20px;border-radius:10px;border-left:10px solid green'>
        <h2 style='color:green;'>✅ Low Risk of Leaving</h2>
        <p>Employee cenderung <b>tetap bertahan</b> di perusahaan.</p>
        </div>
        """, unsafe_allow_html=True)

    st.metric("Probability of Attrition", f"{probability:.2f}")

    # =============== FEATURE IMPORTANCE ==================
    st.subheader("📊 Feature Importance")
    try:
        importances = model.feature_importances_
        fig_imp = go.Figure([go.Bar(
            x=model_columns,
            y=importances
        )])
        fig_imp.update_layout(title="Feature Importance (XGBoost)")
        st.plotly_chart(fig_imp, use_container_width=True)
    except:
        st.info("Feature importance tidak tersedia untuk model ini.")

    # =============== RADAR CHART ==================
    st.subheader("🕸 Risk Radar Chart")

    radar_labels = ["Satisfaction", "Evaluation", "Projects", "Monthly Hours", "Tenure"]
    radar_values = [
        satisfaction_level,
        last_evaluation,
        number_project / 10,
        average_montly_hours / 350,
        time_spend_company / 20
    ]

    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(
        r=radar_values,
        theta=radar_labels,
        fill="toself"
    ))
    radar.update_layout(polar=dict(radialaxis=dict(visible=True)))
    st.plotly_chart(radar, use_container_width=True)

    # =============== SHAP EXPLAINABILITY ==================
    st.subheader("🔥 SHAP Explanation")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(scaled_input)

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.plots.waterfall(shap_values[0], show=False_
