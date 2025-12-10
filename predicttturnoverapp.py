import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
import base64

# =========================================================
# LOAD FILES
# =========================================================
def load_file(filename):
    try:
        return joblib.load(filename)
    except:
        with open(filename, "rb") as f:
            return pickle.load(f)

model = load_file("xgb_attrition_model.pkl")
scaler = load_file("scaler.pkl")
encoder = load_file("encoder.pkl")

# Salary mapping
salary_map = {"low": 0, "medium": 1, "high": 2}

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Employee Turnover Prediction",
    page_icon="📉",
    layout="wide"
)

# =========================================================
# HEADER IMAGE
# =========================================================
img_path = "/mnt/data/turnover-adalah.jpg"

if img_path:
    img_b64 = base64.b64encode(open(img_path, "rb").read()).decode()
    st.markdown(
        f"""
        <div style='display:flex; justify-content:center; margin-bottom:20px;'>
            <img src='data:image/jpg;base64,{img_b64}' 
                 style='width:80%; border-radius:14px; box-shadow:0 4px 18px rgba(0,0,0,0.35);'>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("<h1 style='text-align:center;'>📉 Employee Turnover Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# =========================================================
# INPUT FORM
# =========================================================

col1, col2 = st.columns(2)

with col1:
    satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.50)
    last_evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.50)
    number_project = st.number_input("Number of Projects", 1, 10, 3)
    average_montly_hours = st.number_input("Average Monthly Hours", 50, 350, 160)

with col2:
    time_spend_company = st.number_input("Years at Company", 1, 20, 3)
    work_accident = st.selectbox("Work Accident?", ["No", "Yes"])
    promotion_last_5years = st.selectbox("Promotion Last 5 Years?", ["No", "Yes"])
    salary = st.selectbox("Salary Level", ["low", "medium", "high"])

work_accident = 1 if work_accident == "Yes" else 0
promotion_last_5years = 1 if promotion_last_5years == "Yes" else 0

# =========================================================
# BUILD INPUT DATA
# =========================================================

input_data = pd.DataFrame({
    "satisfaction_level": [satisfaction_level],
    "last_evaluation": [last_evaluation],
    "number_project": [number_project],
    "average_montly_hours": [average_montly_hours],
    "time_spend_company": [time_spend_company],
    "Work_accident": [work_accident],
    "promotion_last_5years": [promotion_last_5years],
    "salary": [salary_map[salary]]
})

scaled_input = scaler.transform(input_data)

# =========================================================
# PREDICTION
# =========================================================
st.subheader("🔍 Prediction Result")

if st.button("Predict Turnover"):

    prediction = model.predict(scaled_input)[0]
    pred_proba = model.predict_proba(scaled_input)[0][1]

    # ----------- Glassmorphism Card ----------
    st.markdown("""
        <style>
            .glass-card {
                margin-top: 20px;
                padding: 25px;
                border-radius: 18px;
                background: rgba(255,255,255,0.05);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.15);
                text-align: center;
                box-shadow: 0 4px 22px rgba(0,0,0,0.25);
            }
        </style>
    """, unsafe_allow_html=True)

    label = (
        "❌ <b style='color:#ff6b6b;'>High Risk — Employee Likely to Leave</b>"
        if prediction == 1 else
        "✅ <b style='color:#6bff90;'>Low Risk — Employee Likely to Stay</b>"
    )

    st.markdown(
        f"""
        <div class='glass-card'>
            <h2>{label}</h2>
            <h3>Probability Score: <b>{pred_proba:.2f}</b></h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================================================
    # VISUALIZATIONS (Gauge + Insight)
    # =========================================================

    col_gauge, col_info = st.columns([1.3, 1])

    with col_gauge:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(pred_proba * 100, 2),
            title={'text': "Turnover Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#e74c3c" if pred_proba > 0.5 else "#2ecc71"},
                'steps': [
                    {'range': [0, 50], 'color': "#27ae60"},
                    {'range': [50, 100], 'color': "#c0392b"},
                ],
            }
        ))
        st.plotly_chart(gauge, use_container_width=True)

    with col_info:
        if prediction == 1:
            st.error("""
            ### 🔥 High Risk Insight
            - Risiko turnover **tinggi**
            - Cek faktor: beban kerja, lembur, kepuasan kerja
            - Rekomendasi: coaching & engagement program
            """)
        else:
            st.success("""
            ### 🌱 Stability Insight
            - Karyawan cenderung stabil
            - Risiko turnover rendah
            - Pertahankan kebijakan yang mendukung employee wellbeing
            """)

    # =========================================================
    # RADAR CHART
    # =========================================================
    st.subheader("🕸 Risk Radar Chart")

    radar_features = ["Satisfaction", "Evaluation", "Projects", "Monthly Hours", "Tenure"]
    radar_values = [
        satisfaction_level,
        last_evaluation,
        number_project / 10,
        average_montly_hours / 350,
        time_spend_company / 20
    ]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_values,
        theta=radar_features,
        fill="toself"
    ))

    st.plotly_chart(fig_radar, use_container_width=True)

    # =========================================================
    # SHAP PLOT
    # =========================================================
    st.subheader("🔥 SHAP Explainability")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(scaled_input)

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
