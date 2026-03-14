import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import shap
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")  # Fix: non-interactive backend untuk server deploy
import matplotlib.pyplot as plt

# ======================== PAGE CONFIG ========================
st.set_page_config(
    page_title="Employee Turnover Prediction",
    page_icon="📉",
    layout="wide"
)

# ======================== LOAD FILES ========================
@st.cache_resource  # Fix: cache supaya tidak reload setiap interaksi
def load_all_models():
    def load_file(filename):
        try:
            return joblib.load(filename)
        except Exception:
            with open(filename, "rb") as f:
                return pickle.load(f)

    model             = load_file("xgb_attrition_model.pkl")
    scaler            = load_file("scaler.pkl")
    encoder           = load_file("encoder.pkl")
    feature_imp       = load_file("feature_importances.pkl")
    final_columns     = load_file("final_columns.pkl")
    return model, scaler, encoder, feature_imp, final_columns

model, scaler, encoder, feature_imp, final_columns = load_all_models()

# ======================== HEADER ========================
st.image("turnover-adalah.jpg", use_container_width=True)  # Fix: deprecated use_column_width
st.title("📉 Employee Turnover Prediction Dashboard")

# ======================== INPUT SECTION ========================
st.subheader("📌 Employee Data Input")

col1, col2 = st.columns(2)

with col1:
    satisfaction_level    = st.slider("Satisfaction Level", 0.0, 1.0, 0.50, step=0.01)
    last_evaluation       = st.slider("Last Evaluation Score", 0.0, 1.0, 0.60, step=0.01)
    number_project        = st.number_input("Number of Projects", 1, 10, 3)

with col2:
    average_montly_hours  = st.number_input("Average Monthly Hours", 50, 350, 160)
    time_spend_company    = st.number_input("Years at Company", 1, 20, 3)

work_accident_label       = st.selectbox("Work Accident?", ["No", "Yes"])
promotion_label           = st.selectbox("Promotion in Last 5 Years?", ["No", "Yes"])

work_accident             = 1 if work_accident_label == "Yes" else 0
promotion_last_5years     = 1 if promotion_label == "Yes" else 0

salary                    = st.selectbox("Salary Level", ["low", "medium", "high"])

# Fix: handle encoder transform dengan aman
try:
    salary_encoded = encoder.transform([salary])[0]
except Exception:
    salary_map     = {"low": 0, "medium": 1, "high": 2}
    salary_encoded = salary_map[salary]

# ======================== BUILD INPUT ========================
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

input_data = pd.DataFrame([[
    satisfaction_level,
    last_evaluation,
    number_project,
    average_montly_hours,
    time_spend_company,
    salary_encoded,
    work_accident,
    promotion_last_5years
]], columns=model_columns)

# Fix: pastikan kolom sesuai final_columns dari training
try:
    input_data = input_data[final_columns]
except Exception:
    pass  # fallback ke urutan default jika final_columns tidak cocok

scaled_input = scaler.transform(input_data)

# ======================== PREDICT BUTTON ========================
if st.button("🔮 Predict Turnover"):

    prediction  = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    # ======================== RESULT CARD ========================
    st.subheader("🎯 Prediction Result")

    if prediction == 1:
        st.markdown(f"""
        <div style="
            padding: 20px;
            border-radius: 12px;
            background-color: #ffdddd;
            border-left: 8px solid #ff4d4d;">
            <h2>⚠️ High Risk of Attrition</h2>
            <h3>Employee likely to <b>LEAVE</b></h3>
            <h3>Probability: <b>{probability:.2%}</b></h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="
            padding: 20px;
            border-radius: 12px;
            background-color: #ddffdd;
            border-left: 8px solid #37c837;">
            <h2>✅ Low Risk of Attrition</h2>
            <h3>Employee likely to <b>STAY</b></h3>
            <h3>Probability: <b>{probability:.2%}</b></h3>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ======================== RADAR CHART ========================
    st.subheader("🕸 Risk Radar Chart")

    radar_features = ["Satisfaction", "Evaluation", "Projects", "Monthly Hours", "Tenure"]
    radar_values   = [
        satisfaction_level,
        last_evaluation,
        number_project / 10,
        average_montly_hours / 350,
        time_spend_company / 20
    ]
    radar_values_closed = radar_values + [radar_values[0]]  # tutup polygon
    radar_features_closed = radar_features + [radar_features[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_values_closed,
        theta=radar_features_closed,
        fill="toself",
        line_color="#ff4d4d" if prediction == 1 else "#37c837"
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # ======================== FEATURE IMPORTANCE ========================
    st.subheader("📊 Feature Importance")

    try:
        # Gunakan feature_importances.pkl yang sudah ada di repo
        imp_series = pd.Series(feature_imp, index=model_columns if not isinstance(feature_imp, dict) else list(feature_imp.keys()))
        imp_series = imp_series.sort_values(ascending=True)

        fig_imp = go.Figure([go.Bar(
            x=imp_series.values,
            y=imp_series.index,
            orientation="h",
            marker_color="#636EFA"
        )])
        fig_imp.update_layout(
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=350
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception as e:
        # Fallback ke model.feature_importances_
        try:
            importances = model.feature_importances_
            fig_imp = go.Figure([go.Bar(x=model_columns, y=importances, marker_color="#636EFA")])
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception:
            st.info(f"Feature importance tidak tersedia: {e}")

    st.divider()

    # ======================== SHAP EXPLAINABILITY ========================
    st.subheader("🔥 SHAP Explainability")

    try:
        explainer  = shap.TreeExplainer(model)
        shap_values = explainer(scaled_input)
        sample_sv  = shap_values[0]

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(sample_sv, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)  # Fix: tutup figure supaya tidak memory leak
    except Exception as e:
        st.warning(f"SHAP tidak dapat ditampilkan: {e}")
