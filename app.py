import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shap

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Turnover Prediction",
    page_icon="📉",
    layout="wide"
)

# ─────────────────────────────────────────────
# LOAD MODELS (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_all():
    def _load(path):
        try:
            return joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                return pickle.load(f)

    return (
        _load("xgb_attrition_model.pkl"),
        _load("scaler.pkl"),
        _load("encoder.pkl"),
        _load("feature_importances.pkl"),
        _load("final_columns.pkl"),
    )

model, scaler, encoder, feature_imp, final_columns = load_all()

# Salary encoding — gunakan mapping manual agar aman
SALARY_MAP = {"low": 0, "medium": 1, "high": 2}

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
try:
    st.image("turnover-adalah.jpg", use_container_width=True)
except Exception:
    pass  # skip jika gambar tidak ada

st.title("📉 Employee Turnover Prediction Dashboard")
st.markdown("Prediksi risiko resign karyawan menggunakan model **XGBoost** (ROC-AUC: **0.9853**)")
st.divider()

# ─────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────
st.subheader("📌 Input Data Karyawan")

col1, col2, col3 = st.columns(3)

with col1:
    satisfaction_level   = st.slider("Satisfaction Level", 0.0, 1.0, 0.50, step=0.01,
                                      help="Tingkat kepuasan kerja karyawan (0 = sangat tidak puas, 1 = sangat puas)")
    last_evaluation      = st.slider("Last Evaluation Score", 0.0, 1.0, 0.60, step=0.01,
                                      help="Skor evaluasi kinerja terakhir")

with col2:
    number_project       = st.number_input("Jumlah Proyek", 1, 10, 3,
                                            help="Jumlah proyek yang sedang ditangani")
    average_montly_hours = st.number_input("Rata-rata Jam Kerja/Bulan", 50, 350, 160,
                                            help="Rata-rata jam kerja per bulan")
    time_spend_company   = st.number_input("Lama Bekerja (Tahun)", 1, 20, 3,
                                            help="Masa kerja di perusahaan")

with col3:
    salary               = st.selectbox("Level Gaji", ["low", "medium", "high"])
    work_accident_label  = st.selectbox("Pernah Kecelakaan Kerja?", ["Tidak", "Ya"])
    promotion_label      = st.selectbox("Promosi dalam 5 Tahun Terakhir?", ["Tidak", "Ya"])

work_accident         = 1 if work_accident_label == "Ya" else 0
promotion_last_5years = 1 if promotion_label == "Ya" else 0
salary_encoded        = SALARY_MAP[salary]

# ─────────────────────────────────────────────
# BUILD INPUT DATAFRAME
# ─────────────────────────────────────────────
input_data = pd.DataFrame([[
    satisfaction_level,
    last_evaluation,
    number_project,
    average_montly_hours,
    time_spend_company,
    salary_encoded,
    work_accident,
    promotion_last_5years
]], columns=final_columns)

scaled_input = scaler.transform(input_data)

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
st.divider()
predict_btn = st.button("🔮 Prediksi Sekarang", use_container_width=True, type="primary")

if predict_btn:

    prediction  = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    # ── RESULT CARD ──────────────────────────
    st.subheader("🎯 Hasil Prediksi")

    if prediction == 1:
        st.markdown(f"""
        <div style="padding:24px;border-radius:12px;background:#fff0f0;border-left:8px solid #e74c3c;">
            <h2 style="color:#e74c3c;margin:0">⚠️ Risiko Tinggi — Karyawan Berpotensi RESIGN</h2>
            <h3 style="margin:8px 0 0 0">Probabilitas Resign: <b>{probability:.1%}</b></h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="padding:24px;border-radius:12px;background:#f0fff4;border-left:8px solid #2ecc71;">
            <h2 style="color:#27ae60;margin:0">✅ Risiko Rendah — Karyawan Cenderung BERTAHAN</h2>
            <h3 style="margin:8px 0 0 0">Probabilitas Resign: <b>{probability:.1%}</b></h3>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── LAYOUT: RADAR + GAUGE ─────────────────
    left_col, right_col = st.columns(2)

    # Radar Chart
    with left_col:
        st.subheader("🕸 Profil Risiko Karyawan")
        radar_labels = ["Satisfaction", "Evaluation", "Projects", "Hours", "Tenure"]
        radar_vals   = [
            satisfaction_level,
            last_evaluation,
            number_project / 10,
            average_montly_hours / 350,
            time_spend_company / 20,
        ]
        radar_closed = radar_vals + [radar_vals[0]]
        label_closed = radar_labels + [radar_labels[0]]

        fig_radar = go.Figure(go.Scatterpolar(
            r=radar_closed,
            theta=label_closed,
            fill="toself",
            line_color="#e74c3c" if prediction == 1 else "#2ecc71",
            fillcolor="rgba(231,76,60,0.2)" if prediction == 1 else "rgba(46,204,113,0.2)"
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Gauge Chart
    with right_col:
        st.subheader("🎚 Probabilitas Resign")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis"  : {"range": [0, 100]},
                "bar"   : {"color": "#e74c3c" if prediction == 1 else "#2ecc71"},
                "steps" : [
                    {"range": [0, 40],  "color": "#eafaf1"},
                    {"range": [40, 70], "color": "#fef9e7"},
                    {"range": [70, 100],"color": "#fdedec"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": 70}
            }
        ))
        fig_gauge.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()

    # ── FEATURE IMPORTANCE ────────────────────
    st.subheader("📊 Feature Importance Model")

    imp_series = pd.Series(feature_imp, index=final_columns).sort_values(ascending=True)
    colors     = ["#e74c3c" if v > imp_series.median() else "#3498db" for v in imp_series.values]

    fig_imp = go.Figure(go.Bar(
        x=imp_series.values,
        y=imp_series.index,
        orientation="h",
        marker_color=colors
    ))
    fig_imp.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="",
        height=320,
        margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()

    # ── SHAP WATERFALL ────────────────────────
    st.subheader("🔥 SHAP — Kontribusi Fitur terhadap Prediksi Ini")

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer(scaled_input)
        sample_sv   = shap_values[0]

        # Override feature names agar lebih readable
        sample_sv.feature_names = final_columns

        fig_shap, ax = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(sample_sv, show=False)
        plt.tight_layout()
        st.pyplot(fig_shap)
        plt.close(fig_shap)
    except Exception as e:
        st.warning(f"SHAP tidak dapat ditampilkan: {e}")

    st.divider()

    # ── REKOMENDASI ───────────────────────────
    if prediction == 1:
        st.subheader("💡 Rekomendasi Tindakan HR")
        st.markdown("""
        | Aspek | Rekomendasi |
        |---|---|
        | **Kepuasan Kerja** | Lakukan stay interview dan survei kepuasan segera |
        | **Beban Kerja** | Evaluasi jumlah proyek dan jam kerja — kurangi jika berlebihan |
        | **Karier** | Diskusikan career path dan peluang promosi yang jelas |
        | **Kompensasi** | Review gaji dan bandingkan dengan standar industri |
        | **Monitoring** | Pantau karyawan ini setiap bulan menggunakan dashboard HR |
        """)
