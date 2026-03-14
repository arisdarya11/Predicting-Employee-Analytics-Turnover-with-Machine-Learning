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
import io

# ════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════
st.set_page_config(
    page_title="HR Attrition Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ════════════════════════════════════════════
# GLOBAL CSS — Clean White Theme
# ════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.stApp { background: #ffffff; color: #000000; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { display: none; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 2px; }

.nav-bar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 0 20px 0; border-bottom: 2px solid #e2e8f0; margin-bottom: 32px;
}
.nav-logo {
    font-family: 'DM Mono', monospace; font-size: 13px; font-weight: 600;
    color: #000000; letter-spacing: 0.08em; text-transform: uppercase;
}
.nav-badge {
    font-family: 'DM Mono', monospace; font-size: 11px; color: #000000;
    background: #f8fafc; border: 1px solid #e2e8f0;
    padding: 4px 12px; border-radius: 20px;
}

.section-label {
    font-family: 'DM Mono', monospace; font-size: 10px;
    letter-spacing: 0.15em; text-transform: uppercase; color: #000000;
    margin-bottom: 16px; display: flex; align-items: center; gap: 8px;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: #e2e8f0; }

.kpi-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 24px; }
.kpi-card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 20px 24px; position: relative; overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
.kpi-card.danger::before  { background: linear-gradient(90deg, #f97316, #ef4444); }
.kpi-card.safe::before    { background: linear-gradient(90deg, #10b981, #06b6d4); }
.kpi-card.neutral::before { background: linear-gradient(90deg, #6366f1, #8b5cf6); }
.kpi-label {
    font-family: 'DM Mono', monospace; font-size: 10px;
    letter-spacing: 0.12em; text-transform: uppercase; color: #000000; margin-bottom: 8px;
}
.kpi-value { font-family: 'DM Mono', monospace; font-size: 32px; font-weight: 500; line-height: 1; margin-bottom: 4px; }
.kpi-value.danger  { color: #f97316; }
.kpi-value.safe    { color: #10b981; }
.kpi-value.neutral { color: #6366f1; }
.kpi-sub { font-size: 11px; color: #000000; }

.result-banner {
    border-radius: 12px; padding: 28px 32px; margin-bottom: 24px;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.result-banner.danger { background: #fff7ed; border: 1px solid #fed7aa; }
.result-banner.safe   { background: #f0fdf4; border: 1px solid #bbf7d0; }
.result-banner-icon { font-size: 48px; }
.result-banner-title { font-size: 20px; font-weight: 700; margin-bottom: 4px; }
.result-banner-title.danger { color: #ea580c; }
.result-banner-title.safe   { color: #16a34a; }
.result-banner-sub { font-size: 13px; color: #000000; }
.result-banner-prob { font-family: 'DM Mono', monospace; font-size: 48px; font-weight: 600; text-align: right; }
.result-banner-prob.danger { color: #f97316; }
.result-banner-prob.safe   { color: #10b981; }
.result-banner-prob-label {
    font-family: 'DM Mono', monospace; font-size: 10px;
    letter-spacing: 0.12em; text-transform: uppercase; color: #000000; text-align: right;
}

.reco-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 8px; }
.reco-item {
    background: #fff7ed; border: 1px solid #fed7aa;
    border-left: 3px solid #f97316; border-radius: 8px; padding: 14px 16px;
}
.reco-aspect {
    font-family: 'DM Mono', monospace; font-size: 10px;
    letter-spacing: 0.1em; text-transform: uppercase; color: #ea580c; margin-bottom: 4px;
}
.reco-text { font-size: 13px; color: #000000; line-height: 1.5; }

.info-box {
    background: #f0f9ff; border: 1px solid #bae6fd; border-left: 3px solid #0284c7;
    border-radius: 8px; padding: 16px 20px; margin-bottom: 16px;
    font-family: 'DM Mono', monospace; font-size: 12px; color: #000000; line-height: 1.8;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div > div {
    background: #f8fafc !important; border: 1px solid #e2e8f0 !important;
    color: #000000 !important; border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important; font-size: 13px !important;
}
label[data-testid="stWidgetLabel"] p {
    font-family: 'DM Mono', monospace !important; font-size: 11px !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important; color: #000000 !important;
}
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0f766e, #0d9488) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important; font-size: 13px !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
    padding: 14px 28px !important; box-shadow: 0 4px 12px rgba(15,118,110,0.25) !important;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #0d9488, #14b8a6) !important;
    box-shadow: 0 6px 16px rgba(15,118,110,0.35) !important;
}
hr { border-color: #e2e8f0 !important; }

div[data-testid="stTabs"] button {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; color: #000000 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #000000 !important; border-bottom-color: #000000 !important;
}
div[data-testid="stDataFrame"] { border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════
# LOAD MODELS
# ════════════════════════════════════════════
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
SALARY_MAP = {"low": 0, "medium": 1, "high": 2}

# ════════════════════════════════════════════
# NAV BAR
# ════════════════════════════════════════════
st.markdown("""
<div class="nav-bar">
    <div class="nav-logo">⬡ &nbsp;HR Attrition Intelligence</div>
    <div style="display:flex;gap:8px;">
        <span class="nav-badge">XGBoost · ROC-AUC 0.9853</span>
        <span class="nav-badge">n=14,999 records</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════
tab1, tab2 = st.tabs(["  🔍  Single Prediction  ", "  📂  Batch via File Upload  "])

# ════════════════════════════════════════════
# TAB 1 — SINGLE PREDICTION
# ════════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">01 &nbsp; Employee Profile Input</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.2, 1.2, 1], gap="medium")

    with col1:
        satisfaction_level   = st.slider("Satisfaction Level", 0.0, 1.0, 0.50, step=0.01)
        last_evaluation      = st.slider("Last Evaluation Score", 0.0, 1.0, 0.60, step=0.01)

    with col2:
        number_project       = st.number_input("Number of Projects", 1, 10, 3)
        average_montly_hours = st.number_input("Avg Monthly Hours", 50, 350, 160)
        time_spend_company   = st.number_input("Tenure (Years)", 1, 20, 3)

    with col3:
        salary               = st.selectbox("Salary Level", ["low", "medium", "high"])
        work_accident_label  = st.selectbox("Work Accident", ["No", "Yes"])
        promotion_label      = st.selectbox("Promoted (Last 5Y)", ["No", "Yes"])

    work_accident         = 1 if work_accident_label == "Yes" else 0
    promotion_last_5years = 1 if promotion_label == "Yes" else 0
    salary_encoded        = SALARY_MAP[salary]

    input_data = pd.DataFrame([[
        satisfaction_level, last_evaluation, number_project,
        average_montly_hours, time_spend_company, salary_encoded,
        work_accident, promotion_last_5years
    ]], columns=final_columns)
    scaled_input = scaler.transform(input_data)

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        predict_btn = st.button("⟶  Run Prediction", use_container_width=True, key="single_predict")

    if predict_btn:
        prediction  = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]
        prob_pct    = probability * 100
        is_danger   = prediction == 1
        tone        = "danger" if is_danger else "safe"

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">02 &nbsp; Prediction Result</div>', unsafe_allow_html=True)

        if is_danger:
            st.markdown(f"""
            <div class="result-banner danger">
                <div style="display:flex;align-items:center;gap:20px;">
                    <div class="result-banner-icon">⚠️</div>
                    <div>
                        <div class="result-banner-title danger">High Attrition Risk Detected</div>
                        <div class="result-banner-sub">This employee profile matches patterns associated with resignation</div>
                    </div>
                </div>
                <div>
                    <div class="result-banner-prob danger">{prob_pct:.1f}%</div>
                    <div class="result-banner-prob-label">Resign Probability</div>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-banner safe">
                <div style="display:flex;align-items:center;gap:20px;">
                    <div class="result-banner-icon">✅</div>
                    <div>
                        <div class="result-banner-title safe">Low Attrition Risk</div>
                        <div class="result-banner-sub">This employee profile indicates likely retention</div>
                    </div>
                </div>
                <div>
                    <div class="result-banner-prob safe">{prob_pct:.1f}%</div>
                    <div class="result-banner-prob-label">Resign Probability</div>
                </div>
            </div>""", unsafe_allow_html=True)

        risk_level = "CRITICAL" if prob_pct >= 70 else "MODERATE" if prob_pct >= 40 else "LOW"
        risk_tone  = "danger" if prob_pct >= 70 else "neutral" if prob_pct >= 40 else "safe"
        retention  = 100 - prob_pct

        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card {tone}">
                <div class="kpi-label">Resign Probability</div>
                <div class="kpi-value {tone}">{prob_pct:.1f}%</div>
                <div class="kpi-sub">Model confidence score</div>
            </div>
            <div class="kpi-card {risk_tone}">
                <div class="kpi-label">Risk Level</div>
                <div class="kpi-value {risk_tone}">{risk_level}</div>
                <div class="kpi-sub">Based on probability threshold</div>
            </div>
            <div class="kpi-card safe">
                <div class="kpi-label">Retention Probability</div>
                <div class="kpi-value safe">{retention:.1f}%</div>
                <div class="kpi-sub">Likelihood of staying</div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-label">03 &nbsp; Risk Analysis</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="medium")

        with c1:
            accent    = "#f97316" if is_danger else "#10b981"
            step_low  = "#dcfce7"
            step_mid  = "#fef9c3"
            step_high = "#fee2e2"
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=prob_pct,
                number={"suffix": "%", "font": {"size": 42, "color": accent, "family": "DM Mono"}},
                title={"text": "Attrition Probability", "font": {"size": 13, "color": "#000000", "family": "DM Mono"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#e2e8f0", "tickfont": {"color": "#000000", "size": 10}},
                    "bar":  {"color": accent, "thickness": 0.25},
                    "bgcolor": "#ffffff", "bordercolor": "#e2e8f0",
                    "steps": [
                        {"range": [0,  40], "color": step_low},
                        {"range": [40, 70], "color": step_mid},
                        {"range": [70,100], "color": step_high},
                    ],
                    "threshold": {"line": {"color": "#000000", "width": 2}, "thickness": 0.8, "value": 70}
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                height=280, margin=dict(t=40, b=10, l=20, r=20), font_color="#000000"
            )
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

        with c2:
            radar_labels = ["Satisfaction", "Evaluation", "Projects", "Hours", "Tenure"]
            radar_vals   = [satisfaction_level, last_evaluation, number_project/10, average_montly_hours/350, time_spend_company/20]
            radar_closed = radar_vals + [radar_vals[0]]
            label_closed = radar_labels + [radar_labels[0]]
            fill_color   = "rgba(249,115,22,0.12)" if is_danger else "rgba(16,185,129,0.12)"
            line_color   = "#f97316" if is_danger else "#10b981"

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_closed, theta=label_closed, fill="toself", fillcolor=fill_color,
                line=dict(color=line_color, width=2), marker=dict(size=5, color=line_color),
            ))
            fig_radar.update_layout(
                paper_bgcolor="#ffffff",
                polar=dict(
                    bgcolor="#f8fafc",
                    radialaxis=dict(visible=True, range=[0,1], tickfont=dict(color="#000000", size=9), gridcolor="#e2e8f0", linecolor="#e2e8f0"),
                    angularaxis=dict(tickfont=dict(color="#000000", size=11, family="DM Mono"), gridcolor="#e2e8f0", linecolor="#e2e8f0")
                ),
                showlegend=False,
                title=dict(text="Employee Risk Profile", font=dict(size=13, color="#000000", family="DM Mono")),
                height=280, margin=dict(t=50, b=10, l=40, r=40),
            )
            st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-label">04 &nbsp; Feature Importance</div>', unsafe_allow_html=True)
        imp_series = pd.Series(feature_imp, index=final_columns).sort_values(ascending=True)
        bar_colors = ["#f97316" if v >= imp_series.quantile(0.6) else "#6366f1" if v >= imp_series.quantile(0.3) else "#cbd5e1"
                      for v in imp_series.values]
        fig_imp = go.Figure(go.Bar(
            x=imp_series.values, y=imp_series.index, orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"{v:.3f}" for v in imp_series.values],
            textposition="outside",
            textfont=dict(family="DM Mono", size=10, color="#000000"),
        ))
        fig_imp.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(tickfont=dict(family="DM Mono", size=11, color="#000000"), gridcolor="#f1f5f9"),
            height=300, margin=dict(t=10, b=10, l=160, r=80), bargap=0.35,
        )
        st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-label">05 &nbsp; SHAP Explainability</div>', unsafe_allow_html=True)
        try:
            plt.style.use("default")
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer(scaled_input)
            sample_sv   = shap_values[0]
            sample_sv.feature_names = final_columns
            fig_shap, ax = plt.subplots(figsize=(10, 5))
            fig_shap.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#ffffff")
            shap.plots.waterfall(sample_sv, show=False)
            plt.tight_layout()
            st.pyplot(fig_shap, bbox_inches="tight")
            plt.close(fig_shap)
        except Exception as e:
            st.warning(f"SHAP tidak dapat ditampilkan: {e}")

        if is_danger:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">06 &nbsp; HR Action Recommendations</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="reco-grid">
                <div class="reco-item"><div class="reco-aspect">Satisfaction</div>
                    <div class="reco-text">Conduct a stay interview immediately. Run quarterly pulse surveys and act on key pain points.</div></div>
                <div class="reco-item"><div class="reco-aspect">Workload</div>
                    <div class="reco-text">Audit project assignments and monthly hours. Redistribute if above 200 hrs/month.</div></div>
                <div class="reco-item"><div class="reco-aspect">Career Path</div>
                    <div class="reco-text">Clarify promotion criteria. Schedule a career development discussion within 2 weeks.</div></div>
                <div class="reco-item"><div class="reco-aspect">Compensation</div>
                    <div class="reco-text">Benchmark against industry standards. Consider performance-based incentives.</div></div>
                <div class="reco-item"><div class="reco-aspect">Monitoring</div>
                    <div class="reco-text">Flag this employee for monthly HR check-ins and track satisfaction score trend.</div></div>
                <div class="reco-item"><div class="reco-aspect">Recognition</div>
                    <div class="reco-text">Implement an employee recognition program. Acknowledge contributions publicly.</div></div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 2 — BATCH FILE UPLOAD
# ════════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">01 &nbsp; Upload Survey File</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        <b style="color:#000000;">Kolom yang dibutuhkan dalam file:</b><br>
        nama_karyawan &nbsp;·&nbsp; {' &nbsp;·&nbsp; '.join(final_columns)}<br><br>
        <b style="color:#000000;">Salary:</b> low · medium · high &nbsp;&nbsp;
        <b style="color:#000000;">Binary (Work_accident, promotion_last_5years):</b> 0 atau 1
    </div>
    """, unsafe_allow_html=True)

    # ── Template CSV dengan nama karyawan ──
    import random, io as _io
    random.seed(42)
    first_names = ["Budi","Siti","Andi","Dewi","Reza","Fitri","Hendra","Lestari",
                   "Fajar","Nurul","Agus","Ratna","Dian","Yusuf","Mega"]
    last_names  = ["Santoso","Rahayu","Pratama","Kusuma","Hidayat","Sari","Wijaya",
                   "Nugroho","Permata","Utama","Cahyono","Lestari","Saputra","Wibowo","Suryadi"]
    salaries    = ["low","medium","high"]

    template_rows = []
    for i in range(5):
        template_rows.append({
            "nama_karyawan"        : f"{random.choice(first_names)} {random.choice(last_names)}",
            "satisfaction_level"   : round(random.uniform(0.2, 0.9), 2),
            "last_evaluation"      : round(random.uniform(0.4, 1.0), 2),
            "number_project"       : random.randint(2, 7),
            "average_montly_hours" : random.randint(100, 280),
            "time_spend_company"   : random.randint(2, 8),
            "salary"               : random.choices(salaries, weights=[0.5,0.35,0.15])[0],
            "Work_accident"        : random.choices([0,1], weights=[0.86,0.14])[0],
            "promotion_last_5years": random.choices([0,1], weights=[0.98,0.02])[0],
        })

    template_df  = pd.DataFrame(template_rows)
    csv_template = template_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇  Download Template CSV",
        data=csv_template,
        file_name="survey_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "Upload file CSV atau Excel",
        type=["csv", "xlsx", "xls"],
        help="Sertakan kolom nama_karyawan beserta 8 kolom fitur model"
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            st.stop()

        st.markdown(f"**{len(df_raw)} baris** terdeteksi dalam file.")

        # Ambil nama karyawan jika ada
        has_name = "nama_karyawan" in df_raw.columns
        names    = df_raw["nama_karyawan"].tolist() if has_name else [f"Karyawan {i+1}" for i in range(len(df_raw))]

        # Validasi kolom fitur
        missing_cols = [c for c in final_columns if c not in df_raw.columns]
        if missing_cols:
            st.error(f"Kolom berikut tidak ditemukan: `{', '.join(missing_cols)}`")
            st.stop()

        df_proc = df_raw[final_columns].copy()

        if df_proc["salary"].dtype == object:
            df_proc["salary"] = df_proc["salary"].str.lower().map(SALARY_MAP)

        if df_proc.isnull().any().any():
            st.warning("Ada nilai kosong atau salary tidak valid. Baris tersebut dilewati.")
            valid_idx = df_proc.dropna().index
            df_proc   = df_proc.loc[valid_idx]
            names     = [names[i] for i in valid_idx]

        scaled_batch = scaler.transform(df_proc)
        preds        = model.predict(scaled_batch)
        probas       = model.predict_proba(scaled_batch)[:, 1]

        # Susun hasil dengan nama
        df_result = pd.DataFrame({
            "Nama Karyawan"          : names,
            **{col: df_proc[col].values for col in final_columns},
            "Resign Probability (%)" : (probas * 100).round(1),
            "Prediction"             : ["⚠ RESIGN" if p == 1 else "✅ STAY" for p in preds],
            "Risk Level"             : ["CRITICAL" if p >= 70 else "MODERATE" if p >= 40 else "LOW" for p in (probas * 100)],
        })

        # ── Summary KPI ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">02 &nbsp; Batch Prediction Summary</div>', unsafe_allow_html=True)

        total     = len(df_result)
        high_risk = int((probas >= 0.7).sum())
        moderate  = int(((probas >= 0.4) & (probas < 0.7)).sum())
        low_risk  = int((probas < 0.4).sum())
        avg_prob  = probas.mean() * 100

        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card danger">
                <div class="kpi-label">High Risk (≥70%)</div>
                <div class="kpi-value danger">{high_risk}</div>
                <div class="kpi-sub">{high_risk/total*100:.1f}% dari total karyawan</div>
            </div>
            <div class="kpi-card neutral">
                <div class="kpi-label">Moderate Risk (40–70%)</div>
                <div class="kpi-value neutral">{moderate}</div>
                <div class="kpi-sub">{moderate/total*100:.1f}% dari total karyawan</div>
            </div>
            <div class="kpi-card safe">
                <div class="kpi-label">Low Risk (&lt;40%)</div>
                <div class="kpi-value safe">{low_risk}</div>
                <div class="kpi-sub">{low_risk/total*100:.1f}% dari total karyawan</div>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Charts ──
        st.markdown('<div class="section-label">03 &nbsp; Risk Distribution</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="medium")

        with c1:
            fig_pie = go.Figure(go.Pie(
                labels=["High Risk", "Moderate", "Low Risk"],
                values=[high_risk, moderate, low_risk],
                hole=0.6,
                marker=dict(colors=["#f97316", "#6366f1", "#10b981"], line=dict(color="#ffffff", width=2)),
                textfont=dict(family="DM Mono", size=11),
            ))
            fig_pie.update_layout(
                paper_bgcolor="#ffffff",
                showlegend=True,
                legend=dict(font=dict(family="DM Mono", size=11, color="#000000"), bgcolor="rgba(0,0,0,0)"),
                title=dict(text=f"Risk Distribution · {total} Karyawan", font=dict(size=13, color="#000000", family="DM Mono")),
                annotations=[dict(text=f"<b>{avg_prob:.1f}%</b><br>avg risk", x=0.5, y=0.5, font_size=14,
                                  font_color="#000000", font_family="DM Mono", showarrow=False)],
                height=300, margin=dict(t=50, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

        with c2:
            fig_hist = go.Figure(go.Histogram(
                x=probas * 100, nbinsx=20,
                marker=dict(color="#6366f1", line=dict(color="#ffffff", width=1)),
            ))
            fig_hist.add_vline(x=70, line=dict(color="#f97316", width=1.5, dash="dot"))
            fig_hist.add_vline(x=40, line=dict(color="#6366f1", width=1.5, dash="dot"))
            fig_hist.update_layout(
                paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                xaxis=dict(title="Resign Probability (%)", tickfont=dict(family="DM Mono", size=10, color="#000000"), gridcolor="#e2e8f0", color="#000000"),
                yaxis=dict(title="Jumlah Karyawan", tickfont=dict(family="DM Mono", size=10, color="#000000"), gridcolor="#e2e8f0", color="#000000"),
                title=dict(text="Distribusi Probabilitas", font=dict(size=13, color="#000000", family="DM Mono")),
                height=300, margin=dict(t=50, b=10, l=10, r=10), bargap=0.05,
            )
            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

        # ── Tabel Hasil ──
        st.markdown('<div class="section-label">04 &nbsp; Tabel Hasil Prediksi</div>', unsafe_allow_html=True)
        st.dataframe(
            df_result.sort_values("Resign Probability (%)", ascending=False),
            use_container_width=True,
            height=420,
        )

        # ── Download ──
        csv_out = df_result.sort_values("Resign Probability (%)", ascending=False).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇  Download Hasil Prediksi (CSV)",
            data=csv_out,
            file_name="attrition_prediction_results.csv",
            mime="text/csv",
        )

# ════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:20px 0;border-top:2px solid #e2e8f0;">
    <span style="font-family:'DM Mono',monospace;font-size:11px;color:#000000;letter-spacing:0.1em;">
        HR ATTRITION INTELLIGENCE · XGBOOST · ROC-AUC 0.9853 · BUILT WITH STREAMLIT
    </span>
</div>
""", unsafe_allow_html=True)
