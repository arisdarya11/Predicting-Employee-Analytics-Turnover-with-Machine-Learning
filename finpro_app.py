import streamlit as st
import pandas as pd
import pickle
import joblib

# =============== LOAD =================

def load_file(filename):
    try:
        return joblib.load(filename)
    except:
        with open(filename, "rb") as f:
            return pickle.load(f)

model = load_file("xgb_attrition_model.pkl")
scaler = load_file("scaler.pkl")
encoder = load_file("encoder.pkl")

# ===== FIXED COLUMN ORDER (SESUAI SAAT TRAINING) =====
model_columns = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",  # PENTING: jangan ubah spelling
    "time_spend_company",
    "salary",
    "Work_accident",
    "promotion_last_5years"
]

# =============== UI =================

st.title("Employee Turnover Prediction")

satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
last_evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.5)
number_project = st.number_input("Number of Projects", 1, 10, 3)
average_montly_hours = st.number_input("Average Monthly Hours", 50, 350, 160)
time_spend_company = st.number_input("Years at Company", 1, 20, 3)
work_accident = st.selectbox("Work Accident", [0, 1])
promotion_last_5years = st.selectbox("Promotion in Last 5 Years", [0, 1])
salary = st.selectbox("Salary Level", ["low", "medium", "high"])

# =============== BUILD INPUT (NO DEPARTMENT) =================

input_data = pd.DataFrame([[ 
    satisfaction_level,
    last_evaluation,
    number_project,
    average_montly_hours,
    time_spend_company,
    encoder.transform([salary])[0],
    work_accident,
    promotion_last_5years
]], columns=model_columns)

# =============== SCALING =================

scaled_input = scaler.transform(input_data)

# =============== PREDICT =================

if st.button("Predict Turnover"):
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Employee likely to leave (Prob: {probability:.2f})")
    else:
        st.success(f"✅ Employee likely to stay (Prob: {probability:.2f})")
