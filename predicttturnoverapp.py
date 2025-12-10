import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import plotly.express as px

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Employee Turnover Prediction", layout="wide")

# ---------------------------
# LOAD MODEL
# ---------------------------
model = pickle.load(open("turnover_model.pkl", "rb"))
feature_names = [
    "satisfaction_level", "last_evaluation", "number_project",
    "average_monthly_hours", "time_spend_company", "salary",
    "Work_accident", "promotion_last_5years"
]

# ---------------------------
# HEADER & IMAGE
# ---------------------------
st.title("📊 Employee Turnover Prediction App")

image = Image.open("turnover-adalah.jpg")
st.image(image, caption="Employee Turnover Illustration", use_column_width=True)

st.markdown("---")

# ---------------------------
# INPUT FORM
# ---------------------------
st.subheader("🔍 Input Employee Data")

col1, col2 = st.columns(2)

with col1:
    satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
    last_evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.6)
    number_project = st.number_input("Number of Projects", 1, 10, 3)
    salary = st.selectbox("Salary Level", ["low", "medium", "high"])

with col2:
    average_monthly_hours = st.number_input("Average Monthly Hours", 80, 350, 160)
    time_spend_company = st.number_input("Years at Company", 1, 15, 3)
    work_accident = st.selectbox("Work Accident (Yes/No)", ["No", "Yes"])
    promotion_last_5years = st.selectbox("Promotion in Last 5 Years (Yes/No)", ["No", "Yes"])

# Convert Yes/No → 1/0
work_accident = 1 if work_accident == "Yes" else 0
promotion_last_5years = 1 if promotion_last_5years == "Yes" else 0

# Salary encoding
salary_map = {"low": 0, "medium": 1, "high": 2}
salary = salary_map[salary]

# Prepare data
input_data = np.array([[satisfaction_level, last_evaluation, number_project,
                        average_monthly_hours, time_spend_company, salary,
                        work_accident, promotion_last_5years]])

st.markdown("---")

# ---------------------------
# PREDICTION
# ---------------------------
if st.button("🔮 Predict Turnover"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📌 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Employee is **likely to RESIGN**.\nProbability: **{probability:.2f}**")
    else:
        st.success(f"✅ Employee is predicted to **STAY**.\nProbability of turnover: **{probability:.2f}**")

    st.markdown("---")

    # ---------------------------
    # FEATURE IMPORTANCE
    # ---------------------------
    st.subheader("📊 Feature Importance")

    try:
        importance = model.feature_importances_
        df_importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        fig = px.bar(df_importance, x="Feature", y="Importance",
                     title="Feature Importance in Prediction")
        st.plotly_chart(fig, use_container_width=True)

        # ----------------------------------
        # EXPLANATION SECTION
        # ----------------------------------
        st.subheader("📝 Explanation of Results")

        explanation = """
### 🔍 What Influenced the Prediction?

Based on the model, the following features have the strongest impact on predicting employee turnover:

1. **Time Spent at Company**  
   Employees with longer years in the company show higher risk of turnover due to stagnation or burnout.

2. **Number of Projects**  
   Too many or too few projects can cause disengagement or overwork.

3. **Satisfaction Level**  
   One of the strongest indicators: lower satisfaction → higher turnover probability.

4. **Last Evaluation Score**  
   Extremely low or extremely high evaluations may correlate with turnover.

5. **Monthly Working Hours**  
   Very high working hours often indicate burnout risk.

6. **Work Accident**  
   Employees who experienced accidents usually have slight different retention patterns.

7. **Promotion in Last 5 Years**  
   Employees without promotion for long periods may feel stuck.

8. **Salary Level**  
   Lower salary generally increases turnover risk, but impact is weaker in this dataset.

---

### 🎯 Final Insight
The prediction is generated based on the combination of these features and how strongly each contributes to the model.  
Understanding these factors can help HR identify risk early and create better retention strategies.
"""
        st.markdown(explanation)

    except:
        st.warning("Model does not provide feature importance.")

