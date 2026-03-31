import streamlit as st
import pandas as pd
import os
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score

# =========================
# 🎨 ADVANCED UI STYLE
# =========================
st.set_page_config(page_title="AI Analytics System", layout="wide")

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: #00ffe7;
}
.stButton>button {
    background-color: #00ffe7;
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-weight: bold;
}
.css-1d391kg {
    background-color: #1c1f26;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
# 🎓 AI Smart Data Analytics & Prediction System
### 📊 Predict • Analyze • Explain (AI Powered)
---
""")

# =========================
# 📁 DATA LOADING
# =========================
uploaded_file = st.file_uploader("📁 Upload Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel("student_data.csv.xlsx")

st.subheader("📊 Dataset Preview")
st.dataframe(df, use_container_width=True)

# =========================
# 🤖 MODEL LOAD
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "lr.pkl")
lr = joblib.load(model_path)

x = df[['Study_Hours', 'Attendance (%)', 'Sleep_Hours',
        'Previous_Marks', 'Internet_Usage (hrs)',
        'Assignments_Completed']]

y_actual = df['Final_Marks']
y_pred = lr.predict(x)
accuracy = r2_score(y_actual, y_pred)

st.sidebar.header("📈 Model Performance")
st.sidebar.metric("R² Score", f"{accuracy:.2f}")

# SHAP
explainer = shap.LinearExplainer(lr, np.array(x))

# =========================
# 🔀 MODE SELECTION
# =========================
mode = st.radio("Select Mode", ["✍️ Manual Input", "🎯 Select Student ID"])

# =========================
# ✍️ MANUAL INPUT MODE
# =========================
if mode == "✍️ Manual Input":

    st.subheader("Enter Student Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        study_hours = st.slider("Study Hours", 0, 15, 5)
        attendance = st.slider("Attendance (%)", 0, 100, 75)

    with col2:
        sleep_hours = st.slider("Sleep Hours", 0, 12, 7)
        previous_marks = st.slider("Previous Marks", 0, 100, 60)

    with col3:
        internet_usage = st.slider("Internet Usage", 0, 10, 3)
        assignments = st.slider("Assignments Completed", 0, 10, 5)

    if st.button("🚀 Predict Now"):

        input_data = [[study_hours, attendance, sleep_hours,
                       previous_marks, internet_usage, assignments]]

        prediction = lr.predict(input_data)
        score = prediction[0]

        st.success(f"🎯 Predicted Marks: {score:.2f}")

# =========================
# 🎯 STUDENT ID MODE
# =========================
else:

    if 'Student_ID' in df.columns:

        student_id = st.selectbox("Select Student ID", df['Student_ID'].unique())
        student = df[df['Student_ID'] == student_id].iloc[0]

        st.subheader("📋 Student Details")
        st.json(student.to_dict())

        if st.button("🔍 Analyze Student"):

            input_data = [[
                student['Study_Hours'],
                student['Attendance (%)'],
                student['Sleep_Hours'],
                student['Previous_Marks'],
                student['Internet_Usage (hrs)'],
                student['Assignments_Completed']
            ]]

            prediction = lr.predict(input_data)
            score = prediction[0]

            st.success(f"🎯 Predicted Marks: {score:.2f}")

    else:
        st.error("❌ Student_ID column not found")

# =========================
# 🎯 COMMON ANALYSIS (USED BY BOTH)
# =========================
if 'score' in locals():

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Performance")
        if score >= 80:
            st.success("🟢 Excellent")
        elif score >= 50:
            st.warning("🟡 Average")
        else:
            st.error("🔴 Poor")

    with col2:
        st.subheader("⚠️ Risk Level")
        if score < 40:
            st.error("🔴 High Risk")
        elif score < 70:
            st.warning("🟡 Medium Risk")
        else:
            st.success("🟢 Low Risk")

    # SHAP
    st.subheader("🧠 AI Explanation (SHAP)")
    input_df = pd.DataFrame(input_data, columns=x.columns)
    shap_values = explainer(input_df)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    # Feedback
    st.subheader("🎯 Smart Feedback")

    feedback = []
    if input_data[0][0] < 3:
        feedback.append("📉 Increase Study Hours")
    if input_data[0][1] < 60:
        feedback.append("📉 Improve Attendance")
    if input_data[0][4] > 6:
        feedback.append("📉 Reduce Internet Usage")

    for f in feedback:
        st.warning(f)

    # Final Recommendation
    st.subheader("🧠 Final Recommendation")

    if score < 50:
        st.info("🔴 Needs serious improvement")
    elif score < 75:
        st.info("🟡 Can improve with consistency")
    else:
        st.info("🟢 Excellent performance")

# =========================
# 📊 DASHBOARD
# =========================
st.subheader("📊 Data Dashboard")

st.write("### Statistics")
st.write(df.describe())

st.write("### Study vs Marks")
st.bar_chart(df[['Study_Hours', 'Previous_Marks']])

st.write("### Correlation Heatmap")
fig2, ax2 = plt.subplots()
sns.heatmap(df.drop(columns=['Student_ID'], errors='ignore').corr(),
            annot=True, cmap='coolwarm', ax=ax2)
st.pyplot(fig2)

st.write("### Feature Importance")
importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': lr.coef_
})
st.bar_chart(importance.set_index('Feature'))