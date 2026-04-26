import streamlit as st
import pandas as pd
import joblib

# -------------------- Load Files --------------------
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')
categorical_info = joblib.load('categorical_info.pkl')

# -------------------- App Title --------------------
st.set_page_config(page_title="HR Attrition Predictor", layout="wide")

st.title("💼 HR Attrition Prediction System")
st.write("🔍 Enter employee details to check if they are likely to leave the company.")

st.write("---")

# -------------------- Input Section --------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Employee Details")
    age = st.slider('Age', 18, 60, 30)
    job_role = st.selectbox('Job Role', categorical_info['JobRole'])
    department = st.selectbox('Department', categorical_info['Department'])
    marital_status = st.selectbox('Marital Status', categorical_info['MaritalStatus'])

with col2:
    st.subheader("💼 Work Details")
    monthly_income = st.number_input('Monthly Income', 1000, 20000, 5000)
    over_time = st.selectbox('Over Time', ['Yes', 'No'])
    job_satisfaction = st.selectbox('Job Satisfaction', [1, 2, 3, 4])
    work_life_balance = st.selectbox('Work Life Balance', [1, 2, 3, 4])

st.write("---")

# -------------------- Prediction --------------------
if st.button("🚀 Predict Attrition"):

    # Create input data
    input_data = pd.DataFrame([{
        'Age': age,
        'JobRole': job_role,
        'Department': department,
        'MaritalStatus': marital_status,
        'MonthlyIncome': monthly_income,
        'OverTime': over_time,
        'JobSatisfaction': job_satisfaction,
        'WorkLifeBalance': work_life_balance
    }])

    # Convert categorical to numeric
    input_df = pd.get_dummies(input_data)

    # Add missing columns
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Arrange columns
    input_df = input_df[feature_names]

    # Scaling
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    # -------------------- Result Section --------------------
    st.write("## 📊 Prediction Result")

    if prediction[0] == 1:
        st.error("🚨 High Attrition Risk")
        st.write(f"👉 Probability of Leaving: **{probability[0][1]*100:.2f}%**")
    else:
        st.success("✅ Low Attrition Risk")
        st.write(f"👉 Probability of Staying: **{probability[0][0]*100:.2f}%**")

    # Progress bar
    st.write("### 📈 Risk Level")
    st.progress(int(probability[0][1]*100))