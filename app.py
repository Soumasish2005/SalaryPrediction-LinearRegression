import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load Model + Encoders
# -------------------------------

model = joblib.load("./saved_models/best_salary_model.joblib")
label_encoders = joblib.load("./saved_encoders/label_encoders.joblib")

# Load dataset to extract categories
df = pd.read_csv("Salary Data.csv")

# Clean NaN values
df["Gender"] = df["Gender"].fillna("Prefer not to say")
df["Job Title"] = df["Job Title"].fillna("Others")
df["Education Level"] = df["Education Level"].fillna("Others")

# Unique categories for dropdowns
genders = sorted(df["Gender"].unique())
education_levels = sorted(df["Education Level"].unique())
job_titles = sorted(df["Job Title"].unique())



st.set_page_config(page_title="Salary Prediction App", layout="centered")
st.title("💼 Salary Prediction App")
st.write("Fill the details below to estimate the employee salary.")


age = st.number_input("Age", min_value=18, max_value=70, value=30)

gender = st.selectbox("Gender", genders)

education = st.selectbox("Education Level", education_levels)

job_title = st.selectbox("Job Title", job_titles)

experience = st.slider("Years of Experience", 0, 40, 3)



input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Education Level": [education],
    "Job Title": [job_title],
    "Years of Experience": [experience]
})

# Encoding categorical columns
for col in label_encoders:
    try:
        input_data[col] = label_encoders[col].transform(input_data[col])
    except:
        st.error(f"Unknown category for {col}. Please update label encoders.")
        st.stop()

# Predicting the salary

if st.button("Predict Salary"):
    prediction = model.predict(input_data)[0]
    st.subheader("Predicted Salary: ")
    st.success(f"Estimated Salary: **${prediction:,.2f}**")

    st.write("---")
    st.caption("Prediction generated using an ensemble regression model.")
