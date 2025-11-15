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
st.write("Fill the details below to estimate your salary.")

# Main two-column layout: inputs on the left, prediction on the right
main_left, main_right = st.columns([3, 3])

with main_left:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=70, value=25)
    with col2:
        gender = st.selectbox("Gender", genders)

    education = st.selectbox("Education Level", education_levels)

    job_title = st.selectbox("Job Title", job_titles)

    experience = st.slider("Years of Experience", 0, 40, 3)

    # Button lives with the inputs so user presses it after selecting values
    predict_button = st.button("Predict Salary")

with main_right:
    pred_placeholder = st.empty()
    pred_placeholder.info("Prediction will appear here after pressing the button.")


# Build input dataframe (keeps same variable names)
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
    except Exception:
        with main_right:
            pred_placeholder.error(f"Unknown category for {col}. Please update label encoders.")
        st.stop()

# Predicting the salary and render the result in the right column
if predict_button:
    prediction = model.predict(input_data)[0]
    with main_right:
        pred_placeholder.markdown("### Predicted Salary")
        pred_placeholder.success(f"Estimated Salary: **${prediction:,.2f}**")
        st.caption("Prediction generated using an ensemble regression model.")
