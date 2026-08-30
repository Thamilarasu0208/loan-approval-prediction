import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

MODEL_PATH = "../models/final_loan_approval_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error("Unable to load the loan prediction model.")
    st.error(str(e))
    st.stop()

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🏦 Loan Approval Prediction")
st.write(
    "Enter the applicant details below to predict the loan approval status."
)

st.divider()

# --------------------------------------------------
# Applicant Details
# --------------------------------------------------

st.subheader("Applicant Information")

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

married = st.selectbox(
    "Married",
    ["No", "Yes"]
)

dependents = st.selectbox(
    "Dependents",
    ["0", "1", "2", "3+"]
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["No", "Yes"]
)

employment_type = st.selectbox(
    "Employment Type",
    [
        "Business Owner",
        "Freelancer",
        "Salaried",
        "Self-Employed",
        "Unemployed"
    ]
)

# --------------------------------------------------
# Financial Information
# --------------------------------------------------

st.subheader("Financial Information")

applicant_income = st.number_input(
    "Applicant Income",
    min_value=0.0,
    value=5000.0,
    step=100.0
)

coapplicant_income = st.number_input(
    "Coapplicant Income",
    min_value=0.0,
    value=0.0,
    step=100.0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0.0,
    value=150.0,
    step=10.0
)

loan_amount_term = st.number_input(
    "Loan Amount Term",
    min_value=1.0,
    value=360.0,
    step=1.0
)

credit_history = st.selectbox(
    "Credit History",
    [1.0, 0.0]
)

property_area = st.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"]
)

# --------------------------------------------------
# Encoding
# --------------------------------------------------

gender_map = {
    "Female": 0,
    "Male": 1
}

married_map = {
    "No": 0,
    "Yes": 1
}

dependents_map = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3+": 3
}

education_map = {
    "Graduate": 0,
    "Not Graduate": 1
}

self_employed_map = {
    "No": 0,
    "Yes": 1
}

employment_type_map = {
    "Business Owner": 0,
    "Freelancer": 1,
    "Salaried": 2,
    "Self-Employed": 3,
    "Unemployed": 4
}

property_area_map = {
    "Rural": 0,
    "Semiurban": 1,
    "Urban": 2
}

# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

if st.button("🔮 Predict Loan Status", use_container_width=True):

    input_data = pd.DataFrame({
        "Gender": [gender_map[gender]],
        "Married": [married_map[married]],
        "Dependents": [dependents_map[dependents]],
        "Education": [education_map[education]],
        "Self_Employed": [self_employed_map[self_employed]],
        "EmploymentType": [employment_type_map[employment_type]],
        "ApplicantIncome": [applicant_income],
        "CoapplicantIncome": [coapplicant_income],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_amount_term],
        "Credit_History": [credit_history],
        "Property_Area": [property_area_map[property_area]]
    })

    # Ensure exact feature order
    input_data = input_data[
        [
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "EmploymentType",
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History",
            "Property_Area"
        ]
    ]

    prediction = model.predict(input_data)[0]

    # --------------------------------------------------
    # Display Result
    # --------------------------------------------------

    if prediction == 1:
        st.success("✅ LOAN APPROVED")
        st.write("The model predicts that the loan application is likely to be approved.")
    else:
        st.error("❌ LOAN REJECTED")
        st.write("The model predicts that the loan application is likely to be rejected.")

    # Show encoded input for transparency
    with st.expander("View Processed Input"):
        st.dataframe(input_data)
