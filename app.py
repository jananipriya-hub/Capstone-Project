import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Risk Assessment", layout="centered")

st.title("Credit Risk Prediction App")
st.write(
    "Enter applicant details in the sidebar to check loan default"
    " probability."
)

# Load saved model artifacts
model = joblib.load("model/credit_risk_rf_model.pkl")
feature_names = joblib.load("model/feature_names.pkl")

st.sidebar.header("Applicant Details")

age = st.sidebar.number_input("Age", 18, 100, 25)
income = st.sidebar.number_input(
    "Annual Income", 0, 1000000, 45000, step=1000
)
emp_len = st.sidebar.number_input(
    "Employment Length (Years)", 0.0, 50.0, 3.0, step=0.5
)
amount = st.sidebar.number_input("Loan Amount", 500, 100000, 10000, step=500)
rate = st.sidebar.number_input(
    "Interest Rate (%)", 1.0, 30.0, 10.5, step=0.1
)
cred_hist = st.sidebar.number_input("Credit History Length", 0, 50, 3)

default_history = st.sidebar.selectbox("Past Default on File", ["No", "Yes"])
grade = st.sidebar.selectbox(
    "Loan Grade", ["A", "B", "C", "D", "E", "F", "G"]
)
home = st.sidebar.selectbox(
    "Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]
)
intent = st.sidebar.selectbox(
    "Loan Purpose",
    [
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    ],
)

# Feature mapping
grade_dict = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
percent_income = amount / income if income > 0 else 0

row = {
    "person_age": age,
    "person_income": income,
    "person_emp_length": emp_len,
    "loan_grade": grade_dict[grade],
    "loan_amnt": amount,
    "loan_int_rate": rate,
    "loan_percent_income": percent_income,
    "cb_person_default_on_file": 1 if default_history == "Yes" else 0,
    "cb_person_cred_hist_length": cred_hist,
}

# One-hot encoded indicators
for col in feature_names:
  if col.startswith("person_home_ownership_"):
    cat = col.replace("person_home_ownership_", "")
    row[col] = 1 if home == cat else 0
  elif col.startswith("loan_intent_"):
    cat = col.replace("loan_intent_", "")
    row[col] = 1 if intent == cat else 0

input_df = pd.DataFrame([row])[feature_names]

if st.button("Predict"):
  prob = model.predict_proba(input_df)[0][1]
  pct = round(prob * 100, 2)

  st.subheader(f"Default Probability: {pct}%")

  if prob < 0.30:
    st.success("Decision: Approved")
  elif prob <= 0.60:
    st.warning("Decision: Manual Review Needed")
  else:
    st.error("Decision: Rejected")

  st.write("---")
  st.write("Input Summary:")
  st.dataframe(input_df)