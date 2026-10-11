import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Credit Risk Analysis System", page_icon="💳", layout="wide"
)

# Load Model & Features
@st.cache_resource
def load_artifacts():
  model = joblib.load("model/credit_risk_rf_model.pkl")
  feature_names = joblib.load("model/feature_names.pkl")
  return model, feature_names


try:
  model, feature_names = load_artifacts()
except Exception as e:
  st.error(
      f"Model files not found in 'model/' directory. Please ensure artifacts"
      f" are trained. Error: {e}"
  )
  st.stop()

# Header Banner
st.title("Credit Risk Analysis System")
st.markdown("Predict. Analyze. Make Better Decisions.")
st.markdown("---")

# Sidebar Navigation / Inputs
st.sidebar.header("⚙️ Applicant Parameters")
age = st.sidebar.number_input("Applicant Age", 18, 100, 25)
income = st.sidebar.number_input(
    "Annual Income ($)", 0, 1000000, 50000, step=1000
)
emp_len = st.sidebar.number_input(
    "Employment Length (Years)", 0.0, 50.0, 3.0, step=0.5
)
amount = st.sidebar.number_input(
    "Loan Amount Requested ($)", 500, 100000, 15000, step=500
)
rate = st.sidebar.number_input("Interest Rate (%)", 1.0, 35.0, 11.5, step=0.1)
cred_hist = st.sidebar.number_input("Credit History Length (Years)", 0, 50, 5)

default_history = st.sidebar.selectbox("Historical Default on File?", ["No", "Yes"])
grade = st.sidebar.selectbox(
    "Loan Grade", ["A", "B", "C", "D", "E", "F", "G"]
)
home = st.sidebar.selectbox(
    "Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]
)
intent = st.sidebar.selectbox(
    "Loan Intent",
    [
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    ],
)

# Main Navigation Tabs
tab_home, tab_eda, tab_predict = st.tabs(
    ["🏠 Home / Overview", "📊 EDA & Insights", "🔮 Live Prediction Engine"]
)

with tab_home:
  st.subheader("Portfolio Summary Metrics")
  col1, col2, col3, col4 = st.columns(4)
  col1.metric("Total Applicants", "32,581", "+1,200 this month")
  col2.metric("Low Risk (Prime)", "23,133", "71.0%")
  col3.metric("High Risk (Default)", "9,448", "29.0%")
  col4.metric("Avg Approval Probability", "76.4%", "+2.1%")

  st.markdown("---")
  st.markdown("### Executive Summary")
  st.write(
      "This dashboard serves as an automated underwriting command center."
      " Utilizing a Random Forest machine learning pipeline trained on over"
      " 32,000 historical loan records, it evaluates multi-dimensional risk"
      " factors in real time to optimize lending decisions and minimize portfolio"
      " default rates."
  )

  # Quick Mock Chart or Summary Box
  st.info(
      "💡 **Tip:** Navigate to the **Live Prediction Engine** tab or use the"
      " sidebar to test custom applicant profiles instantly."
  )

with tab_eda:
  st.subheader("Exploratory Data Analysis & Risk Drivers")
  st.write(
      "Visualizing core correlations between financial attributes and default"
      " status."
  )

  col1, col2 = st.columns(2)
  with col1:
    st.markdown("#### Income vs Loan Amount Distribution")
    # Display saved assets if available
    try:
      st.image(
          "assets/eda_income_patterns.png",
          caption="Income vs Loan Default Patterns",
      )
    except:
      st.warning(
          "Run your notebook EDA cells to generate assets/eda_income_patterns.png"
      )

  with col2:
    st.markdown("#### Risk Distribution by Loan Grade")
    try:
      st.image(
          "assets/eda_risk_distribution_grade.png",
          caption="Default Rates Across Loan Grades A-G",
      )
    except:
      st.warning(
          "Run your notebook EDA cells to generate"
          " assets/eda_risk_distribution_grade.png"
      )

with tab_predict:
  st.subheader("Interactive Underwriting Prediction Engine")
  st.write(
      "Review the parameters selected in the sidebar and click assess below to"
      " generate an institutional risk score."
  )

  # Feature formatting
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

  for col in feature_names:
    if col.startswith("person_home_ownership_"):
      cat = col.replace("person_home_ownership_", "")
      row[col] = 1 if home == cat else 0
    elif col.startswith("loan_intent_"):
      cat = col.replace("loan_intent_", "")
      row[col] = 1 if intent == cat else 0

  input_df = pd.DataFrame([row])[feature_names]

  if st.button("Assess Applicant Risk", type="primary", use_container_width=True):
    prob = model.predict_proba(input_df)[0][1]
    pct = round(prob * 100, 2)

    res_col1, res_col2 = st.columns(2)

    with res_col1:
      st.metric(label="Calculated Default Probability", value=f"{pct}%")
      st.progress(float(prob))

    with res_col2:
      if prob < 0.30:
        st.success(
            "### Status: Low Risk (Approved)\nRisk profile is well within prime"
            " lending criteria."
        )
      elif prob <= 0.60:
        st.warning(
            "### Status: Moderate Risk (Manual Review)\nSecondary income and"
            " credit verification required."
        )
      else:
        st.error(
            "### Status: High Risk (Rejected)\nDefault probability exceeds"
            " institutional risk appetite threshold."
        )

    st.markdown("---")
    st.markdown("#### Processed Feature Vector for Model Inference")
    st.dataframe(input_df.T.rename(columns={0: "Value"}))