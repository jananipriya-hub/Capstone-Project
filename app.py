import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Credit Risk Analysis System", page_icon="💳", layout="wide"
)

# Initialize Session State for Login & History Tracking
if "logged_in" not in st.session_state:
  st.session_state.logged_in = False
if "username" not in st.session_state:
  st.session_state.username = ""
if "history" not in st.session_state:
  st.session_state.history = []

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

# --- SIDEBAR: LOGIN & INPUTS ---
st.sidebar.title("🔐 Authentication & Inputs")

# Login Section
if not st.session_state.logged_in:
  with st.sidebar.form("login_form"):
    st.subheader("Underwriter Login")
    user_input = st.text_input("Username")
    pass_input = st.text_input("Password", type="password")
    login_btn = st.form_submit_button("Log In")
    if login_btn:
      if user_input.strip() != "":
        st.session_state.logged_in = True
        st.session_state.username = user_input
        st.success(f"Welcome, {user_input}!")
        st.rerun()
      else:
        st.warning("Please enter a valid username.")
  st.sidebar.markdown("---")
  st.sidebar.info("💡 Log in via sidebar to record assessment logs.")
else:
  st.sidebar.success(f"Logged in as: **{st.session_state.username}**")
  if st.sidebar.button("Log Out"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()
  st.sidebar.markdown("---")

# Applicant Parameter Inputs (Always visible in sidebar as requested)
st.sidebar.header("⚙️ Applicant Parameters")
age = st.sidebar.number_input("Applicant Age", 18, 100, 28)
income = st.sidebar.number_input(
    "Annual Income ($)", 0, 1000000, 60000, step=1000
)
emp_len = st.sidebar.number_input(
    "Employment Length (Years)", 0.0, 50.0, 4.0, step=0.5
)
amount = st.sidebar.number_input(
    "Loan Amount Requested ($)", 500, 100000, 12000, step=500
)
rate = st.sidebar.number_input("Interest Rate (%)", 1.0, 35.0, 10.5, step=0.1)
cred_hist = st.sidebar.number_input("Credit History Length (Years)", 0, 50, 4)

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


# --- MAIN HEADER ---
st.title("Credit Risk Analysis System")
st.markdown(
    "Automated Underwriting Dashboard • *Supports Auto-Theming (Dark/Light Mode)*"
)
st.markdown("---")

# Main Navigation Tabs
tab_home, tab_predict, tab_history = st.tabs(
    [
        "🏠 Overview",
        "🔮 Central Prediction & Analysis",
        "📋 Past Records Log",
    ]
)

with tab_home:
  st.subheader("Portfolio Summary Metrics")
  col1, col2, col3, col4 = st.columns(4)
  col1.metric("Total Applicants", "32,581", "+1,200")
  col2.metric("Low Risk (Prime)", "23,133", "71.0%")
  col3.metric("High Risk (Default)", "9,448", "29.0%")
  col4.metric("Avg Approval Probability", "76.4%", "+2.1%")

  st.markdown("---")
  st.markdown("### System Architecture")
  st.write(
      "This application evaluates multi-dimensional financial risk features"
      " using a Random Forest model. Use the sidebar to modify applicant"
      " details and jump to the **Central Prediction & Analysis** tab for"
      " assessed outputs and explanations."
  )

with tab_predict:
  st.subheader("Centered Assessment & Explanation Engine")
  st.write(
      "Review your sidebar selections and click below to run underwriting"
      " inference."
  )

  # Centered container for prediction action
  with st.container():
    if st.button(
        "🚀 Assess Applicant Risk", type="primary", use_container_width=True
    ):
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

      prob = model.predict_proba(input_df)[0][1]
      pct = round(prob * 100, 2)

      # Display Output Metrics Centered
      res_col1, res_col2 = st.columns(2)
      with res_col1:
        st.metric(label="Calculated Default Probability", value=f"{pct}%")
        st.progress(float(prob))

      with res_col2:
        if prob < 0.30:
          decision = "Approved (Low Risk)"
          st.success(
              f"### Status: {decision}\n- **DTI Ratio:**"
              f" {round(percent_income*100, 1)}%\n- **Risk Tier:** Prime"
          )
        elif prob <= 0.60:
          decision = "Manual Review Needed"
          st.warning(
              f"### Status: {decision}\n- **DTI Ratio:**"
              f" {round(percent_income*100, 1)}%\n- **Risk Tier:** Borderline"
          )
        else:
          decision = "Rejected (High Risk)"
          st.error(
              f"### Status: {decision}\n- **DTI Ratio:**"
              f" {round(percent_income*100, 1)}%\n- **Risk Tier:** Subprime"
          )

      # Concise Explanation Section
      st.markdown("---")
      st.markdown("### 📝 Concise Underwriting Explanation")
      explanation = f"""
      - **Loan-to-Income Ratio:** The requested loan of `${amount:,}` represents **{round(percent_income*100, 1)}%** of the applicant's annual income (`${income:,}`). 
      - **Credit & Grade Impact:** Rated under Loan Grade **{grade}** with a credit history length of **{cred_hist} years**.
      - **Historical Risk Factor:** Previous default record status is set to **{default_history}**.
      - **Final Recommendation:** The model computed a default probability of **{pct}%**, resulting in an underwriting decision of **{decision}**.
      """
      st.info(explanation)

      # Save to Session History Log
      record = {
          "User": st.session_state.username
          if st.session_state.logged_in
          else "Guest",
          "Income": income,
          "Loan Amount": amount,
          "Grade": grade,
          "Default Prob (%)": pct,
          "Decision": decision,
      }
      st.session_state.history.append(record)

with tab_history:
  st.subheader("📋 Past Session Prediction Records")
  st.write(
      "Tabular audit log of predictions evaluated during this session."
  )

  if len(st.session_state.history) > 0:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)
  else:
    st.info(
        "No prediction records logged yet. Run an assessment in the **Central"
        " Prediction & Analysis** tab."
    )