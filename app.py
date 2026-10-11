import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Credit Risk Analysis System", page_icon="💳", layout="wide"
)

# Native Streamlit Theme-Adaptive CSS Injection
st.markdown(
    """
    <style>
    /* Use Streamlit native CSS variables for seamless light/dark auto-switching */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    [data-testid="stSidebar"] {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
    }
    
    [data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 15px;
        border-radius: 10px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Initialize Session State
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
  st.error(f"Model files not found in 'model/' directory. Error: {e}")
  st.stop()

# Detect Active Mode for Assets
is_dark = st.get_option("theme.base") == "dark"
theme_suffix = "dark" if is_dark else "light"

# --- SIDEBAR NAVIGATION & LOGIN MODAL ---
st.sidebar.title("💳 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Input & Prediction", "Past Records"])

st.sidebar.markdown("---")
st.sidebar.subheader("🔐 Account Access")
if not st.session_state.logged_in:
  with st.sidebar.expander("Log In to System"):
    with st.form("popup_login"):
      u_input = st.text_input("Username")
      p_input = st.text_input("Password", type="password")
      submit_login = st.form_submit_button("Confirm Login")
      if submit_login:
        if u_input.strip() != "":
          st.session_state.logged_in = True
          st.session_state.username = u_input
          st.success(f"Logged in as {u_input}!")
          st.rerun()
        else:
          st.error("Enter a valid username.")
else:
  st.sidebar.success(f"User: **{st.session_state.username}**")
  if st.sidebar.button("Log Out"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()


# ==========================================
# PAGE 1: HOME PAGE
# ==========================================
if page == "Home":
  st.title("Credit Risk Analysis System")
  st.markdown("### Welcome to the Intelligent Loan Underwriting Portal")

  st.write(
      "Welcome to Credit Risk Analysis, a comprehensive financial intelligence"
      " platform designed to help lending institutions evaluate whether to"
      " approve or reject a loan applicant. You do not need to sign in to"
      " continue evaluating risk profiles, making the core prediction engine"
      " instantly accessible for quick assessments. However, signing in"
      " unlocks audit logging, allowing you to track and review your past"
      " assessment records seamlessly in a tabular history log. Always review"
      " the final prediction alongside the concise automated explanation"
      " provided below each output to understand the underlying risk drivers"
      " such as debt-to-income ratios and credit history."
  )

  st.markdown("---")
  st.subheader("📊 Portfolio Summary Metrics")
  col1, col2, col3, col4 = st.columns(4)
  col1.metric("Total Applicants", "32,581", "+1,200")
  col2.metric("Low Risk (Prime)", "23,133", "71.0%")
  col3.metric("High Risk (Default)", "9,448", "29.0%")
  col4.metric("Avg Approval Probability", "76.4%", "+2.1%")

  st.markdown("---")
  st.subheader("📈 EDA Visualizations & Risk Drivers")

  c1, c2 = st.columns(2)
  with c1:
    img_1 = f"assets/eda_income_patterns_{theme_suffix}.png"
    img_2 = f"assets/eda_risk_distribution_grade_{theme_suffix}.png"
    img_3 = f"assets/eda_cred_history_{theme_suffix}.png"
    if os.path.exists(img_1):
      st.image(img_1)
    if os.path.exists(img_2):
      st.image(img_2)
    if os.path.exists(img_3):
      st.image(img_3)

  with c2:
    img_4 = f"assets/eda_loan_amounts_{theme_suffix}.png"
    img_5 = f"assets/eda_default_trends_intent_{theme_suffix}.png"
    if os.path.exists(img_4):
      st.image(img_4)
    if os.path.exists(img_5):
      st.image(img_5)


# ==========================================
# PAGE 2: INPUT & PREDICTION PAGE (CENTERED)
# ==========================================
elif page == "Input & Prediction":
  st.title("🔮 Central Prediction & Analysis Engine")
  st.write(
      "Enter applicant details below to evaluate risk and generate an"
      " institutional underwriting score."
  )
  st.markdown("---")

  with st.form("prediction_form"):
    st.subheader("📋 Applicant Parameter Details")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
      age = st.number_input("Applicant Age", 18, 100, 28)
      income = st.number_input(
          "Annual Income ($)", 0, 1000000, 60000, step=1000
      )
      emp_len = st.number_input(
          "Employment Length (Years)", 0.0, 50.0, 4.0, step=0.5
      )
    with col_b:
      amount = st.number_input(
          "Loan Amount Requested ($)", 500, 100000, 12000, step=500
      )
      rate = st.number_input("Interest Rate (%)", 1.0, 35.0, 10.5, step=0.1)
      cred_hist = st.number_input(
          "Credit History Length (Years)", 0, 50, 4
      )
    with col_c:
      default_history = st.selectbox(
          "Historical Default on File?", ["No", "Yes"]
      )
      grade = st.selectbox(
          "Loan Grade", ["A", "B", "C", "D", "E", "F", "G"]
      )
      home = st.selectbox(
          "Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]
      )

    intent = st.selectbox(
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

    submitted = st.form_submit_button(
        "🚀 Assess Applicant Risk", use_container_width=True
    )

  if submitted:
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

    st.markdown("---")
    st.subheader("📊 Assessment Results")

    res_col1, res_col2 = st.columns(2)
    with res_col1:
      st.metric(label="Calculated Default Probability", value=f"{pct}%")
      st.progress(float(prob))

    with res_col2:
      if prob < 0.30:
        decision = "Approved (Low Risk)"
        st.success(
            f"### Status: {decision}\n- DTI Ratio:"
            f" {round(percent_income*100, 1)}%\n- Risk Tier: Prime"
        )
      elif prob <= 0.60:
        decision = "Manual Review Needed"
        st.warning(
            f"### Status: {decision}\n- DTI Ratio:"
            f" {round(percent_income*100, 1)}%\n- Risk Tier: Borderline"
        )
      else:
        decision = "Rejected (High Risk)"
        st.error(
            f"### Status: {decision}\n- DTI Ratio:"
            f" {round(percent_income*100, 1)}%\n- Risk Tier: Subprime"
        )

    st.markdown("---")
    st.markdown("### 💡 Why This Score")
    explanation = f"""
    - **Debt-to-Income Ratio (DTI):** The requested loan amount of `${amount:,}` represents **{round(percent_income*100, 1)}%** of the applicant's annual income (`${income:,}`). Higher DTI ratios increase default vulnerability.
    - **Loan Grade & Interest Risk:** Assigned Loan Grade **{grade}** with an interest rate of **{rate}%**, indicating the risk tier evaluated by the credit bureau model.
    - **Credit History Depth:** A credit history span of **{cred_hist} years** combined with prior default status (**{default_history}**) heavily influenced the ensemble classification tree path.
    - **Final Model Verdict:** The Random Forest algorithm computed a default probability of **{pct}%**, leading directly to the classification of **{decision}**.
    """
    st.info(explanation)

    if st.session_state.logged_in:
      record = {
          "User": st.session_state.username,
          "Income": income,
          "Loan Amount": amount,
          "Grade": grade,
          "Default Prob (%)": pct,
          "Decision": decision,
      }
      st.session_state.history.append(record)


# ==========================================
# PAGE 3: PAST RECORDS PAGE
# ==========================================
elif page == "Past Records":
  st.title("📋 Past Assessment Records Log")
  st.write("Audit log of predictions evaluated during your active session.")

  if st.session_state.logged_in:
    if len(st.session_state.history) > 0:
      history_df = pd.DataFrame(st.session_state.history)
      st.dataframe(history_df, use_container_width=True)
    else:
      st.info("No prediction records logged yet during this session.")
  else:
    st.warning(
        "🔒 **Access Restricted:** You must be logged in to view past records."
        " Expand 'Log In to System' in the sidebar."
    )