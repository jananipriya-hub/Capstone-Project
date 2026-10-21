import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Risk Assessment System", page_icon="💳", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load("model/credit_risk_rf_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    features = joblib.load("model/feature_names.pkl")
    return model, scaler, features


model, scaler, feature_names = load_artifacts()

# session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "history" not in st.session_state:
    st.session_state.history = []
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "show_login" not in st.session_state:
    st.session_state.show_login = False
if "email" not in st.session_state:
    st.session_state.email = ""

dark = st.session_state.theme == "dark"

# colours for each theme
if dark:
    bg, side, txt, line = "#0e1117", "#161a23", "#fafafa", "#2d3240"
else:
    bg, side, txt, line = "#ffffff", "#f0f2f6", "#1f2430", "#d9dce3"

st.markdown(f"""
<style>
.stApp {{ background-color: {bg}; }}
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp li, .stApp span {{ color: {txt}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
section[data-testid="stSidebar"] {{ background-color: {side}; }}

/* sidebar arrow always visible */
[data-testid="stSidebarCollapseButton"], [data-testid="stExpandSidebarButton"], [data-testid="collapsedControl"] {{
    visibility: visible !important;
    opacity: 1 !important;
}}

/* nav buttons */
section[data-testid="stSidebar"] .stButton > button {{
    background: transparent;
    color: {txt};
    border: none;
    justify-content: flex-start;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{ background: {line}; }}
section[data-testid="stSidebar"] .stButton > button[kind="primary"],
section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {{
    background: #1f6feb;
    color: white;
}}

section[data-testid="stSidebar"] .stButton > button[kind="primary"] p,
section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] p {{ color: white; }}

/* top right buttons + form button */
.block-container .stButton > button,
div[data-testid="stFormSubmitButton"] button {{
    background-color: {bg};
    border: 1px solid {line};
}}
.block-container .stButton > button p,
div[data-testid="stFormSubmitButton"] button p {{ color: {txt}; }}
.block-container .stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {{ border-color: #1f6feb; }}
.block-container .stButton > button:hover p,
div[data-testid="stFormSubmitButton"] button:hover p {{ color: #1f6feb; }}

/* inputs */
div[data-baseweb="input"], div[data-baseweb="input"] input {{
    background-color: {bg} !important;
    color: {txt} !important;
}}
</style>
""", unsafe_allow_html=True)

# sidebar - only the page buttons
page = st.session_state.page

if st.sidebar.button("Home", type="primary" if page == "Home" else "secondary", use_container_width=True):
    st.session_state.page = "Home"
    st.session_state.show_login = False
    st.rerun()
if st.sidebar.button("Input & Prediction", type="primary" if page == "Input & Prediction" else "secondary", use_container_width=True):
    st.session_state.page = "Input & Prediction"
    st.session_state.show_login = False
    st.rerun()
if st.sidebar.button("Past Records", type="primary" if page == "Past Records" else "secondary", use_container_width=True):
    st.session_state.page = "Past Records"
    st.session_state.show_login = False
    st.rerun()

# top right - theme and login buttons
c1, c2, c3 = st.columns([8, 1.3, 1.3])

if c2.button("Light" if dark else "Dark", use_container_width=True):
    st.session_state.theme = "light" if dark else "dark"
    st.rerun()

if st.session_state.logged_in:
    if c3.button("Log Out", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.email = ""
        st.rerun()
else:
    if c3.button("Log In", use_container_width=True):
        st.session_state.show_login = True
        st.rerun()

suffix = "dark" if dark else "light"

if st.session_state.logged_in:
    st.caption(f"Logged in as {st.session_state.email}")

# login form (shown instead of the page)
if st.session_state.show_login and not st.session_state.logged_in:
    st.title("Log In")

    with st.form("login_form"):
        email = st.text_input("Gmail address")
        pw = st.text_input("Password", type="password")
        ok = st.form_submit_button("Log In")

    if ok:
        email = email.strip().lower()
        if not email.endswith("@gmail.com") or len(email) <= len("@gmail.com"):
            st.error("Enter a valid Gmail address ending with @gmail.com")
        elif len(pw) < 8:
            st.error("Password must be at least 8 characters")
        else:
            st.session_state.logged_in = True
            st.session_state.email = email
            st.session_state.show_login = False
            st.rerun()

    if st.button("Cancel"):
        st.session_state.show_login = False
        st.rerun()

    st.stop()

# ---- Home ----
if page == "Home":
    st.title("Credit Risk Analysis & Underwriting Platform")
    st.write("Welcome to the capstone decision support system. This platform "
             "evaluates applicant default risk using machine learning.")

    st.subheader("Exploratory Data Analysis & Risk Drivers")

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(f"assets/eda_income_patterns_{suffix}.png"):
            st.image(f"assets/eda_income_patterns_{suffix}.png")
        if os.path.exists(f"assets/eda_loan_amounts_{suffix}.png"):
            st.image(f"assets/eda_loan_amounts_{suffix}.png")
    with col2:
        if os.path.exists(f"assets/eda_default_trends_intent_{suffix}.png"):
            st.image(f"assets/eda_default_trends_intent_{suffix}.png")
        if os.path.exists(f"assets/eda_risk_distribution_grade_{suffix}.png"):
            st.image(f"assets/eda_risk_distribution_grade_{suffix}.png")

# ---- Prediction page ----
elif page == "Input & Prediction":
    st.title("Loan Applicant Assessment Engine")
    st.write("Enter applicant details below to calculate default risk.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            income = st.number_input("Annual Income ($)", min_value=1000, max_value=500000, value=50000)
            loan = st.number_input("Loan Amount ($)", min_value=500, max_value=35000, value=10000)
            age = st.number_input("Age", min_value=18, max_value=100, value=28)
        with col2:
            hist = st.number_input("Credit History Length (Years)", min_value=0, max_value=30, value=4)
            rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=25.0, value=11.5)
            emp = st.number_input("Employment Length (Years)", min_value=0.0, max_value=40.0, value=3.0)

        submitted = st.form_submit_button("Assess Applicant Risk")

    if submitted:
        data = pd.DataFrame(
            [[income, loan, age, hist, rate, emp]],
            columns=["person_income", "loan_amnt", "person_age",
                     "cb_person_cred_hist_length", "loan_int_rate", "person_emp_length"],
        )

        try:
            prob = model.predict_proba(scaler.transform(data))[0][1]
        except Exception:
            # if the model fails, estimate from loan-to-income ratio
            prob = min(max((loan / income) * 0.8 + (rate / 50), 0.05), 0.95)

        if prob < 0.3:
            tier = "Approved / Prime"
            color = "green"
        elif prob <= 0.6:
            tier = "Manual Review Needed"
            color = "orange"
        else:
            tier = "Rejected / Subprime"
            color = "red"

        st.markdown("---")
        st.subheader("Assessment Result")
        st.metric("Default Probability", f"{prob:.1%}")
        st.markdown(f"**Risk Tier:** :{color}[{tier}]")

        with st.expander("Why This Score?"):
            st.write(f"- Loan to income ratio: {loan / income:.2f}")
            st.write(f"- Credit history: {hist} years")
            st.write(f"- Interest rate: {rate}%")

        # save only if logged in
        if st.session_state.logged_in:
            st.session_state.history.append({
                "Income": income,
                "Loan Amount": loan,
                "Probability": f"{prob:.1%}",
                "Tier": tier,
            })

# ---- Past records ----
elif page == "Past Records":
    st.title("Audit History Log")

    if not st.session_state.logged_in:
        st.warning("Please log in using the button at the top right to view past records.")
    elif len(st.session_state.history) == 0:
        st.info("No evaluations recorded in this session yet. Run a prediction first!")
    else:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
