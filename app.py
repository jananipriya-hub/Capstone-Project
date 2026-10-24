import os
import streamlit as st
import utils

st.set_page_config(page_title="Credit Risk Assessment System", page_icon="💳", layout="wide")


@st.cache_resource
def load_artifacts():
    return utils.load_model()


model, feature_names = load_artifacts()


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "show_login" not in st.session_state:
    st.session_state.show_login = False
if "email" not in st.session_state:
    st.session_state.email = ""

dark = st.session_state.theme == "dark"


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


if st.session_state.show_login and not st.session_state.logged_in:
    st.title("Log In")

    tab_in, tab_up = st.tabs(["Log In", "Sign Up"])

    with tab_in:
        with st.form("login_form"):
            email = st.text_input("Email")
            pw = st.text_input("Password", type="password")
            ok = st.form_submit_button("Log In")
        if ok:
            email = email.strip().lower()
            if utils.check_login(email, pw):
                st.session_state.logged_in = True
                st.session_state.email = email
                st.session_state.show_login = False
                st.rerun()
            else:
                st.error("Wrong email or password")

    with tab_up:
        with st.form("signup_form"):
            new_email = st.text_input("Email")
            new_pw = st.text_input("Password", type="password",
                                   help="Min 8 characters with upper case, lower case, a number and a special character")
            new_pw2 = st.text_input("Confirm password", type="password")
            create = st.form_submit_button("Create Account")
        if create:
            new_email = new_email.strip().lower()
            if not utils.EMAIL_RE.match(new_email):
                st.error("Enter a valid email address")
            elif not utils.PASS_RE.match(new_pw):
                st.error("Password needs 8+ characters with upper case, lower case, a number and a special character, and no spaces")
            elif new_pw != new_pw2:
                st.error("Passwords do not match")
            elif utils.signup(new_email, new_pw):
                st.success("Account created, now log in from the Log In tab")
            else:
                st.error("This email is already registered")

    if st.button("Cancel"):
        st.session_state.show_login = False
        st.rerun()

    st.stop()


if page == "Home":
    st.title("Credit Risk Analysis & Underwriting Platform")
    st.write("Welcome to the capstone decision support system. This platform "
             "evaluates applicant default risk using machine learning.")

    st.subheader("Exploratory Data Analysis & Risk Drivers")

    def chart(name, note):
        path = f"assets/{name}_{suffix}.png"
        if os.path.exists(path):
            st.image(path)
            st.caption(note)

    col1, col2 = st.columns(2)
    with col1:
        chart("eda_income_patterns",
              "Officer note: defaulters earn less (median about 40k vs 60k for repaid loans). "
              "Low income needs a closer look, but the two groups overlap a lot, so never decide on income alone.")
        chart("eda_loan_amounts",
              "Officer note: most loans are between 5k and 15k. Requests above about 20k are rare, "
              "so the model has seen fewer similar cases and the score is less certain there.")
        chart("eda_cred_history",
              "Officer note: most applicants have only 2 to 4 years of credit history, so most defaults also "
              "come from that group. A short history means less proof of repayment behaviour.")
    with col2:
        chart("eda_default_trends_intent",
              "Officer note: debt consolidation (29%), medical (27%) and home improvement (26%) loans default "
              "most often. Venture (15%) and education (17%) are the safest purposes.")
        chart("eda_risk_distribution_grade",
              "Officer note: from grade D onwards defaulters outnumber people who repaid. "
              "Grades A to C are mostly repaid, so treat D to G as high risk.")


elif page == "Input & Prediction":
    st.title("Loan Applicant Assessment Engine")
    st.write("Enter applicant details below to calculate default risk. "
             "Field names are the same as the training dataset columns.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            person_age = st.number_input("person_age", 18, 100, 28)
            person_income = st.number_input("person_income ($)", 1000, 6000000, 50000)
            person_home_ownership = st.selectbox("person_home_ownership", utils.HOME)
            person_emp_length = st.number_input("person_emp_length (years)", 0.0, 40.0, 3.0)
            loan_intent = st.selectbox("loan_intent", utils.INTENT)
            loan_grade = st.selectbox("loan_grade", utils.GRADES)
        with col2:
            loan_amnt = st.number_input("loan_amnt ($)", 500, 35000, 10000)
            loan_int_rate = st.number_input("loan_int_rate (%)", 5.0, 25.0, 11.5)
            loan_percent_income = st.number_input("loan_percent_income (loan_amnt / person_income)", 0.0, 1.0, 0.20, step=0.01)
            cb_person_default_on_file = st.selectbox("cb_person_default_on_file", ["N", "Y"])
            cb_person_cred_hist_length = st.number_input("cb_person_cred_hist_length (years)", 0, 30, 4)

        submitted = st.form_submit_button("Assess Applicant Risk")

    if submitted:
        d = {
            "person_age": person_age, "person_income": person_income,
            "person_home_ownership": person_home_ownership, "person_emp_length": person_emp_length,
            "loan_intent": loan_intent, "loan_grade": loan_grade, "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
        }
        prob = utils.predict(model, feature_names, d)

        if prob < 0.3:
            tier, color = "Approved / Prime", "green"
        elif prob <= 0.6:
            tier, color = "Manual Review Needed", "orange"
        else:
            tier, color = "Rejected / Subprime", "red"

        st.markdown("---")
        st.subheader("Assessment Result")
        if abs(loan_percent_income - loan_amnt / person_income) > 0.05:
            st.warning(f"loan_percent_income does not match loan_amnt / person_income "
                       f"({loan_amnt / person_income:.2f}). Check the values.")
        st.metric("Default Probability", f"{prob:.1%}")
        st.write(f"**Predicted loan_status:** {1 if prob > 0.5 else 0} "
                 f"({'will default' if prob > 0.5 else 'will repay'})")
        st.markdown(f"**Risk Tier:** :{color}[{tier}]")

        with st.expander("Why This Score?", expanded=True):
            up, down = utils.explain(model, feature_names, d, prob)
            st.caption("Each factor is swapped for a typical lower-risk value to see how much the score would change.")
            for name, change in up:
                st.write(f"🔺 **{name}** = {d[name]} adds about {change * 100:.0f} points of risk")
            for name, change in down:
                st.write(f"🔻 **{name}** = {d[name]} lowers risk by about {-change * 100:.0f} points")
            if not up and not down:
                st.write("No single factor moves the score much, it comes from the overall combination.")

        if st.session_state.logged_in:
            utils.save_assessment(st.session_state.email, d, float(prob), tier)
            st.caption("Saved to your past records.")
        else:
            st.caption("Log in to save this assessment.")


elif page == "Past Records":
    st.title("Audit History Log")

    if not st.session_state.logged_in:
        st.warning("Please log in using the button at the top right to view past records.")
    else:
        hist = utils.get_history(st.session_state.email)
        if hist.empty:
            st.info("No assessments saved yet. Run a prediction first!")
        else:
            st.dataframe(hist, use_container_width=True)
