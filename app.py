import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Credit Risk Assessment System", page_icon="💳", layout="wide")

DATA_PATH = "credit_risk_dataset_cleaned1.csv"
GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
HOME_OPTIONS = ["RENT", "OWN", "MORTGAGE", "OTHER"]
INTENT_OPTIONS = ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]


@st.cache_resource
def load_artifacts():
    # The Random Forest was trained on raw (unscaled) features, so no scaler is used here.
    model = joblib.load("model/credit_risk_rf_model.pkl")
    features = joblib.load("model/feature_names.pkl")
    return model, features


@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


model, feature_names = load_artifacts()


def risk_tier(prob):
    if prob < 0.3:
        return "Approved / Prime", "green"
    if prob <= 0.6:
        return "Manual Review Needed", "orange"
    return "Rejected / Subprime", "red"


def importance_chart(top_n, text_color):
    """Horizontal bar chart of the Random Forest's most important features."""
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(5, 0.45 * top_n + 0.8))
    sns.barplot(x=imp.values, y=imp.index, color="#1f6feb", ax=ax)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.set_xlabel("Importance", color=text_color)
    ax.set_ylabel("")
    ax.tick_params(colors=text_color, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


# session state
defaults = {
    "logged_in": False,
    "history": [],
    "theme": "light",
    "page": "Home",
    "show_login": False,
    "email": "",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

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

/* metric numbers and labels (Snapshot, Assessment Result) */
[data-testid="stMetricValue"], [data-testid="stMetricValue"] *,
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *,
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * {{
    color: {txt} !important;
}}

/* captions */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{ color: {txt} !important; opacity: 0.8; }}

/* inputs */
div[data-baseweb="input"], div[data-baseweb="input"] input {{
    background-color: {bg} !important;
    color: {txt} !important;
}}
</style>
""", unsafe_allow_html=True)

# sidebar - page buttons
page = st.session_state.page
for name in ["Home", "Input & Prediction", "Past Records"]:
    if st.sidebar.button(name, type="primary" if page == name else "secondary", use_container_width=True):
        st.session_state.page = name
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
    st.caption(f"Logged in as {st.session_state.email} (demo session)")

# login form (shown instead of the page)
if st.session_state.show_login and not st.session_state.logged_in:
    st.title("Demo Log In")
    st.caption("Demo login only: no accounts or passwords are stored. It unlocks saving and viewing past records "
               "for this browser session.")

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

    left, right = st.columns([3, 1])

    # main area: the saved EDA charts (2 x 2)
    with left:
        st.subheader("Exploratory Data Analysis & Risk Drivers")
        col1, col2 = st.columns(2)
        with col1:
            for img in ("eda_income_patterns", "eda_loan_amounts"):
                path = f"assets/{img}_{suffix}.png"
                if os.path.exists(path):
                    st.image(path)
        with col2:
            for img in ("eda_default_trends_intent", "eda_risk_distribution_grade"):
                path = f"assets/{img}_{suffix}.png"
                if os.path.exists(path):
                    st.image(path)

    # right corner: compact snapshot, no filters
    with right:
        st.subheader("Snapshot")
        df = load_data()
        if df is not None:
            st.metric("Applicants", f"{len(df):,}")
            st.metric("Default rate", f"{df['loan_status'].mean():.1%}")
            st.metric("Avg loan amount", f"${df['loan_amnt'].mean():,.0f}")
            st.metric("Avg income", f"${df['person_income'].mean():,.0f}")
        st.markdown("**Top risk drivers**")
        fig = importance_chart(5, txt)
        st.pyplot(fig, transparent=True)
        plt.close(fig)

# ---- Prediction page ----
elif page == "Input & Prediction":
    st.title("Loan Applicant Assessment Engine")
    st.write("Enter applicant details below to calculate default risk.")
    st.caption("Educational demo built on a public dataset. Not for real lending decisions.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            income = st.number_input("Annual Income ($)", min_value=1000, max_value=500000, value=50000)
            loan = st.number_input("Loan Amount ($)", min_value=500, max_value=35000, value=10000)
            age = st.number_input("Age", min_value=18, max_value=100, value=28)
            hist = st.number_input("Credit History Length (Years)", min_value=0, max_value=30, value=4)
            emp = st.number_input("Employment Length (Years)", min_value=0.0, max_value=40.0, value=3.0)
        with col2:
            rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=25.0, value=11.5)
            grade = st.selectbox("Loan Grade", list(GRADE_MAP.keys()), index=1)
            home = st.selectbox("Home Ownership", HOME_OPTIONS)
            intent = st.selectbox("Loan Intent", INTENT_OPTIONS)
            prior_default = st.selectbox("Previous default on file?", ["N", "Y"])

        submitted = st.form_submit_button("Assess Applicant Risk")

    if submitted:
        provided = {
            "person_age": age,
            "person_income": income,
            "person_emp_length": emp,
            "loan_amnt": loan,
            "loan_int_rate": rate,
            "loan_percent_income": loan / income,
            "cb_person_cred_hist_length": hist,
            "loan_grade": GRADE_MAP[grade],
            "cb_person_default_on_file": 1 if prior_default == "Y" else 0,
        }

        # every column the model was trained on must be accounted for
        one_hot_prefixes = ("person_home_ownership_", "loan_intent_")
        unmapped = [c for c in feature_names if c not in provided and not c.startswith(one_hot_prefixes)]
        if unmapped:
            st.error(f"The model expects columns this form does not collect: {unmapped}. "
                     "Add inputs for them in app.py.")
            st.stop()

        row = dict.fromkeys(feature_names, 0)
        row.update({k: v for k, v in provided.items() if k in row})
        for col in (f"person_home_ownership_{home}", f"loan_intent_{intent}"):
            if col in row:  # the dropped first category has no column (all zeros)
                row[col] = 1

        X = pd.DataFrame([row])[feature_names]

        try:
            prob = float(model.predict_proba(X)[0][1])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        tier, color = risk_tier(prob)

        st.markdown("---")
        st.subheader("Assessment Result")
        st.metric("Default Probability", f"{prob:.1%}")
        st.markdown(f"**Risk Tier:** :{color}[{tier}]")

        with st.expander("Why This Score?"):
            st.write("**This applicant**")
            st.write(f"- Loan to income ratio: {loan / income:.2f}")
            st.write(f"- Interest rate: {rate}%")
            st.write(f"- Loan grade: {grade}")
            st.write(f"- Credit history: {hist} years")
            st.write("**What the model weighs most overall**")
            fig = importance_chart(6, txt)
            st.pyplot(fig, transparent=True)
            plt.close(fig)
            st.caption("Importances show the model's general behaviour, not a per-applicant explanation.")

        # save only if logged in
        if st.session_state.logged_in:
            st.session_state.history.append({
                "Income": income,
                "Loan Amount": loan,
                "Grade": grade,
                "Intent": intent,
                "Interest Rate": rate,
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
        st.caption("Records are kept for this browser session only.")
