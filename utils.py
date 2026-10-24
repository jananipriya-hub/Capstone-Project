import hashlib
import hmac
import json
import os
import re
import sqlite3
from datetime import datetime

import joblib
import pandas as pd

DB_FILE = "credit_risk.db"

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z]{2,})+$")
# 8+ chars, one lower, one upper, one digit, one special char, no spaces
PASS_RE = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^\w\s])\S{8,}$")

HOME = ["RENT", "OWN", "MORTGAGE", "OTHER"]
INTENT = ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
GRADES = list("ABCDEFG")

# a typical lower-risk applicant, used only to explain a score
BASELINE = {
    "person_income": 60000, "loan_amnt": 8000, "loan_grade": "B",
    "loan_int_rate": 10.0, "loan_percent_income": 0.15,
    "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 6,
    "person_emp_length": 5, "person_home_ownership": "MORTGAGE",
    "loan_intent": "VENTURE",
}


def load_model():
    model = joblib.load("model/credit_risk_rf_model.pkl")
    cols = joblib.load("model/feature_names.pkl")
    return model, cols


# ---------- database ----------
def db():
    con = sqlite3.connect(DB_FILE)
    con.execute("CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, salt BLOB, pw BLOB)")
    con.execute("""CREATE TABLE IF NOT EXISTS assessments (
        id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, created TEXT,
        inputs TEXT, probability REAL, tier TEXT)""")
    return con


def _hash(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 200_000)


def signup(email, pw):
    con = db()
    try:
        salt = os.urandom(16)
        con.execute("INSERT INTO users VALUES (?,?,?)", (email, salt, _hash(pw, salt)))
        con.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        con.close()


def check_login(email, pw):
    con = db()
    row = con.execute("SELECT salt, pw FROM users WHERE email=?", (email,)).fetchone()
    con.close()
    return bool(row) and hmac.compare_digest(row[1], _hash(pw, row[0]))


def save_assessment(email, inputs, prob, tier):
    con = db()
    con.execute("INSERT INTO assessments (email, created, inputs, probability, tier) VALUES (?,?,?,?,?)",
                (email, datetime.now().strftime("%Y-%m-%d %H:%M"), json.dumps(inputs), prob, tier))
    con.commit()
    con.close()


def get_history(email):
    con = db()
    rows = con.execute("SELECT created, inputs, probability, tier FROM assessments "
                       "WHERE email=? ORDER BY id DESC", (email,)).fetchall()
    con.close()
    out = []
    for created, inputs, prob, tier in rows:
        out.append({"time": created, **json.loads(inputs),
                    "default_probability": f"{prob:.1%}", "tier": tier})
    return pd.DataFrame(out)


# ---------- model input ----------
def to_row(d, cols):
    """turn the 11 app inputs into the same 17 columns the model was trained on"""
    row = dict.fromkeys(cols, 0)
    for c in ["person_age", "person_income", "person_emp_length", "loan_amnt",
              "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]:
        row[c] = d[c]
    row["loan_grade"] = GRADES.index(d["loan_grade"]) + 1
    row["cb_person_default_on_file"] = 1 if d["cb_person_default_on_file"] == "Y" else 0
    for c in (f"person_home_ownership_{d['person_home_ownership']}",
              f"loan_intent_{d['loan_intent']}"):
        if c in row:          # MORTGAGE and DEBTCONSOLIDATION were dropped in get_dummies
            row[c] = 1
    return pd.DataFrame([row])[cols]


def predict(model, cols, d):
    return model.predict_proba(to_row(d, cols))[0][1]


def explain(model, cols, d, prob):
    """swap one field at a time for a typical value and see how much the score moves"""
    names = list(BASELINE)
    rows = []
    for n in names:
        d2 = dict(d)
        d2[n] = BASELINE[n]
        rows.append(to_row(d2, cols))
    new = model.predict_proba(pd.concat(rows))[:, 1]
    out = [(n, prob - p) for n, p in zip(names, new) if d[n] != BASELINE[n]]
    up = sorted([o for o in out if o[1] > 0.02], key=lambda x: -x[1])[:4]
    down = sorted([o for o in out if o[1] < -0.02], key=lambda x: x[1])[:3]
    return up, down
