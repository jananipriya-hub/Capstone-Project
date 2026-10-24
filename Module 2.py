import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("../credit_risk_dataset_cleaned1.csv")
sns.set_theme(style="whitegrid")

# 1. Customer Income Patterns vs Loan Status
plt.figure(figsize=(8, 5))
sns.boxplot(x="loan_status", y="person_income", data=df[df["person_income"] < 200000], palette="Set2")
plt.title("Income Distribution by Default Status (Filtered < 200k)")
plt.xlabel("Loan Status (0: Non-Default, 1: Default)")
plt.ylabel("Annual Income ($)")
plt.tight_layout()
plt.savefig("assets/eda_income_patterns.png", dpi=300)
plt.show()

# 2. Loan Amounts Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["loan_amnt"], bins=30, kde=True, color="#3498db", element="step")
plt.title("Distribution of Requested Loan Amounts")
plt.xlabel("Loan Amount ($)")
plt.ylabel("Applicant Count")
plt.tight_layout()
plt.savefig("assets/eda_loan_amounts.png", dpi=300)
plt.show()

# 3. Credit History Length vs Default Rate
plt.figure(figsize=(9, 5))
sns.countplot(x="cb_person_cred_hist_length", hue="loan_status", data=df[df["cb_person_cred_hist_length"] <= 15], palette="viridis",)
plt.title("Credit History Length (Years) vs Loan Default Status")
plt.xlabel("Credit History Length (Years)")
plt.ylabel("Count")
plt.legend(["Repaid (0)", "Default (1)"])
plt.tight_layout()
plt.savefig("assets/eda_cred_history.png", dpi=300)
plt.show()

# 4. Default Trends Across Loan Intent
plt.figure(figsize=(9, 5))
default_by_intent = (df.groupby("loan_intent")["loan_status"].mean().sort_values() * 100)
sns.barplot(x=default_by_intent.index, y=default_by_intent.values, palette="mako")
plt.title("Default Rate (%) by Loan Intent")
plt.xlabel("Loan Intent")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig("assets/eda_default_trends_intent.png", dpi=300)
plt.show()

# 5. Risk Distribution Across Loan Grades
plt.figure(figsize=(8, 5))
sns.countplot(x="loan_grade",hue="loan_status", data=df, order=["A", "B", "C", "D", "E", "F", "G"],palette=["#2ecc71", "#e74c3c"])
plt.title("Risk Distribution: Loan Status Across Internal Credit Grades")
plt.xlabel("Loan Grade")
plt.ylabel("Applicant Count")
plt.legend(["Repaid (0)", "Default (1)"])
plt.tight_layout()
plt.savefig("assets/eda_risk_distribution_grade.png", dpi=300)
plt.show()
