# Credit Risk Assessment & Prediction System

An end-to-end Machine Learning project that evaluates loan applicant default risk using financial, demographic, and credit history data.

### 🌐 Live Interactive Application
👉 **[Click here to open the Credit Risk Assessment App](https://capstone-project-9jsaudndaqr9whp4oodcnw.streamlit.app/)**

---

## 📌 Project Overview
Credit risk modeling helps financial institutions automate loan underwriting and minimize default rates. This project analyzes a dataset of 32,581 applicant records, performs data cleaning and exploratory analysis, trains predictive machine learning models, and deploys an interactive web interface for real-time risk scoring.

---

## 📊 Exploratory Data Analysis & Key Insights
The exploratory analysis revealed key default risk indicators:
- **Income Level:** Lower income brackets present significantly higher default probabilities.
- **Loan Grade:** Defaults increase drastically from Grade A to Grade G.
- **Credit History & Intent:** Prior historical default on file is a high-risk factor across medical and personal loan intents.

*(EDA charts exported and saved in `assets/` directory)*

---

## ⚙️ Model Performance & Results
Two machine learning models were developed and evaluated:

| Model | Class Weighting | Key Metric (ROC-AUC) | Decision Recommendation |
| :--- | :--- | :--- | :--- |
| **Logistic Regression (Baseline)** | Balanced | ~0.84 | Baseline linear boundary |
| **Random Forest Classifier (Final)** | Balanced | **~0.93** | Selected for web deployment |

The **Random Forest Classifier** was selected for production deployment due to its superior ROC-AUC score and strong handling of non-linear financial interactions.

---

## 🛠️ Underwriting Decision Logic
The deployed Streamlit app categorizes default risk into 3 actionable tiers:
- **Approved (Green):** Calculated default probability < 30%
- **Manual Review Needed (Yellow):** Calculated default probability between 30% and 60%
- **Rejected (Red):** Calculated default probability > 60%

---

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/jananipriya-hub/Capstone-Project.git](https://github.com/jananipriya-hub/Capstone-Project.git)
   cd Capstone-Project
