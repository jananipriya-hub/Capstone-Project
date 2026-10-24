# Credit Risk Assessment & Prediction System

An end-to-end Machine Learning project that evaluates loan applicant default risk using financial, demographic, and credit history data.

###  Live Interactive Application
 **[Click here to open the Credit Risk Assessment App](https://capstone-project-9jsaudndaqr9whp4oodcnw.streamlit.app/)**

---

##  Project Overview
Manual loan approval is slow, and it gets harder as the number of applications grows. This project builds a faster, more consistent alternative. It uses a dataset of 32,581 applicant records, cleans the data, explores it to find what drives defaults, trains machine learning models, and deploys the best one in an interactive Streamlit app.

For every applicant, the app gives a default probability, a recommended decision (approve, manual review or reject), and a "Why This Score" breakdown that explains the result.

---

##  Exploratory Data Analysis & Key Insights
The exploratory analysis revealed key default risk indicators:
- **Income Level:**  Applicants who defaulted tend to earn less. The median income of defaulters is about 40,000, against about 60,000 for those who repaid. The two groups overlap a lot, so income alone should never decide an application.
- **Loan Grade:**  Grades A to C mostly repay. From grade D onwards, defaulters outnumber those who repaid, so grades D to G are high risk.
- **Credit History & Intent:** Prior historical default on file is a high-risk factor across medical and personal loan intents.
- **Loan intent:** Debt consolidation (29%), medical (27%) and home improvement (26%) loans default most often. Venture (15%) and education (17%) loans are the safest.
-**Credit history:** Most applicants have only 2 to 4 years of credit history, and most defaults come from this group. Defaults become much less common with longer histories.
- **Loan amount:** Most loans are between 5,000 and 15,000. Requests above about 20,000 are rare, so the model's score is less certain for them.

*(EDA charts exported and saved in `assets/` directory)*

---

## Model Performance & Results
Two machine learning models were developed and evaluated:

| Model | Class Weighting | Key Metric (ROC-AUC) | Decision Recommendation |
| :--- | :--- | :--- | :--- |
| **Logistic Regression (Baseline)** | Balanced | ~0.84 | Baseline linear boundary |
| **Random Forest Classifier (Final)** | Balanced | **~0.93** | Selected for web deployment |

The **Random Forest Classifier** The Random Forest was chosen because it has ROC-AUC score higher and handles non-linear relationships between financial features better than the linear baseline.

---

##  Underwriting Decision Logic
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
