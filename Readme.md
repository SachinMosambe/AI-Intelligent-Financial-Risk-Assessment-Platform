# 💰 EMIPredict AI – Intelligent Financial Risk Assessment Platform

EMIPredict AI is a full-stack **FinTech Machine Learning platform** designed to provide **real-time EMI eligibility classification and maximum EMI amount prediction** using advanced ML models and a large-scale financial dataset. The platform integrates **data engineering, machine learning, MLflow tracking, and Streamlit Cloud deployment** into a single production-ready application.

---

## 📌 Project Overview

* **Domain:** FinTech & Banking
* **Type:** End-to-End Machine Learning Web Application
* **Dataset Size:** 400,000 financial records
* **ML Problems Solved:**

  * ✅ EMI Eligibility Classification
  * ✅ Maximum EMI Amount Prediction

This platform addresses the real-world problem of **poor financial planning and high loan default risk** by delivering **data-driven, explainable EMI decisions**.

---

## 🎯 Key Objectives

* Automate EMI eligibility assessment
* Predict safe monthly EMI amount
* Perform large-scale financial risk analysis
* Track model performance using MLflow
* Deploy a real-time prediction system using Streamlit

---

## 🧠 Skills & Technologies Used

* **Programming:** Python
* **Web App:** Streamlit
* **Machine Learning:** Classification & Regression
* **Models:** Logistic Regression, Random Forest, XGBoost
* **Data Processing:** Pandas, NumPy
* **Visualization & EDA:** Matplotlib, Seaborn
* **Experiment Tracking:** MLflow
* **Deployment:** Streamlit Cloud
* **Version Control:** Git & GitHub

---

## 🏦 Business Use Cases

### ✅ Financial Institutions

* Automated loan approval system
* Risk-based EMI pricing
* Real-time customer eligibility checks

### ✅ FinTech Companies

* Instant EMI eligibility APIs
* Digital loan pre-qualification
* Automated borrower risk scoring

### ✅ Banks & Credit Agencies

* Data-driven loan recommendations
* Portfolio risk monitoring
* Regulatory compliance

### ✅ Loan Officers & Underwriters

* AI-powered loan approval recommendations
* Financial profile analysis in seconds
* Risk and default probability tracking

---

## 🗂️ Dataset Summary

* **Total Records:** 400,000
* **Input Features:** 22
* **Target Variables:** 2
* **EMI Categories:** 5 Scenarios

### EMI Scenarios

| Scenario            | Records | Loan Amount | Tenure       |
| ------------------- | ------- | ----------- | ------------ |
| E-commerce EMI      | 80K     | ₹10K–₹200K  | 3–24 Months  |
| Home Appliances EMI | 80K     | ₹20K–₹300K  | 6–36 Months  |
| Vehicle EMI         | 80K     | ₹80K–₹15L   | 12–84 Months |
| Personal Loan EMI   | 80K     | ₹50K–₹10L   | 12–60 Months |
| Education EMI       | 80K     | ₹50K–₹5L    | 6–48 Months  |

---

## 🧾 Input Features (22 Variables)

### Personal Information

* Age, Gender, Marital Status, Education

### Employment & Income

* Monthly Salary, Employment Type, Years of Employment, Company Type

### Housing & Family

* House Type, Monthly Rent, Family Size, Dependents

### Monthly Financial Expenses

* School Fees, College Fees, Travel Expenses, Groceries, Other Expenses

### Credit & Financial History

* Existing Loans, Current EMI, Credit Score, Bank Balance, Emergency Fund

### Loan Details

* EMI Scenario, Requested Amount, Requested Tenure

---

## 🎯 Target Variables

### Classification Target

* EMI Eligibility:

  * ✅ Eligible
  * ⚠️ High Risk
  * ❌ Not Eligible

### Regression Target

* Maximum Safe Monthly EMI (₹500 – ₹50,000)

---

## 🛠️ Project Workflow

### ✅ Step 1: Data Loading & Preprocessing

* Missing value handling
* Duplicate removal
* Data validation
* Train-test split

### ✅ Step 2: Exploratory Data Analysis (EDA)

* EMI eligibility distributions
* Financial risk correlations
* Demographic behavior analysis
* Statistical summaries

### ✅ Step 3: Feature Engineering

* Debt-to-income ratios
* Expense-to-income ratios
* Credit stability metrics
* Categorical encoding and scaling

### ✅ Step 4: Model Development

#### Classification Models

* Logistic Regression
* Random Forest
* XGBoost
* Decision Tree
* Gradient Boosting

#### Regression Models

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor
* Decision Tree Regressor

### ✅ Step 5: Model Selection & MLflow Tracking

* Model comparison via MLflow
* Metric logging
* Best model selection
* Model version tracking

### ✅ Step 6: Streamlit Application Development

* Multi-page interface
* Real-time prediction system
* Interactive data explorer
* Model performance dashboard
* Administrative system panel

### ✅ Step 7: Cloud Deployment

* Deployed via Streamlit Cloud
* GitHub CI/CD integration
* Fully production-ready architecture

---

## 📊 Expected Results

* ✅ Classification Accuracy > 90%
* ✅ Regression RMSE < ₹2000
* ✅ Real-time prediction support
* ✅ Scalable architecture for enterprise usage

---

## 📁 Project Folder Structure

```
ASSIGNMENT_2/
│
├── data/
│   ├── clean_emi_data.csv
│   ├── emi_prediction_dataset.csv
│
├── mlflow_artifacts/
├── model_cache/
│
├── Notebook/
│   ├── classification_feature_engineering.py
│   ├── classification_notebook.ipynb
│   ├── regression_feature_engineering.py
│   ├── regression_notebook.ipynb
│   ├── Exploratory_data_analysis.ipynb
│   ├── dataset_info.txt
│   ├── dataset_info_regression.txt
│
├── app.py
├── data_cleaning.ipynb
├── requirements.txt
├── Readme.md
└── venv/
```

---

## 🧪 Model Performance Summary

### ✅ Final Classification Model: XGBoost

* Accuracy: **94.82%**
* F1-Score: **95.35%**
* Precision: **96.31%**
* Recall: **94.82%**
* ROC-AUC: **0.9953**

### ✅ Final Regression Model: XGBoost

* RMSE: **₹987.76**
* MAE: **₹473.51**
* MAPE: **15.75%**
* R² Score: **0.9778**

---

## 🚀 Application Features

* ✅ Real-time EMI eligibility prediction
* ✅ Maximum EMI affordability estimation
* ✅ Interactive Data Explorer dashboard
* ✅ Model Performance Monitoring
* ✅ System Information & API Health Panel
* ✅ Secure and fast cloud deployment

---

## 📈 Business Impact

* ⏱️ 80% reduction in loan processing time
* 📊 Data-driven EMI recommendations
* ✅ Reduced default risk
* 🏦 Enterprise-ready lending intelligence

---

## 🏁 Deliverables

* Data preprocessing pipeline
* Feature engineering modules
* EDA dashboards & reports
* Trained classification & regression models
* MLflow experiment tracking
* Fully deployed Streamlit web application
* GitHub project repository & documentation

---
🌐 Live Application

🔗 https://ai-intelligent-financial-risk-assessment-platform-czkhsixzenyn.streamlit.app/

(Hosted on Streamlit Cloud — production-ready, fast, and accessible across devices)


---









