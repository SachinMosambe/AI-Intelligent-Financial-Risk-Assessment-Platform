# 💳 EMIPredict AI – Intelligent Financial Risk Assessment Platform

## 🧾 Overview

**EMIPredict AI** is an intelligent machine learning–based financial risk assessment platform designed to predict **EMI eligibility** (classification) and **maximum safe EMI amount** (regression).
It leverages **MLflow** for experiment tracking and **Streamlit** for real-time predictions, with automated deployment via **AWS EC2**, **S3**, and **GitHub Actions**.

---

## 🧠 Key Highlights

* **Dual ML Problem Solving**

  * Classification: Predict EMI eligibility (`Eligible`, `High_Risk`, `Not_Eligible`)
  * Regression: Predict maximum safe EMI amount
* **MLflow Integration** for model tracking, comparison, and registry
* **Streamlit Web App** for interactive, real-time predictions
* **Automated CI/CD Deployment** via GitHub Actions and AWS infrastructure
* **Comprehensive EDA and Feature Engineering** using 22+ financial and demographic variables

---

## 🏷️ Project Structure

```
ASSIGNMENT_2/
│
├── data_cleaning.ipynb                # Data preprocessing and cleaning pipeline
├── Exploratory_data_analysis.ipynb    # EDA, correlation, and visualization
├── classification_model.ipynb         # EMI eligibility (classification) models
├── Regression_model.ipynb             # EMI amount (regression) models
├── clean_emi_data.csv                 # Cleaned dataset
├── emi_prediction_dataset.csv         # Original dataset
├── requirements.txt                   # Project dependencies
├── output/                            # Model outputs, logs, and results
└── .github/workflows/deploy.yml       # Automated deployment pipeline
```

---

## ⚙️ Tech Stack

| Category         | Tools / Libraries                    |
| ---------------- | ------------------------------------ |
| Programming      | Python                               |
| Data Processing  | Pandas, NumPy                        |
| Visualization    | Matplotlib, Seaborn                  |
| Machine Learning | Scikit-learn, XGBoost, Random Forest |
| Model Tracking   | MLflow                               |
| Web Framework    | Streamlit                            |
| Deployment       | AWS EC2, S3, GitHub Actions          |

---

## 🚀 Deployment Pipeline

### GitHub Actions CI/CD Overview

1. **Trigger:** Push to the `main` branch.
2. **Build:** Python environment setup and dependency installation.
3. **Package:** Codebase zipped and uploaded to AWS S3.
4. **Deploy:** Application auto-deployed to AWS EC2.
5. **Run:** Streamlit starts automatically on EC2 at port `8501`.

**Access App:**
`http://<your-ec2-public-dns>:8501`

---

## 🤩 Machine Learning Approach

### **1. Data Processing & Feature Engineering**

* Cleaned and validated 400K+ financial records
* Derived key ratios (Debt-to-Income, Expense-to-Income, Affordability)
* Handled missing values, categorical encoding, and scaling

### **2. Model Development**

#### Classification (EMI Eligibility)

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier

#### Regression (Max EMI Prediction)

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor

### **3. Model Evaluation Metrics**

| Model Type     | Metrics Used                                   |
| -------------- | ---------------------------------------------- |
| Classification | Accuracy, Precision, Recall, F1-Score, ROC-AUC |
| Regression     | RMSE, MAE, R², MAPE                            |

---

## 🤪 MLflow Integration

* Centralized tracking of all experiment runs
* Parameter, metric, and artifact logging
* Model registry for version control and comparison
* Integrated with Streamlit dashboard for live performance monitoring

---

## 🖥️ Streamlit Application Features

* Multi-page financial risk dashboard
* Dual prediction modes (Eligibility & Maximum EMI)
* Dynamic input forms and real-time output visualization
* Model performance insights integrated from MLflow

---

## 💡 Business Impact

| Stakeholder                 | Value Delivered                                             |
| --------------------------- | ----------------------------------------------------------- |
| **Financial Institutions**  | Automates loan approvals, reduces manual risk checks by 80% |
| **FinTech Platforms**       | Instant EMI eligibility assessment for digital lending apps |
| **Banks & Credit Agencies** | Data-driven recommendations and risk scoring                |
| **Loan Officers**           | AI-powered insights for rapid loan decision-making          |

---

## 🔧 Installation & Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/EMIPredictAI.git
cd EMIPredictAI

# Create and activate virtual environment (optional)
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

# Install dependencies
pip install -r requirements.txt

# Run the application locally
streamlit run app.py
```

---

## 🗱️ Deployment Workflow File (`.github/workflows/deploy.yml`)

```yaml
name: Deploy EMIPredict AI to AWS EC2

on:
  push:
    branches:
      - main

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Package application
        run: zip -r app_package.zip . -x "*.git*" "venv/*" "__pycache__/*"

      - name: Upload to S3
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          AWS_REGION: ap-south-1
          S3_BUCKET: your-s3-bucket-name
        run: |
          aws s3 cp app_package.zip s3://$S3_BUCKET/emipredict-app.zip

      - name: Deploy to EC2
        env:
          AWS_SSH_KEY: ${{ secrets.EC2_SSH_KEY }}
          EC2_HOST: ${{ secrets.EC2_HOST }}
          EC2_USER: ubuntu
        run: |
          echo "$AWS_SSH_KEY" > private_key.pem
          chmod 600 private_key.pem
          ssh -o StrictHostKeyChecking=no -i private_key.pem $EC2_USER@$EC2_HOST "
            sudo rm -rf ~/emipredict-app &&
            aws s3 cp s3://your-s3-bucket-name/emipredict-app.zip ~/ &&
            unzip emipredict-app.zip -d ~/emipredict-app &&
            cd ~/emipredict-app &&
            pip install -r requirements.txt &&
            nohup streamlit run app.py --server.port 8501 > app.log 2>&1 &
          "
```

---

## 🗾 Expected Outcomes

* 90%+ classification accuracy and RMSE < 2000 INR
* End-to-end MLflow tracking with model registry
* Fully automated CI/CD deployment pipeline
* Scalable Streamlit application accessible via cloud

---

## 👨‍💻 Author

**Sachin Dattatraay Mosambe**
*Data Science & FinTech Enthusiast*
📧 [LinkedIn Profile](https://www.linkedin.com/in/sachinmosambe)

---

## 🫋 License

This project is released under the **MIT License** – you’re free to use and modify with attribution.

---

⭐ **If you like this project, give it a star on GitHub!**
