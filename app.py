# app.py
# EMIPredict AI - Streamlit Multi-page Application
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import mlflow
from mlflow.tracking import MlflowClient
import os
import warnings
import sys
from pathlib import Path
warnings.filterwarnings("ignore")

# Add Notebook directory to path for imports
sys.path.append(str(Path(__file__).parent / "Notebook"))

try:
    from Notebook.classification_feature_engineering import ClassificationFeatureEngineer
    from Notebook.regression_feature_engineering import RegressionFeatureEngineer
except ImportError as e:
    st.error(f"Failed to import feature engineering modules: {e}")
    st.info("Make sure classification_feature_engineering.py and regression_feature_engineering.py are in the Notebook folder")

# -------------------------
# CONFIG
# -------------------------
MLFLOW_TRACKING_URI = "http://13.204.193.251:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

CLASSIFICATION_MODEL_NAME = "EMI_Classification_XGBoost"
REGRESSION_MODEL_NAME = "EMI_Regression_XGBoost"  # Updated name from notebook
DATA_PATH = "data/clean_emi_data.csv"  # Updated to local path
ARTIFACT_CACHE_DIR = "mlflow_artifacts"
os.makedirs(ARTIFACT_CACHE_DIR, exist_ok=True)

# -------------------------
# UTIL: Get model info
# -------------------------
def get_latest_model_version(model_name):
    """Get latest version of model (Production or None stage)"""
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        
        # Try Production stage first
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            return versions[0]
        
        # Try None stage (latest registered)
        versions = client.get_latest_versions(model_name, stages=["None"])
        if versions:
            return versions[0]
        
        # Try getting all versions and pick latest
        all_versions = client.search_model_versions(f"name='{model_name}'")
        if all_versions:
            return max(all_versions, key=lambda x: int(x.version))
        
        return None
    except Exception as e:
        st.error(f"Error getting model version for {model_name}: {e}")
        return None

# -------------------------
# UTIL: Download artifacts
# -------------------------
def download_artifact_safe(run_id, artifact_path):
    """Safely download artifact from MLflow"""
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        local_path = client.download_artifacts(run_id, artifact_path, dst_path=ARTIFACT_CACHE_DIR)
        return local_path
    except Exception as e:
        return None

# -------------------------
# Load models from MLflow
# -------------------------
@st.cache_resource(show_spinner=True)
def load_models_from_mlflow():
    """Load classification and regression models from MLflow"""
    result = {
        "clf_model": None,
        "reg_model": None,
        "label_encoder": None,
        "clf_meta": None,
        "reg_meta": None,
        "clf_metrics": {},
        "reg_metrics": {},
        "clf_artifacts": {},
        "ok": False,
        "message": ""
    }
    
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        
        # --- Load Classification Model ---
        clf_version = get_latest_model_version(CLASSIFICATION_MODEL_NAME)
        if clf_version is None:
            result["message"] = f"❌ No version found for {CLASSIFICATION_MODEL_NAME}"
            return result
        
        clf_model_uri = f"models:/{CLASSIFICATION_MODEL_NAME}/{clf_version.version}"
        
        try:
            clf_model = mlflow.sklearn.load_model(clf_model_uri)
            result["clf_model"] = clf_model
        except Exception as e:
            st.warning(f"Failed to load sklearn model, trying pyfunc: {e}")
            clf_model = mlflow.pyfunc.load_model(clf_model_uri)
            result["clf_model"] = clf_model
        
        # Get classification run metrics
        try:
            run = client.get_run(clf_version.run_id)
            result["clf_metrics"] = dict(run.data.metrics)
        except Exception:
            pass
        
        result["clf_meta"] = {
            "model_name": CLASSIFICATION_MODEL_NAME,
            "version": clf_version.version,
            "run_id": clf_version.run_id,
            "stage": clf_version.current_stage
        }
        
        # Try to download label encoder
        try:
            artifacts = client.list_artifacts(clf_version.run_id)
            for artifact in artifacts:
                if "label_encoder" in artifact.path.lower() and artifact.path.endswith(".pkl"):
                    local_path = download_artifact_safe(clf_version.run_id, artifact.path)
                    if local_path:
                        result["label_encoder"] = joblib.load(local_path)
                        break
        except Exception:
            pass
        
        # Download confusion matrix and other artifacts
        try:
            artifacts = client.list_artifacts(clf_version.run_id)
            for artifact in artifacts:
                if any(keyword in artifact.path.lower() for keyword in ['confusion', 'cm_', 'final_confusion']):
                    if artifact.path.endswith('.png'):
                        local_path = download_artifact_safe(clf_version.run_id, artifact.path)
                        if local_path:
                            result["clf_artifacts"][artifact.path] = local_path
        except Exception:
            pass
        
        # --- Load Regression Model ---
        reg_version = get_latest_model_version(REGRESSION_MODEL_NAME)
        if reg_version:
            reg_model_uri = f"models:/{REGRESSION_MODEL_NAME}/{reg_version.version}"
            try:
                reg_model = mlflow.sklearn.load_model(reg_model_uri)
                result["reg_model"] = reg_model
            except Exception:
                try:
                    reg_model = mlflow.pyfunc.load_model(reg_model_uri)
                    result["reg_model"] = reg_model
                except Exception as e:
                    st.warning(f"Failed to load regression model: {e}")
            
            # Get regression metrics
            try:
                run = client.get_run(reg_version.run_id)
                result["reg_metrics"] = dict(run.data.metrics)
            except Exception:
                pass
            
            result["reg_meta"] = {
                "model_name": REGRESSION_MODEL_NAME,
                "version": reg_version.version,
                "run_id": reg_version.run_id,
                "stage": reg_version.current_stage
            }
        
        result["ok"] = True
        result["message"] = "✅ Models loaded successfully"
        return result
        
    except Exception as e:
        result["message"] = f"❌ Error loading models: {str(e)}"
        return result

# -------------------------
# Load dataset
# -------------------------
@st.cache_data(show_spinner=True)
def load_data(path=DATA_PATH):
    """Load dataset from local file"""
    try:
        if not os.path.exists(path):
            return None, False, f"❌ File not found: {path}"
        
        df = pd.read_csv(path)
        
        # Validate required columns
        required_cols = ['age', 'gender', 'monthly_salary', 'emi_eligibility']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return None, False, f"❌ Missing columns: {missing}"
        
        return df, True, f"✅ Loaded {len(df):,} records"
    except Exception as e:
        return None, False, f"❌ Error loading data: {str(e)}"

# -------------------------
# Initialize app
# -------------------------
st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load resources
with st.spinner("Loading models and data..."):
    mlflow_data = load_models_from_mlflow()
    df, data_ok, data_msg = load_data()

clf_model = mlflow_data.get("clf_model")
reg_model = mlflow_data.get("reg_model")
label_encoder = mlflow_data.get("label_encoder")
clf_meta = mlflow_data.get("clf_meta")
reg_meta = mlflow_data.get("reg_meta")
clf_metrics = mlflow_data.get("clf_metrics")
reg_metrics = mlflow_data.get("reg_metrics")
clf_artifacts = mlflow_data.get("clf_artifacts")
models_ok = mlflow_data.get("ok")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("💰 EMIPredict AI")
st.sidebar.markdown("---")

# Status indicators
if models_ok:
    st.sidebar.success("✅ Models Connected")
else:
    st.sidebar.error("❌ Models Not Loaded")
    st.sidebar.caption(mlflow_data.get("message"))

if data_ok:
    st.sidebar.success("✅ Data Loaded")
else:
    st.sidebar.error("❌ Data Not Loaded")
    st.sidebar.caption(data_msg)

st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔮 Predictions", "📊 Data Explorer", "📈 Model Performance", "⚙️ Data Management"]
)

st.sidebar.markdown("---")

# Dataset info
if data_ok:
    st.sidebar.metric("Total Records", f"{len(df):,}")
    st.sidebar.metric("Features", len(df.columns))
    
    if 'emi_eligibility' in df.columns:
        st.sidebar.markdown("**Target Distribution**")
        for label, count in df['emi_eligibility'].value_counts().items():
            st.sidebar.caption(f"{label}: {count:,} ({count/len(df)*100:.1f}%)")

st.sidebar.markdown("---")
st.sidebar.markdown("### MLflow Server")
st.sidebar.caption(f"🔗 {MLFLOW_TRACKING_URI}")

# -------------------------
# PAGE: HOME
# -------------------------
if page == "🏠 Home":
    st.title("💰 EMIPredict AI")
    st.subheader("Intelligent Financial Risk Assessment Platform")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 What We Offer
        - **EMI Eligibility Classification**: Predict if customer is Eligible, Not Eligible, or High Risk
        - **Maximum EMI Prediction**: Estimate maximum affordable EMI amount
        - **Interactive Data Explorer**: Visualize and analyze financial data
        - **Model Performance Dashboard**: Track model metrics via MLflow
        - **Data Management**: CRUD operations on dataset
        """)
        
        st.markdown("### 🚀 Technology Stack")
        st.markdown("""
        - **ML Framework**: Scikit-learn, XGBoost
        - **MLOps**: MLflow (Model Registry & Tracking)
        - **Web Framework**: Streamlit
        - **Visualization**: Plotly
        - **Feature Engineering**: Custom pipelines
        """)
    
    with col2:
        st.markdown("### 📊 System Status")
        
        # Model status cards
        if models_ok:
            st.success("✅ **Models Loaded Successfully**")
            
            if clf_meta:
                st.info(f"""
                **Classification Model**
                - Name: {clf_meta['model_name']}
                - Version: {clf_meta['version']}
                - Stage: {clf_meta['stage']}
                """)
            
            if reg_meta:
                st.info(f"""
                **Regression Model**
                - Name: {reg_meta['model_name']}
                - Version: {reg_meta['version']}
                - Stage: {reg_meta['stage']}
                """)
        else:
            st.error(mlflow_data.get("message"))
            st.markdown("""
            **Troubleshooting Steps:**
            1. Check MLflow server is running: `mlflow server --host 0.0.0.0 --port 5000`
            2. Verify models are registered in MLflow UI
            3. Check model names match: `{CLASSIFICATION_MODEL_NAME}` and `{REGRESSION_MODEL_NAME}`
            """)
        
        # Data status
        if data_ok:
            st.success(data_msg)
        else:
            st.error(data_msg)
            st.markdown(f"""
            **Expected path**: `{DATA_PATH}`
            
            Make sure the dataset exists at this location.
            """)
    
    st.markdown("---")
    
    # Metrics overview
    if models_ok and data_ok:
        st.markdown("### 📈 Quick Stats")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Features", len(df.columns))
        
        if clf_metrics:
            col3.metric("Classification F1", f"{clf_metrics.get('test_f1', 0):.4f}")
        
        if reg_metrics:
            col4.metric("Regression R²", f"{reg_metrics.get('test_r2', 0):.4f}")

# -------------------------
# PAGE: PREDICTIONS
# -------------------------
elif page == "🔮 Predictions":
    st.title("🔮 Real-Time EMI Predictions")
    
    if not models_ok:
        st.error("❌ Models not loaded. Please check MLflow connection.")
        st.info(mlflow_data.get("message"))
        st.stop()
    
    # Prediction type selection
    pred_type = st.radio(
        "Select Prediction Type",
        ["Classification (EMI Eligibility)", "Regression (Maximum EMI Amount)"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # Input form
    with st.form("prediction_form"):
        st.markdown("### 📝 Customer Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**👤 Personal Details**")
            age = st.number_input("Age", 18, 80, 30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
            education = st.selectbox("Education", ["High School", "Graduate", "Professional", "Post Graduate"])
            family_size = st.number_input("Family Size", 1, 10, 3)
            dependents = st.number_input("Dependents", 0, 10, 1)
        
        with col2:
            st.markdown("**💼 Employment Details**")
            monthly_salary = st.number_input("Monthly Salary (₹)", 5000, 500000, 50000, step=1000)
            employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
            years_of_employment = st.number_input("Years of Employment", 0, 50, 5)
            company_type = st.selectbox("Company Type", ["MNC", "Large", "Medium", "Small", "Startup"])
            house_type = st.selectbox("House Type", ["Own", "Rented", "Family"])
            monthly_rent = st.number_input("Monthly Rent (₹)", 0, 100000, 0)
        
        with col3:
            st.markdown("**💰 Financial Details**")
            school_fees = st.number_input("School Fees (₹)", 0, 100000, 0)
            college_fees = st.number_input("College Fees (₹)", 0, 200000, 0)
            travel_expenses = st.number_input("Travel Expenses (₹)", 0, 50000, 3000)
            groceries_utilities = st.number_input("Groceries & Utilities (₹)", 0, 50000, 8000)
            other_monthly_expenses = st.number_input("Other Expenses (₹)", 0, 50000, 2000)
            existing_loans = st.selectbox("Existing Loans", ["No", "Yes"])
            current_emi_amount = st.number_input("Current EMI (₹)", 0, 100000, 0)
            credit_score = st.number_input("Credit Score", 300, 850, 700)
            bank_balance = st.number_input("Bank Balance (₹)", 0, 2000000, 50000)
            emergency_fund = st.number_input("Emergency Fund (₹)", 0, 500000, 10000)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if data_ok and 'emi_scenario' in df.columns:
                emi_scenario = st.selectbox("EMI Scenario", df['emi_scenario'].unique())
            else:
                emi_scenario = st.selectbox("EMI Scenario", 
                    ["E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI", 
                     "Personal Loan EMI", "Education EMI"])
        
        with col2:
            requested_amount = st.number_input("Requested Loan Amount (₹)", 10000, 5000000, 100000, step=10000)
            requested_tenure = st.number_input("Requested Tenure (months)", 3, 120, 12)
        
        submitted = st.form_submit_button("🚀 Generate Prediction", type="primary", use_container_width=True)
    
    if submitted:
        # Prepare input
        input_data = {
            'age': age,
            'gender': gender,
            'marital_status': marital_status,
            'education': education,
            'monthly_salary': monthly_salary,
            'employment_type': employment_type,
            'years_of_employment': years_of_employment,
            'company_type': company_type,
            'house_type': house_type,
            'monthly_rent': monthly_rent,
            'family_size': family_size,
            'dependents': dependents,
            'school_fees': school_fees,
            'college_fees': college_fees,
            'travel_expenses': travel_expenses,
            'groceries_utilities': groceries_utilities,
            'other_monthly_expenses': other_monthly_expenses,
            'existing_loans': existing_loans,
            'current_emi_amount': current_emi_amount,
            'credit_score': credit_score,
            'bank_balance': bank_balance,
            'emergency_fund': emergency_fund,
            'emi_scenario': emi_scenario,
            'requested_amount': requested_amount,
            'requested_tenure': requested_tenure
        }
        
        input_df = pd.DataFrame([input_data])
        
        try:
            if pred_type == "Classification (EMI Eligibility)":
                # Classification prediction
                with st.spinner("Predicting EMI eligibility..."):
                    prediction = clf_model.predict(input_df)[0]
                    
                    # Get probability if available
                    try:
                        probabilities = clf_model.predict_proba(input_df)[0]
                    except:
                        probabilities = None
                    
                    # Decode label
                    if label_encoder is not None:
                        try:
                            predicted_label = label_encoder.inverse_transform([int(prediction)])[0]
                        except:
                            predicted_label = str(prediction)
                    else:
                        # Assume encoding: 0=Eligible, 1=High_Risk, 2=Not_Eligible
                        label_map = {0: "Eligible", 1: "High_Risk", 2: "Not_Eligible"}
                        predicted_label = label_map.get(int(prediction), str(prediction))
                
                st.success("✅ Prediction Complete!")
                
                # Display result
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if predicted_label == "Eligible":
                        st.success(f"### ✅ {predicted_label}")
                    elif predicted_label == "High_Risk":
                        st.warning(f"### ⚠️ {predicted_label}")
                    else:
                        st.error(f"### ❌ {predicted_label}")
                
                with col2:
                    if probabilities is not None:
                        confidence = float(max(probabilities)) * 100
                        st.metric("Confidence", f"{confidence:.1f}%")
                
                with col3:
                    st.metric("Input Credit Score", credit_score)
                
                # Probability distribution
                if probabilities is not None:
                    st.markdown("### 📊 Prediction Probabilities")
                    
                    if label_encoder is not None:
                        labels = label_encoder.classes_
                    else:
                        labels = ["Eligible", "High_Risk", "Not_Eligible"]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=labels,
                            y=probabilities,
                            text=[f"{p*100:.1f}%" for p in probabilities],
                            textposition='auto',
                        )
                    ])
                    fig.update_layout(
                        title="Class Probabilities",
                        xaxis_title="EMI Eligibility Class",
                        yaxis_title="Probability",
                        yaxis=dict(range=[0, 1])
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recommendation
                st.markdown("### 💡 Recommendation")
                if predicted_label == "Eligible":
                    st.info("""
                    ✅ **Customer is eligible for EMI**
                    - Good credit profile
                    - Proceed with loan processing
                    - Offer competitive interest rates
                    """)
                elif predicted_label == "High_Risk":
                    st.warning("""
                    ⚠️ **Customer is high risk**
                    - Consider higher interest rate
                    - Request additional collateral
                    - Reduce loan amount or tenure
                    """)
                else:
                    st.error("""
                    ❌ **Customer is not eligible**
                    - Insufficient income or poor credit
                    - Suggest improving credit score
                    - Review after 6 months
                    """)
            
            else:  # Regression
                if reg_model is None:
                    st.error("❌ Regression model not available")
                    st.stop()
                
                # Regression prediction
                with st.spinner("Predicting maximum EMI..."):
                    predicted_emi = float(reg_model.predict(input_df)[0])
                
                st.success("✅ Prediction Complete!")
                
                # Display results
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Predicted Max EMI", f"₹{predicted_emi:,.0f}")
                col2.metric("Total Loan Amount", f"₹{predicted_emi * requested_tenure:,.0f}")
                col3.metric("% of Monthly Salary", f"{(predicted_emi/monthly_salary)*100:.1f}%")
                
                total_expenses = (monthly_rent + school_fees + college_fees + 
                                travel_expenses + groceries_utilities + 
                                other_monthly_expenses + current_emi_amount)
                remaining = monthly_salary - total_expenses
                col4.metric("Remaining Income", f"₹{remaining:,.0f}")
                
                # Financial breakdown
                st.markdown("### 💰 Financial Breakdown")
                
                fig = go.Figure()
                
                categories = ['Current Expenses', 'Predicted Max EMI', 'Remaining Balance']
                values = [total_expenses, predicted_emi, max(0, remaining - predicted_emi)]
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                
                fig.add_trace(go.Bar(
                    x=categories,
                    y=values,
                    text=[f"₹{v:,.0f}" for v in values],
                    textposition='auto',
                    marker_color=colors
                ))
                
                fig.update_layout(
                    title="Monthly Budget Analysis",
                    xaxis_title="Category",
                    yaxis_title="Amount (₹)",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Affordability check
                st.markdown("### 📋 Affordability Analysis")
                
                if predicted_emi <= remaining * 0.4:
                    st.success(f"""
                    ✅ **Highly Affordable**
                    - Predicted EMI is only {(predicted_emi/monthly_salary)*100:.1f}% of monthly salary
                    - Customer has comfortable buffer
                    - Low default risk
                    """)
                elif predicted_emi <= remaining * 0.6:
                    st.info(f"""
                    ℹ️ **Moderately Affordable**
                    - EMI is {(predicted_emi/monthly_salary)*100:.1f}% of monthly salary
                    - Within acceptable range
                    - Monitor repayment closely
                    """)
                else:
                    st.warning(f"""
                    ⚠️ **Stretching Budget**
                    - EMI is {(predicted_emi/monthly_salary)*100:.1f}% of monthly salary
                    - High financial burden
                    - Consider reducing loan amount or extending tenure
                    """)
        
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.exception(e)

# -------------------------
# PAGE: DATA EXPLORER
# -------------------------
elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")
    
    if not data_ok:
        st.error("❌ Dataset not loaded")
        st.info(data_msg)
        st.stop()
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Dataset Overview", "📈 Visualizations", "📊 Statistics", "🔍 Custom Analysis"])
    
    with tab1:
        st.subheader("Dataset Preview")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            show_rows = st.slider("Rows to display", 10, 1000, 100)
        with col2:
            if 'emi_eligibility' in df.columns:
                filter_eligibility = st.multiselect("Filter by Eligibility", df['emi_eligibility'].unique())
        with col3:
            if 'emi_scenario' in df.columns:
                filter_scenario = st.multiselect("Filter by Scenario", df['emi_scenario'].unique())
        
        # Apply filters
        filtered_df = df.copy()
        if filter_eligibility:
            filtered_df = filtered_df[filtered_df['emi_eligibility'].isin(filter_eligibility)]
        if filter_scenario:
            filtered_df = filtered_df[filtered_df['emi_scenario'].isin(filter_scenario)]
        
        st.dataframe(filtered_df.head(show_rows), use_container_width=True)
        
        # Dataset info
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(filtered_df):,}")
        col2.metric("Total Features", len(filtered_df.columns))
        col3.metric("Numeric Features", len(filtered_df.select_dtypes(include=[np.number]).columns))
        col4.metric("Missing Values", int(filtered_df.isnull().sum().sum()))
    
    with tab2:
        st.subheader("Data Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # EMI Eligibility distribution
            if 'emi_eligibility' in df.columns:
                fig = px.pie(
                    df,
                    names='emi_eligibility',
                    title='EMI Eligibility Distribution',
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Monthly salary distribution
            if 'monthly_salary' in df.columns:
                fig = px.histogram(
                    df,
                    x='monthly_salary',
                    nbins=50,
                    title='Monthly Salary Distribution',
                    labels={'monthly_salary': 'Monthly Salary (₹)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Credit score distribution
            if 'credit_score' in df.columns:
                fig = px.box(
                    df,
                    y='credit_score',
                    x='emi_eligibility' if 'emi_eligibility' in df.columns else None,
                    title='Credit Score Distribution by Eligibility',
                    labels={'credit_score': 'Credit Score'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # EMI scenario distribution
            if 'emi_scenario' in df.columns:
                scenario_counts = df['emi_scenario'].value_counts()
                fig = px.bar(
                    x=scenario_counts.index,
                    y=scenario_counts.values,
                    title='EMI Scenario Distribution',
                    labels={'x': 'Scenario', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Max EMI distribution
            if 'max_monthly_emi' in df.columns:
                fig = px.box(
                    df,
                    y='max_monthly_emi',
                    x='emi_eligibility' if 'emi_eligibility' in df.columns else None,
                    title='Max Monthly EMI Distribution',
                    labels={'max_monthly_emi': 'Max Monthly EMI (₹)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Age distribution
            if 'age' in df.columns:
                fig = px.histogram(
                    df,
                    x='age',
                    nbins=30,
                    title='Age Distribution',
                    labels={'age': 'Age (years)'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            # Limit to top correlations for readability
            correlation_features = st.multiselect(
                "Select features for correlation",
                numeric_cols,
                default=numeric_cols[:10] if len(numeric_cols) > 10 else numeric_cols
            )
            
            if len(correlation_features) > 1:
                corr_matrix = df[correlation_features].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    aspect="auto",
                    title="Correlation Heatmap",
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Statistical Summary")
        
        # Descriptive statistics
        st.markdown("#### Numeric Features")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.markdown("#### Categorical Features")
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            cat_summary = pd.DataFrame({
                'Feature': cat_cols,
                'Unique Values': [df[col].nunique() for col in cat_cols],
                'Most Common': [df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A' for col in cat_cols],
                'Most Common Count': [df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0 for col in cat_cols]
            })
            st.dataframe(cat_summary, use_container_width=True)
        
        st.markdown("#### Missing Values")
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) > 0:
            missing_df = pd.DataFrame({
                'Feature': missing.index,
                'Missing Count': missing.values,
                'Percentage': (missing.values / len(df) * 100).round(2)
            })
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("✅ No missing values in dataset!")
        
        st.markdown("#### Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values,
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)
    
    with tab4:
        st.subheader("Custom Analysis")
        
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Scatter Plot", "Box Plot", "Violin Plot", "Joint Distribution"]
        )
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_feature = st.selectbox("X-axis", numeric_cols)
        with col2:
            y_feature = st.selectbox("Y-axis", [col for col in numeric_cols if col != x_feature])
        
        color_feature = None
        if 'emi_eligibility' in df.columns:
            use_color = st.checkbox("Color by EMI Eligibility")
            if use_color:
                color_feature = 'emi_eligibility'
        
        if analysis_type == "Scatter Plot":
            fig = px.scatter(
                df,
                x=x_feature,
                y=y_feature,
                color=color_feature,
                title=f"{x_feature} vs {y_feature}",
                opacity=0.6
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Box Plot":
            fig = px.box(
                df,
                x=color_feature if color_feature else None,
                y=y_feature,
                title=f"{y_feature} Distribution",
                color=color_feature
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Violin Plot":
            fig = px.violin(
                df,
                x=color_feature if color_feature else None,
                y=y_feature,
                title=f"{y_feature} Distribution",
                color=color_feature,
                box=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Joint Distribution":
            fig = px.density_contour(
                df,
                x=x_feature,
                y=y_feature,
                title=f"Joint Distribution: {x_feature} vs {y_feature}",
                marginal_x="histogram",
                marginal_y="histogram"
            )
            st.plotly_chart(fig, use_container_width=True)

# -------------------------
# PAGE: MODEL PERFORMANCE
# -------------------------
elif page == "📈 Model Performance":
    st.title("📈 Model Performance Dashboard")
    
    if not models_ok:
        st.error("❌ Models not loaded")
        st.info(mlflow_data.get("message"))
        st.stop()
    
    tab1, tab2, tab3 = st.tabs(["🎯 Classification", "📊 Regression", "🔗 MLflow Integration"])
    
    with tab1:
        st.subheader("Classification Model Performance")
        
        if clf_model and clf_meta:
            st.success("✅ Classification model loaded")
            
            # Model info
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Model Name", clf_meta['model_name'])
            col2.metric("Version", clf_meta['version'])
            col3.metric("Stage", clf_meta['stage'])
            col4.metric("Run ID", clf_meta['run_id'][:8] + "...")
            
            st.markdown("---")
            
            # Performance metrics
            if clf_metrics:
                st.markdown("### 📊 Performance Metrics")
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                metric_col1.metric(
                    "Accuracy",
                    f"{clf_metrics.get('test_accuracy', 0):.4f}",
                    help="Overall prediction accuracy"
                )
                metric_col2.metric(
                    "F1 Score",
                    f"{clf_metrics.get('test_f1', 0):.4f}",
                    help="Weighted F1 score across all classes"
                )
                metric_col3.metric(
                    "Precision",
                    f"{clf_metrics.get('test_precision', 0):.4f}",
                    help="Weighted precision"
                )
                metric_col4.metric(
                    "Recall",
                    f"{clf_metrics.get('test_recall', 0):.4f}",
                    help="Weighted recall"
                )
                
                # Additional metrics
                st.markdown("#### Additional Metrics")
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'test_roc_auc' in clf_metrics:
                        st.metric("ROC-AUC", f"{clf_metrics['test_roc_auc']:.4f}")
                    if 'cv_accuracy_mean' in clf_metrics:
                        st.metric("CV Accuracy (Mean)", f"{clf_metrics['cv_accuracy_mean']:.4f}")
                
                with col2:
                    if 'cv_f1_mean' in clf_metrics:
                        st.metric("CV F1 (Mean)", f"{clf_metrics['cv_f1_mean']:.4f}")
                    if 'cv_roc_auc_mean' in clf_metrics:
                        st.metric("CV ROC-AUC (Mean)", f"{clf_metrics['cv_roc_auc_mean']:.4f}")
            
            # Display artifacts
            if clf_artifacts:
                st.markdown("### 📁 Model Artifacts")
                
                for artifact_name, artifact_path in clf_artifacts.items():
                    st.markdown(f"#### {artifact_name}")
                    
                    if artifact_path.endswith('.png'):
                        try:
                            st.image(artifact_path, use_column_width=True)
                        except:
                            st.warning(f"Could not load image: {artifact_path}")
                    elif artifact_path.endswith('.csv'):
                        try:
                            artifact_df = pd.read_csv(artifact_path)
                            st.dataframe(artifact_df, use_container_width=True)
                        except:
                            st.warning(f"Could not load CSV: {artifact_path}")
            else:
                st.info("No artifacts found. Confusion matrix and other visualizations may be available in MLflow UI.")
        else:
            st.error("Classification model not available")
    
    with tab2:
        st.subheader("Regression Model Performance")
        
        if reg_model and reg_meta:
            st.success("✅ Regression model loaded")
            
            # Model info
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Model Name", reg_meta['model_name'])
            col2.metric("Version", reg_meta['version'])
            col3.metric("Stage", reg_meta['stage'])
            col4.metric("Run ID", reg_meta['run_id'][:8] + "...")
            
            st.markdown("---")
            
            # Performance metrics
            if reg_metrics:
                st.markdown("### 📊 Performance Metrics")
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                metric_col1.metric(
                    "R² Score",
                    f"{reg_metrics.get('test_r2', 0):.4f}",
                    help="Coefficient of determination"
                )
                metric_col2.metric(
                    "RMSE",
                    f"₹{reg_metrics.get('test_rmse', 0):,.2f}",
                    help="Root Mean Squared Error"
                )
                metric_col3.metric(
                    "MAE",
                    f"₹{reg_metrics.get('test_mae', 0):,.2f}",
                    help="Mean Absolute Error"
                )
                metric_col4.metric(
                    "MAPE",
                    f"{reg_metrics.get('test_mape', 0):.2f}%",
                    help="Mean Absolute Percentage Error"
                )
                
                # CV metrics
                st.markdown("#### Cross-Validation Metrics")
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'cv_rmse_mean' in reg_metrics:
                        st.metric("CV RMSE (Mean)", f"₹{reg_metrics['cv_rmse_mean']:,.2f}")
                    if 'cv_mae_mean' in reg_metrics:
                        st.metric("CV MAE (Mean)", f"₹{reg_metrics['cv_mae_mean']:,.2f}")
                
                with col2:
                    if 'cv_r2_mean' in reg_metrics:
                        st.metric("CV R² (Mean)", f"{reg_metrics['cv_r2_mean']:.4f}")
                    if 'cv_mape_mean' in reg_metrics:
                        st.metric("CV MAPE (Mean)", f"{reg_metrics['cv_mape_mean']:.2f}%")
        else:
            st.info("Regression model not registered or not available")
    
    with tab3:
        st.subheader("🔗 MLflow Integration")
        
        st.markdown(f"""
        ### MLflow Tracking Server
        **URI**: `{MLFLOW_TRACKING_URI}`
        
        Access the MLflow UI to view:
        - Experiment tracking
        - Model registry
        - Artifact storage
        - Metric comparisons
        - Run parameters
        """)
        
        st.markdown("---")
        
        st.markdown("### 📋 Registered Models")
        
        try:
            client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            
            # Classification model versions
            st.markdown("#### Classification Model Versions")
            clf_versions = client.search_model_versions(f"name='{CLASSIFICATION_MODEL_NAME}'")
            
            if clf_versions:
                versions_data = []
                for v in clf_versions[:5]:  # Show last 5 versions
                    versions_data.append({
                        'Version': v.version,
                        'Stage': v.current_stage,
                        'Run ID': v.run_id[:8] + "...",
                        'Status': v.status
                    })
                st.dataframe(pd.DataFrame(versions_data), use_container_width=True)
            else:
                st.warning("No versions found")
            
            # Regression model versions
            st.markdown("#### Regression Model Versions")
            reg_versions = client.search_model_versions(f"name='{REGRESSION_MODEL_NAME}'")
            
            if reg_versions:
                versions_data = []
                for v in reg_versions[:5]:
                    versions_data.append({
                        'Version': v.version,
                        'Stage': v.current_stage,
                        'Run ID': v.run_id[:8] + "...",
                        'Status': v.status
                    })
                st.dataframe(pd.DataFrame(versions_data), use_container_width=True)
            else:
                st.warning("No versions found")
        
        except Exception as e:
            st.error(f"Error fetching MLflow data: {str(e)}")

# -------------------------
# PAGE: DATA MANAGEMENT
# -------------------------
elif page == "⚙️ Data Management":
    st.title("⚙️ Data Management")
    
    if not data_ok:
        st.error("❌ Dataset not loaded")
        st.info(data_msg)
        st.stop()
    
    operation = st.radio(
        "Select Operation",
        ["📖 Read Records", "➕ Create Record", "✏️ Update Record", "🗑️ Delete Record"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if operation == "📖 Read Records":
        st.subheader("📖 View and Search Records")
        
        # Search functionality
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_column = st.selectbox("Search by column", df.columns)
        
        with col2:
            search_value = st.text_input("Search value")
        
        with col3:
            case_sensitive = st.checkbox("Case sensitive", value=False)
        
        if search_value:
            if df[search_column].dtype == 'object':
                if case_sensitive:
                    filtered_df = df[df[search_column].astype(str).str.contains(search_value)]
                else:
                    filtered_df = df[df[search_column].astype(str).str.contains(search_value, case=False)]
            else:
                try:
                    search_numeric = float(search_value)
                    filtered_df = df[df[search_column] == search_numeric]
                except:
                    st.warning("Invalid numeric value")
                    filtered_df = df
            
            st.success(f"✅ Found {len(filtered_df)} matching records")
            st.dataframe(filtered_df, use_container_width=True)
            
            # Export filtered data
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv,
                file_name="filtered_data.csv",
                mime="text/csv"
            )
        else:
            st.dataframe(df, use_container_width=True)
    
    elif operation == "➕ Create Record":
        st.subheader("➕ Add New Record")
        st.warning("⚠️ This is a demo. Changes are not persisted to the original CSV file.")
        
        with st.form("create_record"):
            st.markdown("### Enter Record Details")
            
            col1, col2, col3 = st.columns(3)
            
            new_record = {}
            
            for i, column in enumerate(df.columns):
                col = [col1, col2, col3][i % 3]
                
                with col:
                    if df[column].dtype == 'object':
                        unique_values = df[column].dropna().unique().tolist()
                        if len(unique_values) < 20:
                            new_record[column] = st.selectbox(f"{column}", unique_values)
                        else:
                            new_record[column] = st.text_input(f"{column}")
                    elif df[column].dtype in ['int64', 'float64']:
                        min_val = float(df[column].min())
                        max_val = float(df[column].max())
                        mean_val = float(df[column].mean())
                        new_record[column] = st.number_input(
                            f"{column}",
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val
                        )
            
            submitted = st.form_submit_button("Add Record", type="primary")
            
            if submitted:
                st.success("✅ Record added successfully (demo mode)")
                st.json(new_record)
    
    elif operation == "✏️ Update Record":
        st.subheader("✏️ Update Existing Record")
        st.warning("⚠️ This is a demo. Changes are not persisted to the original CSV file.")
        
        record_index = st.number_input("Select Record Index", 0, len(df)-1, 0)
        
        if record_index < len(df):
            st.markdown("#### Current Record")
            st.dataframe(df.iloc[[record_index]], use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                update_column = st.selectbox("Column to update", df.columns)
            
            with col2:
                if df[update_column].dtype == 'object':
                    new_value = st.text_input("New value", value=str(df.iloc[record_index][update_column]))
                else:
                    new_value = st.number_input(
                        "New value",
                        value=float(df.iloc[record_index][update_column])
                    )
            
            if st.button("Update Record", type="primary"):
                st.success(f"✅ Record {record_index} updated (demo mode)")
                st.info(f"Changed {update_column}: {df.iloc[record_index][update_column]} → {new_value}")
    
    elif operation == "🗑️ Delete Record":
        st.subheader("🗑️ Delete Record")
        st.warning("⚠️ This is a demo. Changes are not persisted to the original CSV file.")
        
        record_index = st.number_input("Select Record Index to Delete", 0, len(df)-1, 0)
        
        if record_index < len(df):
            st.markdown("#### Record to Delete")
            st.dataframe(df.iloc[[record_index]], use_container_width=True)
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("🗑️ Delete", type="primary"):
                    st.success(f"✅ Record {record_index} deleted (demo mode)")
            
            with col2:
                if st.button("❌ Cancel"):
                    st.info("Deletion cancelled")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 💰 EMIPredict AI")
    st.caption("Intelligent Financial Risk Assessment")

with col2:
    st.markdown("### 🔗 Quick Links")
    st.caption(f"[MLflow UI]({MLFLOW_TRACKING_URI})")
    st.caption("[Documentation](#)")

with col3:
    st.markdown("### ℹ️ System Info")
    st.caption(f"App Version: 1.0.0")
    st.caption(f"Models: {2 if reg_model else 1} loaded")
    st.caption(f"Records: {len(df):,}" if data_ok else "No data")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>© 2024 EMIPredict AI | Built with Streamlit & MLflow</div>",
    unsafe_allow_html=True
)