# app_s3_mlflow_fast.py
# EMIPredict AI - API-based lightweight client for Streamlit Cloud
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import mlflow
from mlflow.tracking import MlflowClient
import os
import warnings
import hashlib
from pathlib import Path
from io import BytesIO
import time
import requests

warnings.filterwarnings("ignore")

# -------------------------
# CONFIG
# -------------------------
MLFLOW_TRACKING_URI = "http://13.204.193.251:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

CLASSIFICATION_MODEL_NAME = "EMI_Classification_XGBoost"
REGRESSION_MODEL_NAME = "EMI_Regression_XGBoost"

# -------------------------
# GLOBAL DUMMY RECORD (VALID FULL-SCHEMA health-check)
# -------------------------
# Keep this as a global constant so it's easy to update when schema changes.
DUMMY_RECORD = {
    "age": 25,
    "gender": "Male",
    "marital_status": "Single",
    "education": "Graduate",
    "monthly_salary": 0.0,
    "employment_type": "Private",
    "years_of_employment": 0.0,
    "company_type": "Small",
    "house_type": "Rented",
    "monthly_rent": 0.0,
    "family_size": 1.0,
    "dependents": 0.0,
    "school_fees": 0.0,
    "college_fees": 0.0,
    "travel_expenses": 0.0,
    "groceries_utilities": 0.0,
    "other_monthly_expenses": 0.0,
    "existing_loans": "No",
    "current_emi_amount": 0.0,
    "credit_score": 700.0,
    "bank_balance": 0.0,
    "emergency_fund": 0.0,
    "emi_scenario": "Personal Loan EMI",
    "requested_amount": 10000.0,
    "requested_tenure": 3.0
}

# -------------------------
# SAFE SECRET FETCHING
# -------------------------
def get_secret(key, default):
    try:
        return st.secrets[key]
    except Exception:
        return default

CLASSIFICATION_URL = get_secret("CLASSIFICATION_URL", "http://13.204.193.251:9001/invocations")
REGRESSION_URL     = get_secret("REGRESSION_URL",     "http://13.204.193.251:9002/invocations")

# -------------------------
# DATA LOCATION (S3 public object)
# -------------------------
S3_BUCKET = get_secret("S3_BUCKET", "mlflow-tracking-loan")
S3_KEY    = get_secret("S3_KEY", "data/clean_emi_data.csv")
DATA_PATH = get_secret("DATA_PATH", f"https://{S3_BUCKET}.s3.amazonaws.com/{S3_KEY}")

# Local cache directories
MODEL_CACHE_DIR = "model_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# -------------------------
# OPTIMIZED CACHING FUNCTIONS
# -------------------------
def get_model_hash(model_name, version):
    return hashlib.md5(f"{model_name}_{version}".encode()).hexdigest()

def get_latest_model_version(model_name):
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        for stage in ["Production", "Staging", "None"]:
            versions = client.get_latest_versions(model_name, stages=[stage])
            if versions:
                return versions[0]
        all_versions = client.search_model_versions(f"name='{model_name}'")
        if all_versions:
            return max(all_versions, key=lambda x: int(x.version))
        return None
    except Exception:
        return None

def check_cache_validity(cache_path, model_version):
    if not cache_path.exists():
        return False
    version_file = cache_path.parent / f"{cache_path.stem}_version.txt"
    if version_file.exists():
        with open(version_file, 'r') as f:
            cached_version = f.read().strip()
            return cached_version == str(model_version.version)
    return False

def save_cache_version(cache_path, model_version):
    version_file = cache_path.parent / f"{cache_path.stem}_version.txt"
    with open(version_file, 'w') as f:
        f.write(str(model_version.version))

# -------------------------
# MODEL PROXY (calls MLflow serving endpoints)
# -------------------------
class ModelProxy:
    """
    Reliable proxy that:
     - checks MLflow health via GET /ping (preferred)
     - falls back to sending a valid full-schema dummy record
     - posts predictions as {'inputs': [dicts]}
    """
    def __init__(self, url, timeout=10):
        self.url = url.rstrip('/')
        self.timeout = timeout
        self.ready = False
        self.last_error = None
        self._check_ready()

    def _check_ready(self):
        # Try GET /ping first
        try:
            ping_url = self.url.replace("/invocations", "/ping")
            if not ping_url.endswith("/ping"):
                ping_url = self.url + "/ping"
            r = requests.get(ping_url, timeout=self.timeout)
            if r.status_code == 200:
                self.ready = True
                return
        except Exception:
            # ping may not be available or reachable; fall through to dummy POST
            pass

        # Fallback: valid dummy POST with full schema
        try:
            payload = {"inputs": [DUMMY_RECORD]}
            r2 = requests.post(self.url, json=payload, timeout=self.timeout)
            if r2.status_code == 200:
                self.ready = True
                return
            else:
                self.ready = False
                self.last_error = f"Status {r2.status_code} - {r2.text}"
        except Exception as e:
            self.ready = False
            self.last_error = str(e)

    def predict(self, df_or_records):
        try:
            if isinstance(df_or_records, pd.DataFrame):
                payload = {"inputs": df_or_records.to_dict(orient="records")}
            elif isinstance(df_or_records, dict):
                payload = {"inputs": [df_or_records]}
            elif isinstance(df_or_records, (list, tuple)):
                payload = {"inputs": list(df_or_records)}
            else:
                raise ValueError("Unsupported input format for predict()")

            r = requests.post(self.url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            result = r.json()

            preds = result.get("predictions", None)
            if preds is None:
                if isinstance(result, list):
                    preds = result
                else:
                    return np.array([])
            return np.array(preds)
        except Exception as e:
            self.last_error = str(e)
            raise

    def predict_proba(self, df_or_records):
        preds = self.predict(df_or_records)
        if preds.ndim == 2:
            return preds
        raise ValueError("Server did not return probability vector")

# -------------------------
# LOADING PROXIES (lightweight)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_classification_proxy():
    try:
        proxy = ModelProxy(CLASSIFICATION_URL, timeout=20)
        return proxy
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_regression_proxy():
    try:
        proxy = ModelProxy(REGRESSION_URL, timeout=20)
        return proxy
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_label_encoder_fast():
    try:
        clf_version = get_latest_model_version(CLASSIFICATION_MODEL_NAME)
        if clf_version is None:
            return None
        cache_path = Path(MODEL_CACHE_DIR) / f"label_encoder_v{clf_version.version}.pkl"
        if cache_path.exists():
            return joblib.load(cache_path)
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        try:
            artifacts = client.list_artifacts(clf_version.run_id)
            for artifact in artifacts:
                if "label_encoder" in artifact.path.lower() and artifact.path.endswith(".pkl"):
                    local_path = client.download_artifacts(clf_version.run_id, artifact.path, dst_path=MODEL_CACHE_DIR)
                    encoder = joblib.load(local_path)
                    joblib.dump(encoder, cache_path)
                    return encoder
        except Exception:
            pass
        return None
    except Exception:
        return None

# -------------------------
# METADATA LOADING (INSTANT)
# -------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_model_metadata():
    result = {
        "clf_meta": None,
        "reg_meta": None,
        "clf_metrics": {},
        "reg_metrics": {},
        "ok": False,
        "message": ""
    }
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

        clf_version = get_latest_model_version(CLASSIFICATION_MODEL_NAME)
        if clf_version:
            result["clf_meta"] = {
                "model_name": CLASSIFICATION_MODEL_NAME,
                "version": clf_version.version,
                "run_id": clf_version.run_id,
                "stage": clf_version.current_stage,
            }
            try:
                run = client.get_run(clf_version.run_id)
                result["clf_metrics"] = dict(run.data.metrics)
            except Exception:
                pass

        reg_version = get_latest_model_version(REGRESSION_MODEL_NAME)
        if reg_version:
            result["reg_meta"] = {
                "model_name": REGRESSION_MODEL_NAME,
                "version": reg_version.version,
                "run_id": reg_version.run_id,
                "stage": reg_version.current_stage,
            }
            try:
                run = client.get_run(reg_version.run_id)
                result["reg_metrics"] = dict(run.data.metrics)
            except Exception:
                pass

        result["ok"] = True
        result["message"] = "✅ Connected"
        return result
    except Exception as e:
        result["message"] = f"❌ Failed: {str(e)}"
        return result

# -------------------------
# DATA LOADING
# -------------------------
@st.cache_data(show_spinner=False)
def load_data(path=DATA_PATH, timeout=60):
    try:
        if isinstance(path, str) and path.startswith("http"):
            with requests.get(path, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                buffer = BytesIO()
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        buffer.write(chunk)
                buffer.seek(0)
                df = pd.read_csv(buffer)
        else:
            if not os.path.exists(path):
                return None, False, f"❌ File not found"
            df = pd.read_csv(path)
        return df, True, f"✅ {len(df):,} records"
    except Exception as e:
        return None, False, f"❌ Error: {str(e)}"

# -------------------------
# APP CONFIGURATION
# -------------------------
st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stAlert > div { padding: 0.5rem 1rem; }
    .metric-row { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# PRELOAD EVERYTHING AT STARTUP
# -------------------------
st.sidebar.title("💰 EMIPredict AI")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Initialization")

metadata = load_model_metadata()
if metadata["ok"]:
    st.sidebar.success("✅ MLflow connected")
else:
    st.sidebar.error("❌ Connection failed")

df, data_ok, data_msg = load_data()
if data_ok:
    st.sidebar.success(f"✅ Data loaded")
else:
    st.sidebar.warning("⚠️ No data")

st.sidebar.markdown("### 📦 Connecting to Model Serving Endpoints (no heavy loads)")

clf_model = None
reg_model = None
label_encoder = None

with st.spinner("Connecting to classification endpoint..."):
    try:
        clf_model = load_classification_proxy()
        if clf_model and clf_model.ready:
            st.sidebar.success("✅ Classification endpoint reachable")
        else:
            st.sidebar.warning(f"⚠️ Classification endpoint not ready: {getattr(clf_model,'last_error', 'unknown')}")
    except Exception as e:
        st.sidebar.error(f"❌ Classification proxy failed: {e}")
        clf_model = None

with st.spinner("Connecting to regression endpoint..."):
    try:
        reg_model = load_regression_proxy()
        if reg_model and reg_model.ready:
            st.sidebar.success("✅ Regression endpoint reachable")
        else:
            st.sidebar.warning(f"⚠️ Regression endpoint not ready: {getattr(reg_model,'last_error', 'unknown')}")
    except Exception as e:
        st.sidebar.error(f"❌ Regression proxy failed: {e}")
        reg_model = None

label_encoder = load_label_encoder_fast()

if clf_model and reg_model and clf_model.ready and reg_model.ready:
    st.sidebar.success("🎉 All models ready via API!")
else:
    st.sidebar.warning("⚠️ Some models not reachable. Predictions will attempt API calls and show errors if unavailable.")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📑 Navigation",
    ["🏠 Home", "🔮 Predictions", "📊 Data Explorer", "📈 Model Performance", "🔧 System Info"]
)

st.sidebar.markdown("---")

if data_ok:
    st.sidebar.metric("📊 Records", f"{len(df):,}")
    st.sidebar.metric("📋 Features", len(df.columns))

st.sidebar.markdown("---")
st.sidebar.caption(f"🔗 MLflow: {MLFLOW_TRACKING_URI}")

# -------------------------
# HOME PAGE
# -------------------------
if page == "🏠 Home":
    st.title("💰 EMIPredict AI")
    st.markdown("### Intelligent Financial Risk Assessment")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        ### 🎯 Features
        - **EMI Eligibility Classification**
        - **Maximum EMI Prediction**
        - **Interactive Analytics**
        - **Real-time Predictions**

        ### ⚡ Performance
        - **Models served from EC2 (no heavy loading in Streamlit)**
        - **Instant predictions (no wait)**
        - **Smart proxy-based architecture**
        - **S3-backed storage**
        """)
    with col2:
        st.markdown("### 📊 System Status")
        if clf_model and clf_model.ready:
            st.success("✅ **Classification Endpoint Ready**")
            if metadata.get("clf_meta"):
                st.caption(f"Model: {metadata['clf_meta']['model_name']} | v{metadata['clf_meta']['version']}")
        else:
            st.error("❌ Classification endpoint not ready")
        if reg_model and reg_model.ready:
            st.success("✅ **Regression Endpoint Ready**")
            if metadata.get("reg_meta"):
                st.caption(f"Model: {metadata['reg_meta']['model_name']} | v{metadata['reg_meta']['version']}")
        else:
            st.error("❌ Regression endpoint not ready")
        if data_ok:
            st.success(f"✅ **{data_msg}**")

# -------------------------
# PREDICTIONS PAGE (API-based)
# -------------------------
elif page == "🔮 Predictions":
    st.title("🔮 Real-Time EMI Predictions")

    if not (clf_model and reg_model):
        st.error("❌ Model endpoints not configured. Please ensure MLflow serving is running on EC2.")
        st.stop()

    st.success("⚡ Models proxied - predictions will use remote serving endpoints!")

    pred_type = st.radio(
        "Select Prediction Type",
        ["Classification (EMI Eligibility)", "Regression (Maximum EMI Amount)"],
        horizontal=True
    )
    st.markdown("---")

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
            groceries_utilities = st.number_input("Groceries (₹)", 0, 50000, 8000)
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
                emi_scenario = st.selectbox("EMI Scenario", ["E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI", "Personal Loan EMI", "Education EMI"])
        with col2:
            requested_amount = st.number_input("Requested Loan Amount (₹)", 10000, 5000000, 100000, step=10000)
            requested_tenure = st.number_input("Requested Tenure (months)", 3, 120, 12)

        submitted = st.form_submit_button("⚡ Generate Prediction (Instant!)", type="primary", use_container_width=True)

    if submitted:
        input_data = {
            'age': age, 'gender': gender, 'marital_status': marital_status,
            'education': education, 'monthly_salary': monthly_salary,
            'employment_type': employment_type, 'years_of_employment': years_of_employment,
            'company_type': company_type, 'house_type': house_type, 'monthly_rent': monthly_rent,
            'family_size': family_size, 'dependents': dependents, 'school_fees': school_fees,
            'college_fees': college_fees, 'travel_expenses': travel_expenses, 'groceries_utilities': groceries_utilities,
            'other_monthly_expenses': other_monthly_expenses, 'existing_loans': existing_loans,
            'current_emi_amount': current_emi_amount, 'credit_score': credit_score, 'bank_balance': bank_balance,
            'emergency_fund': emergency_fund, 'emi_scenario': emi_scenario, 'requested_amount': requested_amount,
            'requested_tenure': requested_tenure
        }
        input_df = pd.DataFrame([input_data])

        try:
            if pred_type == "Classification (EMI Eligibility)":
                start_time = time.time()
                try:
                    prediction_arr = clf_model.predict(input_df)
                except Exception as e:
                    st.error(f"❌ Classification API failed: {e}")
                    st.stop()
                if len(prediction_arr) > 0:
                    prediction = prediction_arr[0]
                else:
                    st.error("❌ Classification returned no prediction")
                    st.stop()
                probabilities = None
                try:
                    proba = clf_model.predict_proba(input_df)
                    if hasattr(proba, "tolist"):
                        probabilities = np.array(proba).flatten() if np.array(proba).ndim == 1 else np.array(proba)[0]
                    else:
                        probabilities = np.array(proba)
                except Exception:
                    probabilities = None
                inference_time = (time.time() - start_time) * 1000
                if label_encoder:
                    try:
                        predicted_label = label_encoder.inverse_transform([int(prediction)])[0]
                    except Exception:
                        predicted_label = str(prediction)
                else:
                    label_map = {0: "Eligible", 1: "High_Risk", 2: "Not_Eligible"}
                    try:
                        predicted_label = label_map.get(int(prediction), str(prediction))
                    except Exception:
                        predicted_label = str(prediction)
                st.success(f"✅ Prediction Complete in {inference_time:.0f}ms!")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if predicted_label == "Eligible":
                        st.success(f"### ✅ {predicted_label}")
                    elif predicted_label == "High_Risk":
                        st.warning(f"### ⚠️ {predicted_label}")
                    else:
                        st.error(f"### ❌ {predicted_label}")
                with col2:
                    if probabilities is not None and len(probabilities) > 0:
                        confidence = float(max(probabilities)) * 100
                        st.metric("Confidence", f"{confidence:.1f}%")
                with col3:
                    st.metric("Inference Time", f"{inference_time:.0f}ms")
                if probabilities is not None and len(probabilities) > 0:
                    st.markdown("### 📊 Prediction Probabilities")
                    if label_encoder and hasattr(label_encoder, "classes_"):
                        labels = label_encoder.classes_
                    else:
                        labels = ["Eligible", "High_Risk", "Not_Eligible"]
                    try:
                        probabilities = np.array(probabilities).ravel()[:len(labels)]
                    except Exception:
                        pass
                    fig = go.Figure(data=[
                        go.Bar(
                            x=labels,
                            y=probabilities,
                            text=[f"{p*100:.1f}%" for p in probabilities],
                            textposition='auto',
                            marker_color=['#28a745' if l == predicted_label else '#6c757d' for l in labels]
                        )
                    ])
                    fig.update_layout(title="Class Probabilities", yaxis=dict(range=[0, 1]), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

            else:
                start_time = time.time()
                try:
                    predicted_arr = reg_model.predict(input_df)
                except Exception as e:
                    st.error(f"❌ Regression API failed: {e}")
                    st.stop()
                if len(predicted_arr) > 0:
                    predicted_emi = float(predicted_arr[0])
                else:
                    st.error("❌ Regression returned no prediction")
                    st.stop()
                inference_time = (time.time() - start_time) * 1000
                st.success(f"✅ Prediction Complete in {inference_time:.0f}ms!")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Predicted Max EMI", f"₹{predicted_emi:,.0f}")
                col2.metric("Total Loan", f"₹{predicted_emi * requested_tenure:,.0f}")
                col3.metric("% of Salary", f"{(predicted_emi/monthly_salary)*100:.1f}%")
                col4.metric("Inference Time", f"{inference_time:.0f}ms")

        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

# -------------------------
# OTHER PAGES
# -------------------------
elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")
    if not data_ok:
        st.error("❌ Dataset not loaded")
        st.stop()
    st.dataframe(df.head(100), use_container_width=True)

elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    if metadata.get("clf_metrics"):
        st.markdown("### 🎯 Classification Metrics")
        metrics = metadata['clf_metrics']
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics.get('test_accuracy', 0):.4f}")
        col2.metric("F1 Score", f"{metrics.get('test_f1', 0):.4f}")
        col3.metric("Precision", f"{metrics.get('test_precision', 0):.4f}")
        col4.metric("Recall", f"{metrics.get('test_recall', 0):.4f}")
    if metadata.get("reg_metrics"):
        st.markdown("### 📊 Regression Metrics")
        metrics = metadata['reg_metrics']
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R² Score", f"{metrics.get('test_r2', 0):.4f}")
        col2.metric("RMSE", f"₹{metrics.get('test_rmse', 0):,.0f}")
        col3.metric("MAE", f"₹{metrics.get('test_mae', 0):,.0f}")
        col4.metric("MAPE", f"{metrics.get('test_mape', 0):.1f}%")

elif page == "🔧 System Info":
    st.title("🔧 System Information")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🌐 MLflow")
        st.code(f"URI: {MLFLOW_TRACKING_URI}\nStorage: S3\nCache: Local")
        st.markdown("### 💾 Cache Status")
        cache_path = Path(MODEL_CACHE_DIR)
        if cache_path.exists():
            cache_files = list(cache_path.glob("*.pkl"))
            st.success(f"✅ {len(cache_files)} cached files")
            total_size = sum(f.stat().st_size for f in cache_files) / (1024*1024)
            st.metric("Total Cache Size", f"{total_size:.1f} MB")
            if st.button("🗑️ Clear Cache"):
                for f in cache_files:
                    try:
                        f.unlink()
                    except:
                        pass
                st.success("Cache cleared!")
                st.experimental_rerun()
    with col2:
        st.markdown("### 📊 Model Status")
        if clf_model:
            if getattr(clf_model, "ready", False):
                st.success("✅ Classification endpoint reachable")
            else:
                st.error(f"❌ Classification endpoint not reachable: {getattr(clf_model, 'last_error', 'unknown')}")
            if metadata.get("clf_meta"):
                st.caption(f"v{metadata['clf_meta']['version']} - {metadata['clf_meta']['stage']}")
        if reg_model:
            if getattr(reg_model, "ready", False):
                st.success("✅ Regression endpoint reachable")
            else:
                st.error(f"❌ Regression endpoint not reachable: {getattr(reg_model, 'last_error', 'unknown')}")
            if metadata.get("reg_meta"):
                st.caption(f"v{metadata['reg_meta']['version']} - {metadata['reg_meta']['stage']}")
