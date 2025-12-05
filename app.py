# app_fast_final.py
"""
EMIPredict AI - Final Fast Version (Modern Dashboard UI - Option A)
Features:
- Instant UI load (lazy loading of heavy resources)
- Enhanced Data Explorer with rich, explanatory visualizations
- Model Comparison dashboard using MLflow metadata
- Prediction pages using proxied model endpoints (cached)
- Clean, modular structure for easy editing

Notes:
- Update MLFLOW_TRACKING_URI, CLASSIFICATION_URL, REGRESSION_URL, and S3_CSV_URL as needed
- For local dev, create .streamlit/secrets.toml or rely on defaults
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import time
from io import BytesIO
from pathlib import Path
from mlflow.tracking import MlflowClient
import warnings
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# -------------------------
# CONFIG (change if needed)
# -------------------------
MLFLOW_TRACKING_URI = st.secrets.get("MLFLOW_TRACKING_URI", "http://13.204.193.251:5000")
CLASSIFICATION_URL = st.secrets.get("CLASSIFICATION_URL", "http://13.204.193.251:9001/invocations")
REGRESSION_URL = st.secrets.get("REGRESSION_URL", "http://13.204.193.251:9002/invocations")
CLASSIFICATION_MODEL_NAME = "EMI_Classification_XGBoost"
REGRESSION_MODEL_NAME = "EMI_Regression_XGBoost"
S3_CSV_URL = st.secrets.get("S3_CSV_URL", "https://mlflow-tracking-loan.s3.amazonaws.com/data/clean_emi_data.csv")
CACHE_DIR = Path(".cache_fast_app")
CACHE_DIR.mkdir(exist_ok=True)

# Dummy record for endpoint health checks
DUMMY_RECORD = {
    "age": 30, "gender": "Male", "marital_status": "Single", "education": "Graduate",
    "monthly_salary": 40000.0, "employment_type": "Private", "years_of_employment": 5.0,
    "company_type": "Medium", "house_type": "Rented", "monthly_rent": 0.0,
    "family_size": 3.0, "dependents": 1.0, "school_fees": 0.0, "college_fees": 0.0,
    "travel_expenses": 2000.0, "groceries_utilities": 8000.0, "other_monthly_expenses": 1000.0,
    "existing_loans": "No", "current_emi_amount": 0.0, "credit_score": 700.0,
    "bank_balance": 50000.0, "emergency_fund": 15000.0, "emi_scenario": "Personal Loan EMI",
    "requested_amount": 100000.0, "requested_tenure": 12.0
}

# -------------------------
# UTILITIES (cached session)
# -------------------------
@st.cache_resource
def requests_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "EMIPredictAI/fast-v1"})
    return s

# -------------------------
# STYLING (modern dashboard)
# -------------------------
@st.cache_resource
def apply_css():
    css = r"""
    <style>
    :root{--primary:#1f77b4;--accent:#764ba2}
    #MainMenu{visibility:hidden} footer{visibility:hidden}
    .card{background:#fff;border-radius:12px;padding:16px;box-shadow:0 6px 18px rgba(15,23,42,0.06);}
    .kpi{font-size:18px;font-weight:700}
    .muted{color:#6b7280}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

apply_css()

# -------------------------
# MODEL PROXY (lightweight + cached)
# -------------------------
class ModelProxy:
    def __init__(self, url, timeout=8):
        self.url = url.rstrip('/')
        self.timeout = timeout
        self.session = requests_session()
        self.ready = False
        self.last_error = None
        self._checked = False

    def _health_check(self):
        if self._checked:
            return self.ready
        self._checked = True
        # try GET /ping
        try:
            ping = self.url.replace('/invocations', '/ping')
            r = self.session.get(ping, timeout=min(2, self.timeout))
            if r.status_code == 200:
                self.ready = True
                return True
        except Exception:
            pass
        # fallback: short dummy POST
        try:
            r = self.session.post(self.url, json={"inputs": [DUMMY_RECORD]}, timeout=min(4, self.timeout))
            if r.status_code == 200:
                self.ready = True
                return True
            self.last_error = f"Status {r.status_code}"
        except Exception as e:
            self.last_error = str(e)
        return self.ready

    def ensure_ready(self):
        return self._health_check()

    def predict(self, df_or_records):
        if isinstance(df_or_records, pd.DataFrame):
            payload = {"inputs": df_or_records.to_dict(orient='records')}
        elif isinstance(df_or_records, (list, tuple)):
            payload = {"inputs": list(df_or_records)}
        elif isinstance(df_or_records, dict):
            payload = {"inputs": [df_or_records]}
        else:
            raise ValueError("Unsupported input for predict")
        r = self.session.post(self.url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        res = r.json()
        preds = res.get('predictions') if isinstance(res, dict) else res
        return np.array(preds)

    def predict_proba(self, df_or_records):
        preds = self.predict(df_or_records)
        if preds.ndim == 2:
            return preds
        raise ValueError("Server did not return probability vector")

@st.cache_resource
def get_class_proxy():
    return ModelProxy(CLASSIFICATION_URL)

@st.cache_resource
def get_reg_proxy():
    return ModelProxy(REGRESSION_URL)

# -------------------------
# MLflow helpers (lazy, cached)
# -------------------------
@st.cache_data(ttl=600)
def fetch_mlflow_metadata(limit=5):
    result = {"clf_versions": [], "reg_versions": [], "ok": False, "error": None}
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        try:
            clf_versions = client.search_model_versions(f"name='{CLASSIFICATION_MODEL_NAME}'")
            for v in clf_versions[:limit]:
                run = client.get_run(v.run_id)
                result['clf_versions'].append({
                    'version': v.version, 'stage': v.current_stage,
                    'metrics': dict(run.data.metrics), 'run_id': v.run_id
                })
        except Exception:
            pass
        try:
            reg_versions = client.search_model_versions(f"name='{REGRESSION_MODEL_NAME}'")
            for v in reg_versions[:limit]:
                run = client.get_run(v.run_id)
                result['reg_versions'].append({
                    'version': v.version, 'stage': v.current_stage,
                    'metrics': dict(run.data.metrics), 'run_id': v.run_id
                })
        except Exception:
            pass
        result['ok'] = True
    except Exception as e:
        result['error'] = str(e)
    return result

@st.cache_resource
def load_label_encoder_from_mlflow():
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        versions = client.search_model_versions(f"name='{CLASSIFICATION_MODEL_NAME}'")
        if not versions:
            return None
        latest = max(versions, key=lambda x: int(x.version))
        artifacts = client.list_artifacts(latest.run_id)
        for a in artifacts:
            if 'label_encoder' in a.path.lower() and a.path.endswith('.pkl'):
                local = client.download_artifacts(latest.run_id, a.path, dst_path=str(CACHE_DIR))
                return joblib.load(local)
    except Exception:
        return None
    return None

# -------------------------
# Data loader (deferred)
# -------------------------
@st.cache_data(ttl=3600)
def load_csv_from_url(url: str, timeout=60):
    session = requests_session()
    r = session.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    buf = BytesIO()
    for chunk in r.iter_content(1024 * 1024):
        if chunk:
            buf.write(chunk)
    buf.seek(0)
    df = pd.read_csv(buf)
    return df

# -------------------------
# APP CONFIG
# -------------------------
st.set_page_config(page_title="EMIPredict AI — Fast", page_icon="💰", layout="wide")

# placeholders in session state
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'mlflow_meta' not in st.session_state:
    st.session_state['mlflow_meta'] = None
if 'label_encoder' not in st.session_state:
    st.session_state['label_encoder'] = None

# Sidebar
st.sidebar.title("💰 EMIPredict AI")
st.sidebar.markdown("Modern Dashboard — Fast, lazy-loaded")
page = st.sidebar.radio("Navigation", ["Home", "Predictions", "Data Explorer", "Model Comparison", "System"], index=0)
st.sidebar.markdown("---")

# Top quick status (non-blocking)
col_s1, col_s2 = st.sidebar.columns(2)
col_s1.markdown("**MLflow**")
col_s2.markdown("**Endpoints**")

# Basic home
if page == "Home":
    st.title("💰 EMIPredict AI")
    st.markdown("### Fast Modern Dashboard")
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        st.markdown("""
        **What this app does**
        - Predict EMI eligibility (classification) and maximum EMI (regression) via remote model serving.
        - Interactive Data Explorer with rich visualizations.
        - Model Comparison dashboard powered by MLflow metadata.
        """)
        st.markdown("---")
        st.markdown("### Quick Actions")
        if st.button("Open Predictions"):
            st.experimental_set_query_params(page='predictions')
            st.experimental_rerun()
    with c2:
        st.metric("Models Deployed", "2")
        st.markdown("Learn more in Model Comparison")
    with c3:
        st.metric("Dataset", "Not loaded", delta="Click Data Explorer")

# -------------------------
# PREDICTIONS (lazy load proxies + data if needed)
# -------------------------
elif page == "Predictions":
    st.title("🔮 Predictions")
    # Lazy create proxies
    clf_proxy = get_class_proxy()
    reg_proxy = get_reg_proxy()

    # health status
    status_col1, status_col2, status_col3 = st.columns(3)
    status_col1.metric("Classification", "Ready" if clf_proxy.ensure_ready() else "Not Ready")
    status_col2.metric("Regression", "Ready" if reg_proxy.ensure_ready() else "Not Ready")
    if clf_proxy.last_error:
        status_col1.caption(clf_proxy.last_error)

    # load small portion of dataset if not present (for selecting emi_scenario options)
    if st.session_state['df'] is None:
        try:
            # load only headers or small sample by requesting range? fallback to full load
            st.session_state['df'] = None
        except Exception:
            st.session_state['df'] = None

    with st.form('predict_form'):
        st.markdown('### Customer Info')
        cols = st.columns(3)
        with cols[0]:
            age = st.number_input('Age', 18, 80, 30)
            gender = st.selectbox('Gender', ['Male','Female'])
            marital_status = st.selectbox('Marital Status', ['Single','Married'])
        with cols[1]:
            monthly_salary = st.number_input('Monthly Salary (₹)', 5000, 500000, 50000, step=1000)
            employment_type = st.selectbox('Employment Type',['Private','Government','Self-employed'])
            years_of_employment = st.number_input('Years of Employment', 0, 50, 5)
        with cols[2]:
            credit_score = st.number_input('Credit Score', 300, 850, 700)
            requested_amount = st.number_input('Requested Loan (₹)', 10000, 5000000, 100000, step=10000)
            requested_tenure = st.number_input('Tenure (months)', 3, 120, 12)

        pred_type = st.radio('Prediction Type', ['Classification (Eligibility)','Regression (Max EMI)'], horizontal=True)
        submitted = st.form_submit_button('Generate Prediction')

    if submitted:
        input_data = {
            'age': age, 'gender': gender, 'marital_status': marital_status,
            'monthly_salary': monthly_salary, 'employment_type': employment_type,
            'years_of_employment': years_of_employment, 'credit_score': credit_score,
            'requested_amount': requested_amount, 'requested_tenure': requested_tenure
        }
        input_df = pd.DataFrame([input_data])

        with st.spinner('Calling model...'):
            try:
                if pred_type.startswith('Classification'):
                    if not clf_proxy.ensure_ready():
                        st.error('Classification endpoint not available')
                    else:
                        preds = clf_proxy.predict(input_df)
                        try:
                            proba = clf_proxy.predict_proba(input_df)
                        except Exception:
                            proba = None
                        pred = preds[0] if len(preds)>0 else None
                        label_map = {0:'Eligible',1:'High_Risk',2:'Not_Eligible'}
                        predicted_label = label_map.get(int(pred), str(pred)) if pred is not None else 'N/A'
                        st.success(f'Prediction: {predicted_label}')
                        if proba is not None:
                            probs = np.array(proba).ravel()
                            labels = ['Eligible','High_Risk','Not_Eligible']
                            fig = go.Figure(go.Bar(x=labels, y=probs, text=[f"{p*100:.1f}%" for p in probs], textposition='auto'))
                            fig.update_layout(title='Class Probabilities')
                            st.plotly_chart(fig, width='stretch')
                else:
                    if not reg_proxy.ensure_ready():
                        st.error('Regression endpoint not available')
                    else:
                        preds = reg_proxy.predict(input_df)
                        val = float(preds[0]) if len(preds)>0 else None
                        st.success(f'Predicted Max EMI: ₹{val:,.0f}' if val is not None else 'No prediction')
            except Exception as e:
                st.error(f'Prediction failed: {e}')

# -------------------------
# DATA EXPLORER (rich visuals)
# -------------------------
elif page == 'Data Explorer':
    st.title('📊 Data Explorer — Enhanced')

    if st.session_state['df'] is None:
        with st.spinner('Loading dataset (cached)...'):
            try:
                df = load_csv_from_url(S3_CSV_URL)
                st.session_state['df'] = df
            except Exception as e:
                st.error(f'Failed to load dataset: {e}')
                st.stop()
    df = st.session_state['df']

    st.markdown('### Dataset Quick Summary')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Records', f"{len(df):,}")
    c2.metric('Features', len(df.columns))
    c3.metric('Missing', f"{df.isnull().sum().sum():,}")
    mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
    c4.metric('Memory (MB)', f"{mem_mb:.1f}")

    st.markdown('---')
    # Visual selection
    vis = st.selectbox('Choose visualization', [
        'Feature Distribution', 'Box & Violin', 'Correlation Heatmap', 'Feature vs Target', 'Outliers (IQR)', 'Pairwise (sampled)'
    ])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()

    if vis == 'Feature Distribution':
        col = st.selectbox('Numeric feature', numeric_cols)
        bins = st.slider('Bins', 10, 200, 50)
        fig = px.histogram(df, x=col, nbins=bins, marginal='box', title=f'Distribution of {col}')
        st.plotly_chart(fig, width='stretch')

    elif vis == 'Box & Violin':
        col = st.selectbox('Numeric feature', numeric_cols, key='bv')
        fig = make_subplots = None
        # display both
        fig1 = px.box(df, y=col, points='outliers', title=f'Box plot: {col}')
        fig2 = px.violin(df, y=col, box=True, title=f'Violin plot: {col}')
        st.plotly_chart(fig1, width='stretch')
        st.plotly_chart(fig2, width='stretch')

    elif vis == 'Correlation Heatmap':
        corr = df[numeric_cols].corr()
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu', zmid=0))
        fig.update_layout(title='Feature Correlation')
        st.plotly_chart(fig, width='stretch')

    elif vis == 'Feature vs Target':
        if 'emi_eligibility' in df.columns:
            target = 'emi_eligibility'
        elif 'maximum_emi_amount' in df.columns:
            target = 'maximum_emi_amount'
        else:
            target = st.selectbox('Select a categorical target from dataset', cat_cols)
        feat = st.selectbox('Feature', numeric_cols)
        if df[target].dtype == 'object' or df[target].nunique() < 10:
            fig = px.box(df, x=target, y=feat, title=f'{feat} by {target}')
        else:
            fig = px.scatter(df, x=feat, y=target, trendline='ols', title=f'{feat} vs {target}')
        st.plotly_chart(fig, width='stretch')

    elif vis == 'Outliers (IQR)':
        feat = st.selectbox('Numeric feature', numeric_cols, key='out')
        q1 = df[feat].quantile(0.25)
        q3 = df[feat].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[feat] < lower) | (df[feat] > upper)]
        st.markdown(f'**Outliers:** {len(outliers)} rows')
        st.dataframe(outliers.head(200), width='stretch')
        fig = px.box(df, y=feat, points='all')
        st.plotly_chart(fig, width='stretch')

    elif vis == 'Pairwise (sampled)':
        sample = df.sample(min(1000, len(df)), random_state=42)
        cols = st.multiselect('Select up to 4 features', numeric_cols, default=numeric_cols[:3])
        if len(cols) >= 2:
            fig = px.scatter_matrix(sample, dimensions=cols, color=cat_cols[0] if cat_cols else None)
            fig.update_layout(height=700)
            st.plotly_chart(fig, width='stretch')

    st.markdown('---')
    st.markdown('### Raw sample')
    st.dataframe(df.head(200), width='stretch', height=300)

# -------------------------
# MODEL COMPARISON (MLflow metadata + simple charts)
# -------------------------
elif page == 'Model Comparison':
    st.title('📈 Model Comparison')
    if st.session_state['mlflow_meta'] is None:
        with st.spinner('Fetching MLflow metadata...'):
            st.session_state['mlflow_meta'] = fetch_mlflow_metadata()
    meta = st.session_state['mlflow_meta']
    if meta.get('error'):
        st.error(f"MLflow error: {meta['error']}")
    # Classification table
    if meta and meta.get('clf_versions'):
        st.subheader('Classification Models')
        clf_df = pd.DataFrame(meta['clf_versions'])
        # metrics explode
        metrics = []
        for v in meta['clf_versions']:
            m = v.get('metrics', {})
            metrics.append({**{'version': v['version'], 'stage': v['stage']}, **m})
        mdf = pd.DataFrame(metrics).fillna(0)
        st.dataframe(mdf.sort_values('version', ascending=False), width='stretch')
        # accuracy bar
        if 'test_accuracy' in mdf.columns:
            fig = px.bar(mdf, x='version', y='test_accuracy', title='Classification: Test Accuracy by Version')
            st.plotly_chart(fig, width='stretch')

    if meta and meta.get('reg_versions'):
        st.subheader('Regression Models')
        reg_df = pd.DataFrame(meta['reg_versions'])
        metrics = []
        for v in meta['reg_versions']:
            m = v.get('metrics', {})
            metrics.append({**{'version': v['version'], 'stage': v['stage']}, **m})
        mdf = pd.DataFrame(metrics).fillna(0)
        st.dataframe(mdf.sort_values('version', ascending=False), width='stretch')
        if 'test_r2' in mdf.columns:
            fig = px.line(mdf, x='version', y='test_r2', title='Regression: Test R² by Version', markers=True)
            st.plotly_chart(fig, width='stretch')

# -------------------------
# SYSTEM / DEBUG
# -------------------------
else:
    st.title('🔧 System')
    st.markdown('### Endpoints')
    clf = get_class_proxy(); reg = get_reg_proxy()
    st.write('Classification ready:', clf.ensure_ready(), 'Error:', clf.last_error)
    st.write('Regression ready:', reg.ensure_ready(), 'Error:', reg.last_error)
    st.markdown('---')
    st.markdown('### MLflow metadata (cached)')
    st.json(fetch_mlflow_metadata())

# -------------------------
# END
# -------------------------

