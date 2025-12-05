# app.py
"""
EMIPredict AI - Clean, Fast, Production-ready Streamlit app
- Instant UI load (no heavy work at import)
- Lazy-load S3 CSV, MLflow metadata, label encoder, model proxies
- Modern UI styling (Option A)
- Enhanced Data Explorer + Model Comparison
- Uses width='stretch' to conform to Streamlit deprecation notice
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import time
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, Any, Dict
from mlflow.tracking import MlflowClient
import warnings

warnings.filterwarnings("ignore")

# -------------------------
# CONFIG (defaults, overridable via st.secrets)
# -------------------------
MLFLOW_TRACKING_URI = st.secrets.get("MLFLOW_TRACKING_URI", "http://13.204.193.251:5000")
CLASSIFICATION_URL = st.secrets.get("CLASSIFICATION_URL", "http://13.204.193.251:9001/invocations")
REGRESSION_URL = st.secrets.get("REGRESSION_URL", "http://13.204.193.251:9002/invocations")
CLASSIFICATION_MODEL_NAME = st.secrets.get("CLASSIFICATION_MODEL_NAME", "EMI_Classification_XGBoost")
REGRESSION_MODEL_NAME = st.secrets.get("REGRESSION_MODEL_NAME", "EMI_Regression_XGBoost")
S3_CSV_URL = st.secrets.get("S3_CSV_URL", "https://mlflow-tracking-loan.s3.amazonaws.com/data/clean_emi_data.csv")
CACHE_DIR = Path(st.secrets.get("CACHE_DIR", ".cache_app"))
CACHE_DIR.mkdir(exist_ok=True)

# Health-check dummy record (full-schema)
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
# UTILITIES
# -------------------------
@st.cache_resource
def requests_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "EMIPredictAI/fast-v1"})
    return s

def safe_get_secret(key: str, default: Any) -> Any:
    try:
        return st.secrets[key]
    except Exception:
        return default

# -------------------------
# STYLING
# -------------------------
@st.cache_resource
def apply_css():
    css = r"""
    <style>
    :root{--primary:#1f77b4;--accent:#764ba2}
    #MainMenu{visibility:hidden} footer{visibility:hidden}
    .card{background:#fff;border-radius:12px;padding:14px;box-shadow:0 6px 18px rgba(15,23,42,0.06)}
    .kpi{font-size:18px;font-weight:700}
    .muted{color:#6b7280}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

apply_css()

# -------------------------
# MODEL PROXY (fast, cached)
# -------------------------
class ModelProxy:
    def __init__(self, url: str, timeout: int = 8):
        self.url = url.rstrip('/')
        self.timeout = timeout
        self.session = requests_session()
        self.ready = False
        self.last_error: Optional[str] = None
        self._checked = False

    def _health_check(self) -> bool:
        if self._checked:
            return self.ready
        self._checked = True
        # Try /ping (short)
        try:
            ping_url = self.url.replace("/invocations", "/ping")
            r = self.session.get(ping_url, timeout=min(2, self.timeout))
            if r.status_code == 200:
                self.ready = True
                return True
        except Exception:
            pass
        # Fallback to light dummy POST
        try:
            r = self.session.post(self.url, json={"inputs": [DUMMY_RECORD]}, timeout=min(4, self.timeout))
            if r.status_code == 200:
                self.ready = True
                return True
            self.last_error = f"Status {r.status_code}: {r.text[:200]}"
        except Exception as e:
            self.last_error = str(e)
            self.ready = False
        return self.ready

    def ensure_ready(self) -> bool:
        return self._health_check()

    def predict(self, df_or_records):
        if isinstance(df_or_records, pd.DataFrame):
            payload = {"inputs": df_or_records.to_dict(orient="records")}
        elif isinstance(df_or_records, dict):
            payload = {"inputs": [df_or_records]}
        elif isinstance(df_or_records, (list, tuple)):
            payload = {"inputs": list(df_or_records)}
        else:
            raise ValueError("Unsupported input type for predict()")
        r = self.session.post(self.url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        result = r.json()
        preds = result.get("predictions", None)
        if preds is None:
            # sometimes returns list directly
            if isinstance(result, list):
                preds = result
            else:
                # unknown format
                return np.array([])
        return np.array(preds)

    def predict_proba(self, df_or_records):
        preds = self.predict(df_or_records)
        if preds.ndim == 2:
            return preds
        raise ValueError("Server did not return probability vector")

@st.cache_resource
def get_classification_proxy():
    return ModelProxy(CLASSIFICATION_URL)

@st.cache_resource
def get_regression_proxy():
    return ModelProxy(REGRESSION_URL)

# -------------------------
# MLflow + label encoder (lazy)
# -------------------------
@st.cache_data(ttl=600)
def fetch_mlflow_metadata(limit: int = 5) -> Dict:
    out = {"clf_versions": [], "reg_versions": [], "ok": False, "error": None}
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        try:
            cv = client.search_model_versions(f"name='{CLASSIFICATION_MODEL_NAME}'")
            for v in cv[:limit]:
                try:
                    run = client.get_run(v.run_id)
                    metrics = dict(run.data.metrics)
                except Exception:
                    metrics = {}
                out["clf_versions"].append({"version": v.version, "stage": v.current_stage, "metrics": metrics})
        except Exception:
            pass
        try:
            rv = client.search_model_versions(f"name='{REGRESSION_MODEL_NAME}'")
            for v in rv[:limit]:
                try:
                    run = client.get_run(v.run_id)
                    metrics = dict(run.data.metrics)
                except Exception:
                    metrics = {}
                out["reg_versions"].append({"version": v.version, "stage": v.current_stage, "metrics": metrics})
        except Exception:
            pass
        out["ok"] = True
    except Exception as e:
        out["error"] = str(e)
    return out

@st.cache_resource
def load_label_encoder() -> Optional[Any]:
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        versions = client.search_model_versions(f"name='{CLASSIFICATION_MODEL_NAME}'")
        if not versions:
            return None
        latest = max(versions, key=lambda x: int(x.version))
        artifacts = client.list_artifacts(latest.run_id)
        for a in artifacts:
            if "label_encoder" in a.path.lower() and a.path.endswith(".pkl"):
                local = client.download_artifacts(latest.run_id, a.path, dst_path=str(CACHE_DIR))
                return joblib.load(local)
    except Exception:
        return None
    return None

# -------------------------
# DATA LOADER (deferred, cached)
# -------------------------
@st.cache_data(ttl=3600)
def load_csv(url: str, timeout: int = 60) -> pd.DataFrame:
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
# SAFE predict helpers
# -------------------------
def classify_and_render(proxy: ModelProxy, input_df: pd.DataFrame, label_encoder=None):
    start = time.time()
    preds = proxy.predict(input_df)
    try:
        proba = proxy.predict_proba(input_df)
    except Exception:
        proba = None
    elapsed_ms = (time.time() - start) * 1000
    # decode predicted label
    if len(preds) == 0:
        st.error("No prediction returned from server")
        return
    pred = preds[0]
    if label_encoder is not None:
        try:
            decoded = label_encoder.inverse_transform([int(pred)])[0]
        except Exception:
            decoded = str(pred)
    else:
        mapping = {0: "Eligible", 1: "High_Risk", 2: "Not_Eligible"}
        decoded = mapping.get(int(pred), str(pred))
    # display
    st.success(f"✅ Prediction: {decoded} (in {elapsed_ms:.0f} ms)")
    if proba is not None:
        probs = np.array(proba).ravel()
        if label_encoder is not None and hasattr(label_encoder, "classes_"):
            labels = list(label_encoder.classes_)
        else:
            labels = ["Eligible", "High_Risk", "Not_Eligible"]
        fig = go.Figure(go.Bar(x=labels, y=probs, text=[f"{p*100:.1f}%" for p in probs], textposition="auto"))
        fig.update_layout(title="Class Probabilities")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Probability vector not available from server")

def regress_and_render(proxy: ModelProxy, input_df: pd.DataFrame):
    start = time.time()
    preds = proxy.predict(input_df)
    elapsed_ms = (time.time() - start) * 1000
    if len(preds) == 0:
        st.error("No regression prediction")
        return
    val = float(preds[0])
    st.success(f"✅ Predicted Max EMI: ₹{val:,.0f} (in {elapsed_ms:.0f} ms)")
    return val

# -------------------------
# APP LAYOUT
# -------------------------
st.set_page_config(page_title="EMIPredict AI", page_icon="💰", layout="wide")

# Session placeholders (make sure UI loads instantly)
if "df" not in st.session_state:
    st.session_state["df"] = None
if "mlflow_meta" not in st.session_state:
    st.session_state["mlflow_meta"] = None
if "label_encoder" not in st.session_state:
    st.session_state["label_encoder"] = None

# Sidebar (lightweight; NO heavy calls here)
st.sidebar.title("💰 EMIPredict AI")
st.sidebar.markdown("Modern, fast dashboard — lazy loaded")
page = st.sidebar.radio("", ["Home", "Predictions", "Data Explorer", "Model Comparison", "System"])
st.sidebar.markdown("---")
st.sidebar.caption(f"MLflow: {MLFLOW_TRACKING_URI.split('//')[-1]}")
st.sidebar.caption(f"Endpoints: {CLASSIFICATION_URL.split('//')[-1]} , {REGRESSION_URL.split('//')[-1]}")

# Page: Home
if page == "Home":
    st.title("💰 EMIPredict AI")
    st.markdown("### Fast, production-ready dashboard")
    left, right = st.columns([3,1])
    with left:
        st.markdown("""
        **What this app does**
        - Predict EMI eligibility (classification) and maximum EMI (regression) using remote model endpoints.
        - Explore dataset with rich visualizations.
        - Compare model versions via MLflow metadata.
        """)
        if st.button("Start Predictions"):
            st.experimental_set_query_params(page="Predictions")
            st.experimental_rerun()
    with right:
        st.metric("Models Deployed", "2")
        st.metric("Dataset", "Lazy load (Data Explorer)")
    st.markdown("---")

# Page: Predictions
elif page == "Predictions":
    st.title("🔮 Real-time Predictions (API proxied)")
    # create proxies lazily
    clf_proxy = get_classification_proxy()
    reg_proxy = get_regression_proxy()

    # Show status small
    col1, col2 = st.columns(2)
    col1.metric("Classification", "Ready" if clf_proxy.ensure_ready() else "Not Ready")
    if clf_proxy.last_error:
        col1.caption(clf_proxy.last_error)
    col2.metric("Regression", "Ready" if reg_proxy.ensure_ready() else "Not Ready")
    if reg_proxy.last_error:
        col2.caption(reg_proxy.last_error)

    # input form
    with st.form("predict_form"):
        st.markdown("### Customer Information")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 18, 80, 30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        with c2:
            monthly_salary = st.number_input("Monthly Salary (₹)", 5000, 500000, 50000, step=1000)
            employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
            years_of_employment = st.number_input("Years of Employment", 0, 50, 5)
        with c3:
            credit_score = st.number_input("Credit Score", 300, 850, 700)
            requested_amount = st.number_input("Requested Loan (₹)", 10000, 5000000, 100000, step=10000)
            requested_tenure = st.number_input("Tenure (months)", 3, 120, 12)

        pred_type = st.radio("Prediction Type", ["Classification (Eligibility)", "Regression (Max EMI)"], horizontal=True)
        submitted = st.form_submit_button("⚡ Generate Prediction")

    if submitted:
        input_data = {
            'age': age, 'gender': gender, 'marital_status': marital_status,
            'monthly_salary': monthly_salary, 'employment_type': employment_type,
            'years_of_employment': years_of_employment, 'credit_score': credit_score,
            'requested_amount': requested_amount, 'requested_tenure': requested_tenure
        }
        input_df = pd.DataFrame([input_data])
        with st.spinner("Calling model..."):
            try:
                if pred_type.startswith("Classification"):
                    # load label encoder on demand
                    if st.session_state["label_encoder"] is None:
                        st.session_state["label_encoder"] = load_label_encoder()
                    classify_and_render(clf_proxy, input_df, st.session_state["label_encoder"])
                else:
                    regress_and_render(reg_proxy, input_df)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# Page: Data Explorer (heavy action here on demand)
elif page == "Data Explorer":
    st.title("📊 Data Explorer — Enhanced visuals")
    if st.session_state["df"] is None:
        with st.spinner("Loading dataset (cached)..."):
            try:
                st.session_state["df"] = load_csv(S3_CSV_URL)
            except Exception as e:
                st.error(f"Failed to load dataset from S3: {e}")
                st.stop()
    df = st.session_state["df"]

    # Quick KPIs
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Records", f"{len(df):,}")
    r2.metric("Features", len(df.columns))
    r3.metric("Missing", f"{df.isnull().sum().sum():,}")
    mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
    r4.metric("Memory (MB)", f"{mem_mb:.1f}")

    st.markdown("---")
    # Visual selector
    viz = st.selectbox("Choose visualization", [
        "Feature Distribution", "Box & Violin", "Correlation Heatmap",
        "Feature vs Target", "Outliers (IQR)", "Pairwise (sampled)"
    ])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if viz == "Feature Distribution":
        col = st.selectbox("Numeric feature", numeric_cols)
        bins = st.slider("Bins", 10, 200, 50)
        fig = px.histogram(df, x=col, nbins=bins, marginal="box", title=f"Distribution of {col}")
        st.plotly_chart(fig, width="stretch")

    elif viz == "Box & Violin":
        col = st.selectbox("Numeric feature", numeric_cols, key="bv_col")
        fig1 = px.box(df, y=col, points="outliers", title=f"Box plot: {col}")
        fig2 = px.violin(df, y=col, box=True, title=f"Violin plot: {col}")
        st.plotly_chart(fig1, width="stretch")
        st.plotly_chart(fig2, width="stretch")

    elif viz == "Correlation Heatmap":
        corr = df[numeric_cols].corr()
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu", zmid=0))
        fig.update_layout(title="Feature Correlation")
        st.plotly_chart(fig, width="stretch")

    elif viz == "Feature vs Target":
        if "emi_eligibility" in df.columns:
            target = "emi_eligibility"
        elif "maximum_emi_amount" in df.columns:
            target = "maximum_emi_amount"
        else:
            target = st.selectbox("Select a categorical target", cat_cols)
        feat = st.selectbox("Feature", numeric_cols)
        if df[target].dtype == "object" or df[target].nunique() < 10:
            fig = px.box(df, x=target, y=feat, title=f"{feat} by {target}")
        else:
            fig = px.scatter(df, x=feat, y=target, trendline="ols", title=f"{feat} vs {target}")
        st.plotly_chart(fig, width="stretch")

    elif viz == "Outliers (IQR)":
        feat = st.selectbox("Numeric feature", numeric_cols, key="out")
        q1 = df[feat].quantile(0.25); q3 = df[feat].quantile(0.75); iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = df[(df[feat] < lower) | (df[feat] > upper)]
        st.markdown(f"**Outliers:** {len(outliers)} rows")
        st.dataframe(outliers.head(200), width="stretch")
        fig = px.box(df, y=feat, points="all")
        st.plotly_chart(fig, width="stretch")

    elif viz == "Pairwise (sampled)":
        sample = df.sample(min(1000, len(df)), random_state=42)
        cols = st.multiselect("Select up to 4 features", numeric_cols, default=numeric_cols[:3])
        if len(cols) >= 2:
            fig = px.scatter_matrix(sample, dimensions=cols, color=cat_cols[0] if cat_cols else None)
            fig.update_layout(height=700)
            st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.markdown("### Raw sample")
    st.dataframe(df.head(200), width="stretch", height=300)

# Page: Model Comparison (lazy fetch MLflow)
elif page == "Model Comparison":
    st.title("📈 Model Comparison (MLflow)")
    if st.session_state["mlflow_meta"] is None:
        with st.spinner("Fetching MLflow metadata..."):
            st.session_state["mlflow_meta"] = fetch_mlflow_metadata()
    meta = st.session_state["mlflow_meta"]
    if meta.get("error"):
        st.error(f"MLflow error: {meta['error']}")

    if meta.get("clf_versions"):
        st.subheader("Classification models")
        metrics_rows = []
        for v in meta["clf_versions"]:
            row = {"version": v["version"], "stage": v["stage"]}
            row.update(v.get("metrics", {}))
            metrics_rows.append(row)
        mdf = pd.DataFrame(metrics_rows).fillna(0)
        st.dataframe(mdf.sort_values("version", ascending=False), width="stretch")
        if "test_accuracy" in mdf.columns:
            fig = px.bar(mdf, x="version", y="test_accuracy", title="Classification: Test Accuracy")
            st.plotly_chart(fig, width="stretch")

    if meta.get("reg_versions"):
        st.subheader("Regression models")
        metrics_rows = []
        for v in meta["reg_versions"]:
            row = {"version": v["version"], "stage": v["stage"]}
            row.update(v.get("metrics", {}))
            metrics_rows.append(row)
        mdf = pd.DataFrame(metrics_rows).fillna(0)
        st.dataframe(mdf.sort_values("version", ascending=False), width="stretch")
        if "test_r2" in mdf.columns:
            fig = px.line(mdf, x="version", y="test_r2", title="Regression: Test R²", markers=True)
            st.plotly_chart(fig, width="stretch")

# Page: System (debug)
else:
    st.title("🔧 System")
    st.markdown("### Endpoint health")
    c1 = get_classification_proxy()
    c2 = get_regression_proxy()
    st.write("Classification ready:", c1.ensure_ready(), "last_error:", c1.last_error)
    st.write("Regression ready:", c2.ensure_ready(), "last_error:", c2.last_error)
    st.markdown("---")
    st.json(fetch_mlflow_metadata())

# End
