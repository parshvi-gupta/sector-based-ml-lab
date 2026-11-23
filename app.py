import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_breast_cancer
from src.benchmark import benchmark_classifiers

st.set_page_config(page_title="Sector-Based ML Lab", page_icon="🧪", layout="wide")
st.markdown('<style>
.stApp{background:linear-gradient(180deg,#f5f8fc 0%,#fff 40%);color:#10243a}
.block-container{max-width:1320px;padding-top:1rem;padding-bottom:3rem}
[data-testid="stSidebar"]{background:#f0f4f9;border-right:1px solid #e1e8f0}
[data-testid="stMetric"]{background:#fff;border:1px solid #e3ebf3;border-radius:16px;padding:16px 18px;box-shadow:0 3px 14px rgba(16,36,58,.04)}
[data-testid="stMetricLabel"] p{color:#6b7d90;font-size:.78rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase}
[data-testid="stMetricValue"]{color:#10243a;font-weight:750}
.stTabs [data-baseweb="tab-list"]{gap:8px;border-bottom:1px solid #e4ebf2}
.stTabs [data-baseweb="tab"]{height:48px;border-radius:12px 12px 0 0;padding:0 16px;font-weight:650}
.stTabs [aria-selected="true"]{color:#168e83!important;border-bottom-color:#19a89b!important}
div[data-testid="stDataFrame"],div[data-testid="stTable"]{border:1px solid #e4ebf2;border-radius:14px;overflow:hidden}
div[data-testid="stPlotlyChart"]{background:white;border:1px solid #e8eef4;border-radius:16px;padding:8px}
button[kind="primary"]{background:linear-gradient(90deg,#147e78,#19a89b);border:0;border-radius:10px;font-weight:700}
.hero{padding:24px 26px;border-radius:22px;background:linear-gradient(120deg,#10263d,#183c56 62%,#147e78);color:white;margin:0 0 20px;box-shadow:0 10px 30px rgba(16,36,58,.12)}
.hero h1{font-size:2.1rem;margin:0 0 7px;letter-spacing:-.04em;color:white}
.hero p{color:#d1e5ed;margin:0;font-size:1rem}
.hero .eyebrow{font-size:.74rem;font-weight:800;letter-spacing:.14em;color:#83ded3;margin-bottom:8px}
.section-label{color:#718399;font-size:.76rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase}
</style>', unsafe_allow_html=True)
st.markdown("""
<style>
.block-container {padding-top:1.6rem; padding-bottom:2.5rem;}
[data-testid="stMetric"] {background:rgba(120,120,120,.07); padding:14px 16px; border-radius:14px; border:1px solid rgba(120,120,120,.15);}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><div class="eyebrow">MACHINE LEARNING · MODEL EVALUATION</div><h1>Sector-Based ML Lab</h1><p>Benchmark algorithms, compare metrics and assess validation stability before selecting a candidate model.</p></div>', unsafe_allow_html=True)

@st.cache_data
def load_demo():
    data = load_breast_cancer(as_frame=True)
    return data.data, data.target, list(data.target_names)

X,y,target_names = load_demo()
st.warning("This demo uses the public scikit-learn breast cancer dataset for education only. It is not a medical diagnostic tool.")
st.subheader("Experiment setup")
c1,c2,c3 = st.columns(3)
c1.metric("Samples", f"{len(X):,}")
c2.metric("Features", f"{X.shape[1]:,}")
c3.metric("Classes", f"{y.nunique()}")
with st.expander("Dataset details"):
    st.write("Target labels:", target_names)
    st.dataframe(X.head(10), use_container_width=True)
st.divider()
if st.button("▶ Run model benchmark", type="primary"):
    with st.spinner("Running stratified cross-validation for four candidate models..."):
        try:
            results = benchmark_classifiers(X, y)
            st.session_state["benchmark_results"] = results
            st.session_state["benchmark_meta"] = {"samples":len(X),"features":X.shape[1],"classes":int(y.nunique())}
        except Exception as exc:
            st.error(f"Benchmark failed ({type(exc).__name__}): {exc}")
if "benchmark_results" in st.session_state:
    results = st.session_state["benchmark_results"]
    st.subheader("Results")
    best = results.iloc[0]
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Top model by macro F1", best["model"])
    k2.metric("Macro F1", f"{best['f1_mean']:.3f} ± {best['f1_std']:.3f}")
    k3.metric("Accuracy", f"{best['accuracy_mean']:.3f}")
    k4.metric("ROC AUC", f"{best['roc_auc_mean']:.3f}")
    left,right = st.columns(2)
    with left:
        st.plotly_chart(px.bar(results, x="model", y="f1_mean", error_y="f1_std", title="Macro F1 (mean ± fold variability)", labels={"f1_mean":"Mean macro F1"}), use_container_width=True)
    with right:
        st.plotly_chart(px.bar(results, x="model", y="roc_auc_mean", error_y="roc_auc_std", title="ROC AUC (mean ± fold variability)"), use_container_width=True)
    st.subheader("Full benchmark table")
    metric_cols = [c for c in results.columns if c.endswith("_mean") or c.endswith("_std")]
    st.dataframe(results[["model"]+metric_cols].style.format("{:.3f}", subset=metric_cols), use_container_width=True)
    st.subheader("How to interpret")
    st.markdown("""
    - **Macro F1** balances precision and recall across classes.
    - **ROC AUC** measures ranking separation; interpret it alongside class balance and error costs.
    - **Fold variability** (standard deviation) helps reveal stability across validation splits.
    - The best score is not automatically the best business model: consider explainability, calibration, latency, operational cost, and consequences of false positives/negatives.
    """)
