import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_breast_cancer
from src.benchmark import benchmark_classifiers

st.set_page_config(page_title="Sector-Based ML Lab", page_icon="🧪", layout="wide")
st.markdown("""
<style>
.block-container {padding-top:1.6rem; padding-bottom:2.5rem;}
[data-testid="stMetric"] {background:rgba(120,120,120,.07); padding:14px 16px; border-radius:14px; border:1px solid rgba(120,120,120,.15);}
</style>
""", unsafe_allow_html=True)
st.title("🧪 Sector-Based ML Lab")
st.caption("A hands-on model evaluation studio: compare algorithms, metrics, and validation stability.")

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
