import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_breast_cancer
from src.benchmark import benchmark_classifiers

st.set_page_config(page_title="Sector-Based ML Lab", page_icon="🧪", layout="wide")
st.title("🧪 Sector-Based Machine Learning Lab")
st.caption("Compare model families using cross-validation and task-relevant metrics")
@st.cache_data
def load_demo():
    d=load_breast_cancer(as_frame=True)
    return d.data, d.target, d.target_names
X,y,names=load_demo()
st.warning("The current demo uses a built-in public sklearn dataset. It is a research/education dataset and is not a medical diagnostic tool.")
if st.button("Run model benchmark", type="primary"):
    with st.spinner("Cross-validating candidate models..."):
        results=benchmark_classifiers(X,y)
    st.dataframe(results, use_container_width=True)
    st.plotly_chart(px.bar(results, x="model", y="f1_mean", error_y="f1_std", title="Macro F1 comparison"), use_container_width=True)
    st.markdown("**Interpretation:** Compare performance and variance across folds. Select a model based on the relevant metric, stability, calibration needs, explainability and business cost—not a single leaderboard score.")
with st.expander("Dataset preview"):
    st.dataframe(X.head())
