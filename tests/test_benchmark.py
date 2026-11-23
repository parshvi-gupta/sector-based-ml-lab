import pandas as pd
import pytest
from src.benchmark import build_candidates, benchmark_classifiers
from sklearn.datasets import load_breast_cancer

def test_candidates_exist():
    assert "Logistic Regression" in build_candidates()

def test_benchmark_returns_metrics():
    d=load_breast_cancer(as_frame=True)
    result=benchmark_classifiers(d.data, d.target, folds=3)
    assert not result.empty
    assert "f1_mean" in result.columns
    assert result["f1_mean"].between(0,1).all()

def test_one_class_rejected():
    X=pd.DataFrame({"x":[1,2,3,4]})
    y=pd.Series([0,0,0,0])
    with pytest.raises(ValueError):
        benchmark_classifiers(X,y)
