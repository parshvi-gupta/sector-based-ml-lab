"""Benchmark baseline classifiers with reproducible validation."""
from __future__ import annotations
from typing import Any
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC


def build_candidates(seed: int = 42) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=seed))]),
        "Random Forest": Pipeline([("scale", StandardScaler()), ("model", RandomForestClassifier(n_estimators=250, random_state=seed, class_weight="balanced"))]),
        "SVM (RBF)": Pipeline([("scale", StandardScaler()), ("model", SVC(kernel="rbf", class_weight="balanced"))]),
        "Hist Gradient Boosting": Pipeline([("scale", StandardScaler()), ("model", HistGradientBoostingClassifier(random_state=seed))]),
    }

def benchmark_classifiers(X: pd.DataFrame, y: pd.Series, seed: int = 42, folds: int = 5) -> pd.DataFrame:
    if y.nunique() < 2: raise ValueError("Classification requires at least two target classes")
    min_count = int(y.value_counts().min())
    n_splits = min(folds, min_count)
    if n_splits < 2: raise ValueError("Each class needs at least two observations for cross-validation")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scoring = {"accuracy":"accuracy", "precision":"precision_macro", "recall":"recall_macro", "f1":"f1_macro", "roc_auc":"roc_auc"}
    rows: list[dict[str, Any]] = []
    for name, model in build_candidates(seed).items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, error_score="raise")
        rows.append({"model":name, **{f"{metric}_mean":float(scores[f"test_{metric}"].mean()) for metric in scoring}, **{f"{metric}_std":float(scores[f"test_{metric}"].std()) for metric in scoring}})
    return pd.DataFrame(rows).sort_values("f1_mean", ascending=False).reset_index(drop=True)
