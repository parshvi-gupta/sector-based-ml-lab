# Sector-Based ML Lab

An applied machine-learning playground built around public-style sector use cases. It demonstrates disciplined problem framing, preprocessing, baseline models, cross-validation, metric selection, model comparison, and explainability.

## Initial scope
- **Banking:** credit-risk classification using a synthetic starter dataset
- **Retail:** sales regression example (planned module)
- **Telecom:** churn classification (planned module)
- **Healthcare:** tabular classification demo using a public sklearn dataset

## Model selection philosophy
Model choice depends on task type, data size, class balance, leakage risk, business cost of errors, explainability, and validation strategy. The highest metric alone is not always the best production choice.

## Run
```bash
python -m venv .venv
# activate environment
pip install -r requirements.txt
streamlit run app.py
```

## Current app
Interactive model benchmark using scikit-learn's built-in breast-cancer dataset (classification), with stratified train/test split, cross-validation, and several candidate models. This is a learning/portfolio demonstration, not a medical tool.

## Roadmap
- Sector-specific dataset loaders
- Classification and regression task support
- Pipeline-based preprocessing
- Hyperparameter search
- Metrics tailored to business objectives
- SHAP explainability
- MLflow experiment tracking
- Tests, CI, Docker
