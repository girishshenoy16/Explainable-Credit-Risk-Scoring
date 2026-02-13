import pandas as pd
import shap
from joblib import load

df = pd.read_csv("data/processed/features.csv")

X = pd.get_dummies(df.drop(columns=["default"]), drop_first=True)
feature_cols = load("models/feature_columns.pkl")
X = X[feature_cols]

scaler = load("models/scaler.pkl")
model = load("models/credit_model.pkl")

X_scaled = scaler.transform(X)

explainer = shap.LinearExplainer(model, X_scaled)
shap_values = explainer.shap_values(X_scaled)

print("✅ SHAP explainer ready")
print("Run shap.summary_plot() in notebook or Streamlit")