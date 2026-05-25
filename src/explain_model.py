import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import load

# --------------------------------------------------
# Output directory
# --------------------------------------------------
OUTPUT_DIR = Path("outputs/visuals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
df = pd.read_csv("data/processed/features.csv")

X = pd.get_dummies(
    df.drop(columns=["default"]),
    drop_first=True
)

# --------------------------------------------------
# Load model artifacts
# --------------------------------------------------
feature_cols = load("models/feature_columns.pkl")
X = X[feature_cols]

scaler = load("models/scaler.pkl")
model = load("models/credit_model.pkl")

# --------------------------------------------------
# Scale data
# --------------------------------------------------
X_scaled = scaler.transform(X)

# --------------------------------------------------
# SHAP explainer
# --------------------------------------------------
background = X_scaled[:100]

explainer = shap.LinearExplainer(
    model,
    background,
    feature_perturbation="interventional"
)

shap_values = explainer.shap_values(X_scaled)

# --------------------------------------------------
# SHAP Summary Plot
# --------------------------------------------------
shap.summary_plot(
    shap_values,
    X_scaled,
    feature_names=X.columns,
    show=False
)

plt.savefig(
    OUTPUT_DIR / "shap_summary.png",
    bbox_inches="tight"
)

plt.close()

print("✅ SHAP summary plot saved to outputs/visuals/")