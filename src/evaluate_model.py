import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import load
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# --------------------------------------------------
# Output directory
# --------------------------------------------------
OUTPUT_DIR = Path("outputs/visuals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load processed dataset
# --------------------------------------------------
df = pd.read_csv("data/processed/features.csv")

X = pd.get_dummies(
    df.drop(columns=["default"]),
    drop_first=True
)

y = df["default"]

# --------------------------------------------------
# Load saved artifacts
# --------------------------------------------------
feature_cols = load("models/feature_columns.pkl")
X = X[feature_cols]

scaler = load("models/scaler.pkl")
model = load("models/credit_model.pkl")

# --------------------------------------------------
# Scale features
# --------------------------------------------------
X_scaled = scaler.transform(X)

# --------------------------------------------------
# Predictions
# --------------------------------------------------
y_pred = model.predict(X_scaled)
y_prob = model.predict_proba(X_scaled)[:, 1]

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------
cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix")

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# ROC Curve
# --------------------------------------------------
fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig(
    OUTPUT_DIR / "roc_curve.png",
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# Console reports
# --------------------------------------------------
print("📊 Confusion Matrix")
print(cm)

print("\n📋 Classification Report")
print(classification_report(y, y_pred))

print("\n✅ Evaluation visuals saved to outputs/visuals/")