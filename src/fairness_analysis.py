import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import load

from fairlearn.metrics import (
    MetricFrame,
    selection_rate
)

from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Output directory
# --------------------------------------------------
OUTPUT_DIR = Path("outputs/visuals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
df = pd.read_csv("data/processed/features.csv")

# --------------------------------------------------
# Sensitive feature
# --------------------------------------------------
sensitive_feature = "SEX"

X = pd.get_dummies(
    df.drop(columns=["default"]),
    drop_first=True
)

y_true = df["default"]

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
# Predictions
# --------------------------------------------------
y_pred = model.predict(X_scaled)

# --------------------------------------------------
# Fairness Metrics
# --------------------------------------------------
metric_frame = MetricFrame(
    metrics={
        "accuracy": accuracy_score,
        "selection_rate": selection_rate,
    },
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=df[sensitive_feature]
)

fairness_df = metric_frame.by_group.reset_index()

# --------------------------------------------------
# Label encoding
# --------------------------------------------------
fairness_df["Gender"] = fairness_df["SEX"].map({
    1: "Male",
    2: "Female"
})

# --------------------------------------------------
# Visualization
# --------------------------------------------------
plt.figure(figsize=(6, 5))

plt.bar(
    fairness_df["Gender"],
    fairness_df["selection_rate"]
)

plt.ylabel("Approval Rate")
plt.title("Approval Rate by Gender")

plt.savefig(
    OUTPUT_DIR / "fairness_comparison.png",
    bbox_inches="tight"
)

plt.close()

print("⚖️ Fairness Evaluation by Gender")
print(fairness_df)

print("\n✅ Fairness visualization saved to outputs/visuals/")