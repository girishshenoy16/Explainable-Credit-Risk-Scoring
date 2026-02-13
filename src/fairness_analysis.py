import pandas as pd
from joblib import load
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    false_positive_rate,
    false_negative_rate
)
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Load data and model
# --------------------------------------------------
df = pd.read_csv("data/processed/features.csv")

# Sensitive attribute (Gender from UCI dataset)
sensitive_feature = "SEX"

X = pd.get_dummies(df.drop(columns=["default"]), drop_first=True)
y_true = df["default"]

feature_cols = load("models/feature_columns.pkl")
X = X[feature_cols]

scaler = load("models/scaler.pkl")
model = load("models/credit_model.pkl")

X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)

# --------------------------------------------------
# Fairness metrics
# --------------------------------------------------
metric_frame = MetricFrame(
    metrics={
        "accuracy": accuracy_score,
        "selection_rate": selection_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    },
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=df[sensitive_feature]
)

print("⚖️ Fairness Evaluation by Gender (SEX)")
print(metric_frame.by_group)