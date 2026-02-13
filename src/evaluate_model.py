import pandas as pd
from joblib import load
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("data/processed/features.csv")

X = pd.get_dummies(df.drop(columns=["default"]), drop_first=True)
y = df["default"]

feature_cols = load("models/feature_columns.pkl")
X = X[feature_cols]

scaler = load("models/scaler.pkl")
model = load("models/credit_model.pkl")

X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)

print("📊 Confusion Matrix")
print(confusion_matrix(y, y_pred))

print("\n📋 Classification Report")
print(classification_report(y, y_pred))