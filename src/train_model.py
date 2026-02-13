# src/train_model.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from joblib import dump

DATA_PATH = Path("data/processed/features.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["default"])
    y = df["default"]

    X = pd.get_dummies(X, drop_first=True)

    # ✅ FINAL NaN CHECK (industry standard)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs"
    )

    model.fit(X_train_scaled, y_train)

    auc = roc_auc_score(
        y_test, model.predict_proba(X_test_scaled)[:, 1]
    )

    dump(model, MODEL_DIR / "credit_model.pkl")
    dump(scaler, MODEL_DIR / "scaler.pkl")
    dump(X.columns.tolist(), MODEL_DIR / "feature_columns.pkl")

    print("✅ Model trained successfully")
    print(f"📈 ROC-AUC Score: {auc:.3f}")