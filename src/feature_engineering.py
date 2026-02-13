# src/feature_engineering.py

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/processed")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    bill_cols = [
        "BILL_AMT1","BILL_AMT2","BILL_AMT3",
        "BILL_AMT4","BILL_AMT5","BILL_AMT6"
    ]
    pay_cols = [
        "PAY_AMT1","PAY_AMT2","PAY_AMT3",
        "PAY_AMT4","PAY_AMT5","PAY_AMT6"
    ]

    df["avg_bill_amt"] = df[bill_cols].mean(axis=1)
    df["avg_pay_amt"] = df[pay_cols].mean(axis=1)

    # ✅ SAFE division
    df["payment_to_bill_ratio"] = np.where(
        df["avg_bill_amt"] > 0,
        df["avg_pay_amt"] / df["avg_bill_amt"],
        0
    )

    df["high_utilization"] = (
        df["avg_bill_amt"] > 0.7 * df["LIMIT_BAL"]
    ).astype(int)

    delay_cols = ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
    df["has_delay"] = (df[delay_cols] > 0).any(axis=1).astype(int)

    # ✅ FINAL safety net
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df


if __name__ == "__main__":
    input_path = PROCESSED_DIR / "cleaned_data.csv"
    output_path = PROCESSED_DIR / "features.csv"

    df = pd.read_csv(input_path)
    df_features = engineer_features(df)

    df_features.to_csv(output_path, index=False)

    print("✅ Feature engineering completed (NaN-safe)")
    print(f"📄 Saved to {output_path}")