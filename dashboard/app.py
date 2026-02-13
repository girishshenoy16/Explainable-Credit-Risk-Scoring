# src/app.py

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import shap
from fairlearn.metrics import MetricFrame, selection_rate

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Explainable Credit Risk Scoring",
    layout="wide"
)

st.title("🏦 Explainable Credit Risk Scoring System")
st.markdown(
    """
This application simulates a **bank credit risk decision system**.
It predicts **loan default probability**, provides **clear reason codes for rejections**,
and displays **fairness metrics for monitoring purposes only**.
"""
)

# --------------------------------------------------
# Load trained artifacts
# --------------------------------------------------
model = load("models/credit_model.pkl")
scaler = load("models/scaler.pkl")
feature_cols = load("models/feature_columns.pkl")

RISK_THRESHOLD = 0.65  # business-defined cutoff

# --------------------------------------------------
# Sidebar – Applicant Inputs
# --------------------------------------------------
st.sidebar.header("📋 Applicant Details")

LIMIT_BAL = st.sidebar.number_input("Credit Limit", 10000, 1000000, 120000, step=10000)
AGE = st.sidebar.slider("Age", 21, 75, 35)
PAY_0 = st.sidebar.slider("Most Recent Payment Delay (months)", -1, 8, 2)
avg_bill_amt = st.sidebar.number_input("Average Bill Amount", 0, 500000, 100000)
avg_pay_amt = st.sidebar.number_input("Average Payment Amount", 0, 500000, 500000)

# --------------------------------------------------
# Feature engineering (same as training)
# --------------------------------------------------
payment_to_bill_ratio = avg_pay_amt / (avg_bill_amt + 1)
high_utilization = int(avg_bill_amt > 0.7 * LIMIT_BAL)
has_delay = int(PAY_0 > 0)

input_df = pd.DataFrame([{
    "LIMIT_BAL": LIMIT_BAL,
    "AGE": AGE,
    "PAY_0": PAY_0,
    "avg_bill_amt": avg_bill_amt,
    "avg_pay_amt": avg_pay_amt,
    "payment_to_bill_ratio": payment_to_bill_ratio,
    "high_utilization": high_utilization,
    "has_delay": has_delay
}])

input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=feature_cols, fill_value=0)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
scaled_input = scaler.transform(input_df)
default_prob = model.predict_proba(scaled_input)[0][1]


if default_prob < 0.40:
    decision = "Approved"
elif default_prob < 0.70:
    decision = "Manual Review"
else:
    decision = "Rejected"


# --------------------------------------------------
# Decision display
# --------------------------------------------------
st.subheader("📌 Credit Risk Decision")

col1, col2 = st.columns(2)

with col1:
    st.metric("Probability of Default", f"{default_prob:.2%}")

with col2:
    if decision == "Approved":
        st.success("✅ Low Risk – Loan Approved")
    elif decision == "Manual Review":
        st.warning("⚠️ Medium Risk – Sent for Manual Review")
    else:
        st.error("❌ High Risk – Loan Rejected")

# --------------------------------------------------
# Decision explanation
# --------------------------------------------------
st.subheader("🧠 Decision Explanation")

if decision == "Approved":
    st.success(
        "Loan approved due to strong repayment behaviour and low historical risk."
    )

elif decision == "Manual Review":
    st.warning(
        "This application shows mixed risk signals and requires manual credit review."
    )

# --------------------------------------------------
# SHAP background setup
# --------------------------------------------------
background_df = pd.read_csv("data/processed/features.csv")

X_bg = pd.get_dummies(background_df.drop(columns=["default"]), drop_first=True)
X_bg = X_bg.reindex(columns=feature_cols, fill_value=0)
X_bg = scaler.transform(X_bg)

background_sample = X_bg[
    np.random.choice(X_bg.shape[0], 100, replace=False)
]

explainer = shap.LinearExplainer(
    model,
    background_sample,
    feature_perturbation="interventional"
)

shap_values = explainer.shap_values(scaled_input)

shap_df = pd.DataFrame({
    "feature": feature_cols,
    "shap_value": shap_values[0]
})

# Remove sensitive features from explanations
SENSITIVE = ["SEX", "MARRIAGE", "EDUCATION"]
shap_df = shap_df[~shap_df["feature"].isin(SENSITIVE)]
shap_df["abs"] = shap_df["shap_value"].abs()
shap_df = shap_df.sort_values("abs", ascending=False)

# --------------------------------------------------
# Reason Codes (ONLY FOR REJECTIONS)
# --------------------------------------------------
if decision in ["Rejected", "Manual Review"]:
    st.subheader("🧾 Reason Codes")

    reason_codes = []

    for _, row in shap_df.head(5).iterrows():
        if row["feature"] == "has_delay" and row["shap_value"] > 0:
            reason_codes.append("Recent payment delays")
        elif row["feature"] == "PAY_0" and row["shap_value"] > 0:
            reason_codes.append("Latest payment delinquency")
        elif row["feature"] == "payment_to_bill_ratio" and row["shap_value"] > 0:
            reason_codes.append("Weak repayment behaviour")
        elif row["feature"] == "high_utilization" and row["shap_value"] > 0:
            reason_codes.append("High credit utilization")

    if not reason_codes:
        reason_codes.append("Overall elevated credit risk")

    for rc in set(reason_codes):
        st.write(f"- {rc}")

# --------------------------------------------------
# FAIRNESS & BIAS MONITORING (SANITIZED UI)
# --------------------------------------------------
st.subheader("⚖️ Fairness & Bias Monitoring")

st.markdown(
    """
**Note:** The metrics below are used **only for monitoring model fairness** and **do not influence individual loan decisions**.
"""
)

# Map SEX codes to readable labels
sex_map = {1: "Male", 2: "Female"}
background_df["Gender"] = background_df["SEX"].map(sex_map)

X_full = pd.get_dummies(background_df.drop(columns=["default", "Gender"]), drop_first=True)
X_full = X_full.reindex(columns=feature_cols, fill_value=0)
X_full = scaler.transform(X_full)

approval_pred = (
    model.predict_proba(X_full)[:, 1] < RISK_THRESHOLD
).astype(int)

metric_frame = MetricFrame(
    metrics={"Approval Rate": selection_rate},
    y_true=approval_pred,
    y_pred=approval_pred,
    sensitive_features=background_df["Gender"]
)

fairness_df = metric_frame.by_group.reset_index()
fairness_df.columns = ["Gender", "Approval Rate"]

fairness_df_display = fairness_df.copy()
fairness_df_display["Approval Rate"] = (
    fairness_df_display["Approval Rate"] * 100
).round(2).astype(str) + " %"

st.dataframe(fairness_df_display, use_container_width=True)

rates = fairness_df["Approval Rate"]
disparity = rates.max() - rates.min()

st.markdown(f"Approval rate disparity: `{disparity:.2%}`")

st.markdown(f"""Disparities beyond internal thresholds would trigger governance review.""")

# --------------------------------------------------
# Disclaimer
# --------------------------------------------------
st.markdown("---")
st.caption(

    "Note: Even small recent payment delays can significantly increase credit risk."
    
    "⚠️ Educational demo. Real banking systems include policy rules, "
    "manual reviews, and regulatory governance."
)