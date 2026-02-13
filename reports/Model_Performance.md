# 📊 Model Performance Report

## 1. Objective

The objective of this model is to estimate the **probability of loan default** for credit applicants and support automated and semi-automated lending decisions.

---

## 2. Dataset

* Source: UCI Credit Default Dataset
* Records: ~30,000 customers
* Target Variable: `default` (1 = default, 0 = non-default)

Key features include:

* Credit limit
* Repayment history
* Bill amounts
* Payment amounts

---

## 3. Model Selection

**Logistic Regression** was chosen due to:

* High interpretability
* Regulatory acceptance in banking
* Stable performance
* Compatibility with explainability techniques

Class imbalance was handled using `class_weight="balanced"`.

---

## 4. Training Setup

* Train/Test split: 80% / 20%
* Feature scaling: StandardScaler
* Categorical encoding: One-hot encoding

---

## 5. Evaluation Metrics

* ROC-AUC: ~0.74–0.78 (varies slightly by run)
* Confusion Matrix: Reviewed to assess false approvals and rejections

The model prioritizes **risk differentiation** over raw accuracy, which is appropriate for credit risk use cases.

---

## 6. Decision Thresholds

A three-tier decision policy is applied:

* **PD < 40%** → Auto-Approve
* **40% ≤ PD < 70%** → Manual Review
* **PD ≥ 70%** → Auto-Reject

This mirrors real-world lending workflows.

---

## 7. Limitations

* Model is trained on historical data only
* Does not include macroeconomic indicators
* Thresholds are illustrative and not optimized for profit

---

## 8. Conclusion

The model demonstrates strong discriminatory power and is suitable as a **decision-support tool** when combined with policy rules and human oversight.