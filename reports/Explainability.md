# 🔍 Explainability Report

## 1. Purpose

Explainability is critical in credit risk systems to ensure:

* Transparency
* Regulatory compliance
* Customer trust

This project uses **SHAP (SHapley Additive exPlanations)** to explain model decisions.

---

## 2. Explainability Approach

* SHAP values are computed using a representative background dataset
* Explanations are generated for individual applicants
* Only **actionable, non-sensitive features** are shown

Sensitive attributes such as gender, marital status, and education are **excluded from user-facing explanations**.

---

## 3. Interpreting SHAP Values

* Positive SHAP value → increases default risk
* Negative SHAP value → decreases default risk

The magnitude indicates the strength of influence.

---

## 4. Reason Codes

For rejected or manual-review cases, SHAP outputs are mapped to **business-friendly reason codes**, such as:

* Recent payment delays
* High credit utilization
* Weak repayment behaviour

These reason codes align with industry-standard adverse action notices.

---

## 5. Governance Considerations

* Explanations are consistent across similar profiles
* Sensitive features are not used as decision justifications
* Explanations support audit and regulatory review

---

## 6. Conclusion

The explainability framework ensures that model decisions are **transparent, defensible, and understandable** by both technical and non-technical stakeholders.