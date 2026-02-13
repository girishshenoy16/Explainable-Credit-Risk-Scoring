# ⚖️ Fairness & Bias Report

## 1. Objective

The objective of fairness monitoring is to ensure that the credit decision system does not disproportionately disadvantage any protected group.

---

## 2. Sensitive Attribute

* Gender (Male / Female)

Gender is **not used to influence individual loan decisions**.
It is used **only for post-decision monitoring**.

---

## 3. Fairness Metric Used

**Approval Rate Parity**

Approval Rate = Approved Applications / Total Applications

This metric is widely used in financial risk governance to detect potential bias.

---

## 4. Results (Sample)

* Male approval rate: ~70%
* Female approval rate: ~77%
* Disparity: ~6–7%

These values are monitored over time rather than evaluated in isolation.

---

## 5. Interpretation

* A small disparity may arise due to data characteristics
* Persistent or increasing disparity would trigger review

Possible actions include:

* Feature review
* Threshold recalibration
* Policy adjustments

---

## 6. Governance & Compliance

* Fairness metrics are logged regularly
* Results are reviewed by risk governance teams
* Corrective action is taken if thresholds are breached

---

## 7. Conclusion

Fairness monitoring ensures that the system operates responsibly and aligns with ethical and regulatory expectations.