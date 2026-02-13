# 🏦 Explainable Credit Risk Scoring & Approval System

## 📌 Overview

This project implements an **end-to-end, industry-oriented credit risk decision system** inspired by real banking and NBFC workflows. The system predicts the **probability of loan default**, converts model outputs into **business-friendly decisions**, provides **reason codes for rejections**, and monitors **fairness & bias metrics** for governance.

The project is designed to be:

* Fully runnable on a local machine
* Explainable and auditable
* Easy to discuss in technical and business interviews

---

## 🎯 Business Problem

Banks and fintech lenders must balance **risk control** with **customer growth**. Approving high-risk customers leads to losses, while rejecting too many low-risk customers hurts revenue.

This system helps answer:

* *Should this loan be approved, reviewed, or rejected?*
* *Why was this decision made?*
* *Is the model behaving fairly across demographic groups?*

---

## 🧠 Key Features

* Credit default probability prediction (Logistic Regression)
* Three-tier decision policy:

  * ✅ Auto-Approve
  * ⚠️ Manual Review
  * ❌ Auto-Reject
* Explainability using SHAP
* Bank-style **reason codes** for adverse decisions
* Fairness monitoring by gender (approval rate parity)
* Streamlit-based interactive dashboard

---

## 🧰 Tech Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, SHAP, Fairlearn
* **Frontend:** Streamlit
* **Model:** Logistic Regression (interpretable & regulator-friendly)
* **Dataset:** UCI Credit Default Dataset

---

## 📁 Project Structure

```
Explainable-Credit-Risk-Scoring/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── dashboard/
│   └── app.py
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── explain_model.py
│   └── fairness_analysis.py
│
├── models/
│   ├── credit_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── reports/
│   ├── Model_Performance.md
│   ├── Explainability.md
│   └── Fairness_Bias.md
│
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run the Project

```bash
# Clone repository:
git clone https://github.com/girishshenoy16/Explainable-Credit-Risk-Scoring.git

cd Explainable-Credit-Risk-Scoring


# 1. Create virtual environment
python -m venv venv

# 2. Activate environment
venv\Scripts\activate

# 3. Install dependencies
python.exe -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Run preprocessing and training
python src/data_preprocessing.py
python src/feature_engineering.py
python src/train_model.py
python src/evaluate_model.py
python src/explain_model.py
python src/fairness_analysis.py

# 5. Launch the dashboard
streamlit run .\dashboard\app.py
```

---

## 📊 Outputs

* Probability of default for each applicant
* Clear loan decision (Approve / Review / Reject)
* Human-readable reason codes for rejections
* Fairness dashboard showing approval rates by gender

---

## 🚀 Future Improvements

* Threshold optimization using cost-sensitive learning
* Deployment on cloud (AWS / Azure)
* Addition of more fairness metrics
* Audit logs and monitoring dashboards

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. Real-world credit systems involve additional policy rules, manual reviews, and regulatory approvals.
