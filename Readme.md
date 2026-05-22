# 💳 Fraud Detection System using Machine Learning

## 📌 Project Overview

This project focuses on building a **fraud detection system** using machine learning on a highly imbalanced dataset of **6.3 million transactions**. The goal is to identify fraudulent transactions while minimizing the risk of missing actual fraud cases.

---
 
## 🖥️ Live Demo
 
> 🔗 [fastapi-app-url] *(https://transaction-fraud-detection-api.onrender.com/docs)*
 
---

---

## Objective

* Detect fraudulent transactions with **high recall**
* Handle **class imbalance effectively**
* Build a model that is **realistic and production-ready**

---

## Dataset

* Total records: **6.3 million**
* Fraud cases: **~0.13% (highly imbalanced)**

---

## Key Steps

### 🔹 1. Data Preprocessing

* Handled missing values
* Converted categorical variables using one-hot encoding
* Performed train-test split with stratification

---

### 🔹 2. Feature Engineering

Created meaningful features such as:

* `amount_to_balance_ratio`
* `hourly_velocity`
* `amount_deviation`
* `is_night`, `high_amount`

⚠️ Removed leakage-prone features to ensure model generalization.

---

### 🔹 3. Model Selection

Tested multiple models:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost ✅ (Final Model)

---

### 🔹 4. Handling Imbalance

* Used `scale_pos_weight` in XGBoost
* Focused on **Recall as primary metric**

---

### 🔹 5. Threshold Tuning

Adjusted probability threshold to **0.9** to balance:

* High recall (catch frauds)
* Acceptable precision

---

## 📈 Final Results

### 🔥 Classification Report

| Metric            | Value      |
| ----------------- | ---------- |
| Recall (Fraud)    | **~0.997** |
| Precision (Fraud) | **~0.52**  |
| F1 Score          | **~0.69**  |

---

### 🔍 Confusion Matrix

* True Positives: **1639**
* False Negatives: **4** ✅ (very low)
* False Positives: **1483**

---

### Key Insight

> The model successfully captures almost all fraud cases while maintaining a manageable number of false positives.

---

## Feature Importance Insights

Top features:

* `amount_to_balance_ratio` (strongest signal)
* `hourly_velocity` (burst fraud detection)
* Transaction types (`TRANSFER`, `CASH_OUT`)
* Time-based features (`hour`, `day`)

---

## Business Trade-off

* ✅ High Recall → Minimal financial loss
* ⚠️ Moderate Precision → Some false alerts (acceptable in banking)

---

## Model Saving

```python
joblib.dump(model, "fraud_model.pkl")
```

---

## 🚀 Deployment

### 🔹 FastAPI REST API

The trained model is deployed as a REST API using **FastAPI**, enabling easy integration with other services.

**Run the API:**

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 10000 --workers 2
```

**Sample prediction endpoint:**

```http
POST /predict
Content-Type: application/json

{
    'step': 1,
    'type': 'TRANSFER',
    'amount': 181.0,
    'nameOrig': 'C1305486145',
    'oldbalanceOrg': 181.0,
    'newbalanceOrig': 0.0,
    'nameDest': 'C553264065',
    'oldbalanceDest': 0.0,
    'newbalanceDest': 0.0,
    'isFlaggedFraud': 0
}
```

**Response:**

```json
{
    'fraud_probability': 0.9732,
    'is_fraud': True, 
    'threshold_used': 0.9
}
```

---

### 🔹 Real-Time Fraud Detection

The system supports **real-time transaction scoring** — each incoming transaction is evaluated instantly against the trained XGBoost model via the FastAPI endpoint.

**How it works:**

1. Transaction data is received by the API
2. Features are engineered on-the-fly (e.g., `amount_to_balance_ratio`, `hourly_velocity`)
3. The model predicts fraud probability
4. If probability exceeds the **0.9 threshold**, the transaction is flagged as fraudulent
5. Result is returned in milliseconds for downstream action (block, alert, review)

This enables seamless integration into banking pipelines, payment gateways, or monitoring dashboards.

---

## 🗂️ Project Structure

```
fraud-detection/
│
├── src/
│   └── api/
│       └── app.py          # FastAPI application
├── data/                   # Dataset (not included due to size)
├── notebooks/              # EDA and model experimentation
├── fraud_pipeline.pkl         # Trained XGBoost model
├── start.sh                # Server startup script
├── requirements.txt        # Dependencies
└── README.md
```

---

## ⚙️ Installation & Usage

```bash
# Clone the repository
git clone https://github.com/Prakruthisb/financial-fraud-detection-ml
cd fraud-detection

# Install dependencies
pip install -r requirements.txt

# Start the API
bash start.sh
```

---

## Conclusion

This project demonstrates:

* Handling large-scale imbalanced data
* Avoiding data leakage
* Making business-driven ML decisions
* Deploying ML models with FastAPI
* Building a real-time fraud detection pipeline

---

## 🔮 Future Improvements

* Improve precision using advanced features
* ✅ ~~Add a monitoring dashboard for flagged transactions~~ → Built! See [Fraud Detection Monitor](https://github.com/Prakruthisb/financial-fraud-detection-streamlit) — a Streamlit dashboard with live flagged transactions, model monitoring and drift detection, and SHAP explanations
* Integrate with streaming platforms (e.g., Kafka) for high-throughput pipelines

---

⭐ If you like this project, consider giving it a star!