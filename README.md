# MLOps Course Labs

Welcome to the lab repository for the [MLOps Course](https://github.com/Heba-Atef99/MLOps-Course).

Throughout this hands-on journey, you'll develop a **Bank Customer Churn Prediction** application—starting from the research phase and progressing through the full MLOps lifecycle, all the way to deployment.

> **Note:** Currently, the repository contains only the `research` branch. The remaining branches will be built step by step by the reader during the course days, as part of the learning experience.
=======
# 💼 Bank Customer Churn Prediction – MLOps Lab

This project applies machine learning and MLOps best practices to predict customer churn for a bank using MLflow for experiment tracking, evaluation, and model registry.

---

## 📊 Problem Statement

The goal is to predict whether a customer will leave the bank (`Exited = 1`) or remain (`Exited = 0`) based on demographic and account-related features.

---

## 📁 Dataset

- **Source:** [Kaggle – Bank Customer Churn Prediction](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)
- **Filename:** `Churn_Modelling.csv`
- **Rows:** 10,000 customers
- **Target Column:** `Exited` (1 = churned, 0 = stayed)

---

## 🧰 Tools & Technologies

- **Python 3.12**
- `pandas`, `scikit-learn`, `xgboost`, `matplotlib`
- **MLflow** for experiment tracking and model registry
- Environment: `conda` or `venv` named `churn_prediction`

---

## 📂 Project Structure

```bash
MLOps-Course-Labs/
│
├── dataset/
│   └── Churn_Modelling.csv
│
├── src/
│   ├── train.py           # Logistic Regression model
│   ├── train_rf.py        # Random Forest Classifier
│   └── train_xgb.py       # XGBoost Classifier
│
├── confusion_matrix.png       # Logistic regression confusion matrix
├── confusion_matrix_rf.png    # Random Forest confusion matrix
├── confusion_matrix_xgb.png   # XGBoost confusion matrix
│
└── README.md             # This file
```
## 🧪 Experiments Tracked with MLflow

All models were trained and evaluated using **MLflow**.  
Experiments were logged with:

- **Hyperparameters** via `mlflow.log_param()`
- **Metrics** via `mlflow.log_metric()`
- **Confusion matrix plots** via `mlflow.log_artifact()`
- **Developer information** via run tags

---

## 📊 Model Performance Comparison

| Model                | Accuracy | F1 Score | Precision |
|---------------------|----------|----------|-----------|
| Logistic Regression | 0.71     | 0.70     | 0.71      |
| Random Forest       | 0.77     | 0.76     | 0.78      |
| XGBoost             | 0.76     | 0.75     | 0.76      |

---

## ✅ Model Registry & Justification

### 🏁 Production Model

- **Name:** `RandomForestChurnModel`
- **Version:** 3
- **Why:** Achieved the highest scores in:
  - Accuracy: **0.77**
  - F1-score: **0.76**
  - Precision: **0.78**

✅ Selected as the **Production** model for deployment due to consistent and strong results.

---

### 🧪 Staging Model

- **Name:** `XGBoostChurnModel`
- **Version:** 1
- **Why:** Close performance to Random Forest:
  - F1-score: **0.75**
  - Precision: **0.76**

🧪 Promoted to **Staging** for future tuning and experimentation.

---

## 📦 MLflow Artifacts

- **Metrics:**
  - `accuracy`, `precision`, `recall`, `f1_score`

- **Parameters:**
  - `model_type`, `n_estimators`, `max_depth`, etc.

- **Artifacts:**
  - `confusion_matrix.png` – Logistic Regression
  - `confusion_matrix_rf.png` – Random Forest
  - `confusion_matrix_xgb.png` – XGBoost

- **Experiment name:**
  - `Churn Prediction Experiment`

---

## ▶️ How to Run Locally

### 1. Clone the Repository & Switch to `research` Branch

```bash
git clone https://github.com/YOUR_USERNAME/MLOps-Course-Labs.git
cd MLOps-Course-Labs
git checkout research
```
### 2. Setup Environment
```bash
# Create and activate the environment
conda create -n churn_prediction python=3.12
conda activate churn_prediction

# Or using venv
python3.12 -m venv churn_prediction
source churn_prediction/bin/activate  # On Linux/Mac
.\churn_prediction\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```
### 3. Run Training Scripts
```bash
python src/log_regression.py        
python src/random_forest.py     
python src/XGBoost.py   
```
### 4. Launch MLflow UI
```bash
mlflow ui
```
## 🙋 Author

- **Name:** Rowaina  
- **GitHub:** [@Raoina](https://github.com/Raoina)
