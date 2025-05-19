# 🏦 Bank Customer Churn Prediction

This project builds an MLOps-ready pipeline to predict customer churn for a bank using various machine learning models. It includes data preprocessing, class balancing, model training, evaluation, and MLflow-based experiment tracking and model versioning.

---

## 📁 Project Structure

```

.
├── data/                     # Input dataset
├── src/
│   ├── data.py              # Preprocessing and balancing functions
│   ├── train.py             # Model training and experiment logging
│   ├── registry.py          # Model registration logic
│   └── utils.py             # Helpers like plotting/logging
├── requirements.txt
├── README.md
└── mlruns/                  # MLflow tracking directory

````

---

## 🔧 Setup

### 1. Clone the Repo

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
````

### 2. Create Environment

```bash
conda create -n churn-ml python=3.12 -y
conda activate churn-ml
pip install -r requirements.txt
```

---

## 🚀 Running the Pipeline

### 1. Train Models and Log Experiments

```bash
python src/train.py
```

### 2. View MLflow UI

```bash
mlflow ui
```

Open in browser: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧠 Models Trained

* Logistic Regression
* Random Forest
* XGBoost

Each model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Confusion matrices are saved and logged as MLflow artifacts.

---

🗂 Model Registry
After training, the top 2 models are automatically registered with MLflow and promoted to appropriate stages.

✅ Model Promotion
Model	Stage	Justification
RandomForest_model	-> Production ->	Best overall -> performance (F1 and accuracy), reliable and interpretable
XGBoost_model	-> Staging ->	Competitive performance -> ideal backup with high recall capability

To register models:

```bash
# Inside train.py or separately
from src.registry import register_top_models
register_top_models()
```

