# Churn Prediction with MLflow Tracking

## 📌 Project Overview

This project predicts bank customer churn using logistic regression and tracks experiments using **MLflow**. It includes proper data preprocessing, class balancing, and performance evaluation through multiple metrics and visualizations.

-------

## 🚀 Setup Instructions

1. **Create and activate a virtual environment**:
    ```bash
    python -m venv churn_prediction
    churn_prediction\Scripts\activate  # On Windows
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run MLflow UI (in a new terminal)**:
    ```bash
    mlflow ui
    ```

4. **Train and track experiments**:
    ```bash
    python -m src.main
    ```
## 📦 Files Overview      (Bounus Part)
    - src/preprocessing.py: Data cleaning, encoding, scaling, and class balancing
    
    - src/train_model.py: Contains the training logic for logistic regression
    
    - src/visualize.py: Function to generate and log confusion matrix as image
    
    - src/main.py: The orchestrator script that brings everything together and runs MLflow experiment
    
    - requirements.txt: All dependencies for reproducibility
---------


