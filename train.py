#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Bank Customer Churn Prediction with MLflow Tracking
"""

import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Test_Experiment")

with mlflow.start_run():
    mlflow.log_param("test_param", 123)
    mlflow.log_metric("test_metric", 0.95)
    print("Test run completed!")
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score
)
from pathlib import Path
import logging
import sys

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.3
DATA_DIR = Path(r"C:\Users\fayro\MLOps-Course-Labs\data")  # Raw string for Windows path
DATA_FILE = "Churn_Modelling.csv"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def load_data():
    """Load and validate raw data"""
    try:
        df = pd.read_csv(DATA_DIR / DATA_FILE)
        logging.info("Data loaded successfully. Shape: %s", df.shape)
        
        # Validate critical columns
        required_cols = ["CreditScore", "Geography", "Gender", "Exited"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        return df
    except Exception as e:
        logging.error("Failed to load data: %s", str(e))
        raise

def rebalance_data(df):
    """Balance class distribution"""
    try:
        churn_0 = df[df["Exited"] == 0]
        churn_1 = df[df["Exited"] == 1]
        
        majority, minority = (churn_0, churn_1) if len(churn_0) > len(churn_1) else (churn_1, churn_0)
        majority_downsampled = resample(
            majority,
            n_samples=len(minority),
            replace=False,
            random_state=RANDOM_STATE
        )
        
        balanced_df = pd.concat([majority_downsampled, minority])
        logging.info("Class balance after resampling:\n%s", balanced_df["Exited"].value_counts())
        return balanced_df
    except Exception as e:
        logging.error("Failed to balance data: %s", str(e))
        raise

def preprocess_data(df):
    """Feature engineering and preprocessing"""
    try:
        # Feature selection
        features = [
            "CreditScore", "Geography", "Gender", "Age", "Tenure",
            "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
            "EstimatedSalary"
        ]
        target = "Exited"
        
        # Validate features
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
            
        X = df[features]
        y = df[target]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # Preprocessing pipeline
        numeric_features = [f for f in features if f not in ["Geography", "Gender"]]
        categorical_features = ["Geography", "Gender"]
        
        preprocessor = make_column_transformer(
            (StandardScaler(), numeric_features),
            (OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features),
            remainder="passthrough",
        )
        
        # Apply transformations
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
        
        # Get feature names
        feature_names = (
            numeric_features + 
            list(preprocessor.named_transformers_['onehotencoder'].get_feature_names_out(categorical_features))
        )
        
        logging.info("Preprocessing completed. Feature count: %d", len(feature_names))
        return preprocessor, X_train, X_test, y_train, y_test, feature_names
        
    except Exception as e:
        logging.error("Preprocessing failed: %s", str(e))
        raise

def train_model(X_train, y_train, model_type="logistic"):
    """Model training with automatic logging"""
    try:
        if model_type == "logistic":
            model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                solver='lbfgs'
            )
            mlflow.log_param("model_type", "logistic_regression")
        else:
            model = RandomForestClassifier(
                n_estimators=200,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                max_depth=5
            )
            mlflow.log_param("model_type", "random_forest")
        
        model.fit(X_train, y_train)
        logging.info("%s model trained successfully", model_type.upper())
        return model
    except Exception as e:
        logging.error("Model training failed: %s", str(e))
        raise

def evaluate_model(model, X_test, y_test, feature_names=None):
    """Comprehensive model evaluation"""
    try:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            display_labels=["Retained", "Churned"],
            cmap="Blues",
            ax=ax
        )
        plt.title("Confusion Matrix")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
        
        # Feature importance (for tree-based models)
        if hasattr(model, "feature_importances_") and feature_names:
            importance = pd.DataFrame({
                "feature": feature_names,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            importance.plot.bar(x="feature", y="importance", ax=ax)
            plt.title("Feature Importance")
            plt.xticks(rotation=45)
            plt.tight_layout()
            mlflow.log_figure(fig, "feature_importance.png")
            plt.close(fig)
            
        return metrics
    except Exception as e:
        logging.error("Evaluation failed: %s", str(e))
        raise

def main():
    """Main execution flow"""
    # Initialize MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Bank_Churn_Prediction")
    
    try:
        with mlflow.start_run():
            # Load and log data
            df = load_data()
            mlflow.log_param("dataset_shape", df.shape)
            mlflow.log_param("test_size", TEST_SIZE)
            
            # Data preprocessing
            balanced_df = rebalance_data(df)
            preprocessor, X_train, X_test, y_train, y_test, feature_names = preprocess_data(balanced_df)
            
            # Model training and evaluation
            for model_type in ["logistic", "random_forest"]:
                with mlflow.start_run(nested=True, run_name=model_type):
                    model = train_model(X_train, y_train, model_type)
                    metrics = evaluate_model(model, X_test, y_test, feature_names)
                    
                    # Log model
                    mlflow.sklearn.log_model(
                        model,
                        artifact_path=f"{model_type}_model",
                        registered_model_name=f"Churn_{model_type}",
                        input_example=X_test[:1],
                        signature=mlflow.models.infer_signature(X_test, y_test)
                    )
                    
                    # Tag best model
                    if metrics["f1"] > 0.5:  # Adjust threshold as needed
                        mlflow.set_tag("best_model", "true")
            
            logging.info("Pipeline executed successfully")
    except Exception as e:
        logging.error("Pipeline failed: %s", str(e))
        raise

if __name__ == "__main__":
    main()

