"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction using multiple models and MLflow.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import joblib
import mlflow
import mlflow.sklearn


def rebalance(data):
    """Resample data to keep balance between target classes."""
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )
    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """Preprocess and split data into training and test sets."""
    filter_feat = [
        "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary"
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols),
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )


    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())


    # Save the preprocessor for later use
    joblib.dump(
        col_transf, "model/preprocessor.joblib"
    )
    return col_transf, X_train, X_test, y_train, y_test


def train_model(X_train, y_train, model_type="logistic"):
    """Train a model based on model_type and log params."""
    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
        mlflow.log_param("max_iter", 1000)

    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 8)

    elif model_type == "xgboost":
        model = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss"
        )
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.1)

    else:
        raise ValueError("Unsupported model type")

    model.fit(X_train, y_train)
    mlflow.set_tag("model_type", model_type)
    return model


def run_experiment(model_type):
    """Run training, evaluation and MLflow logging for a given model."""
    with mlflow.start_run():
        df = pd.read_csv("dataset/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

        model = train_model(X_train, y_train, model_type=model_type)
        y_pred = model.predict(X_test)

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        disp = ConfusionMatrixDisplay(conf_mat, display_labels=model.classes_)
        disp.plot()
        plt.savefig(f"confusion_matrix_{model_type}.png")
        mlflow.log_artifact(f"confusion_matrix_{model_type}.png")
        plt.close()

        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"{model_type}_model",
            signature=signature,
            input_example=X_train.iloc[:5],
        )

        # Save XGBoost model as .pkl for API use
        if model_type == "xgboost":
            os.makedirs("model", exist_ok=True)
            joblib.dump(model, "model/Best_model_for_production_v1.pkl")
            print("XGBoost model saved to model/Best_model_for_production_v1.pkl")


if __name__ == "__main__":
    # Set up MLflow tracking and experiment name
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("churn-prediction")

    # Run all three experiments
    for model_name in ["logistic", "random_forest", "xgboost"]:
        run_experiment(model_name)