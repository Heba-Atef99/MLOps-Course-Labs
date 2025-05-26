"""
This module trains a RandomForestClassifier for bank customer churn prediction.
"""
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from mlflow.models.signature import infer_signature
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def rebalance(data):
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    churn_maj, churn_min = (churn_0, churn_1) if len(churn_0) > len(churn_1) else (churn_1, churn_0)
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )
    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    filter_feat = [
        "CreditScore", "Geography", "Gender", "Age", "Tenure",
        "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1912)

    col_transf = make_column_transformer(
        (StandardScaler(), num_cols),
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough"
    )

    X_train = pd.DataFrame(col_transf.fit_transform(X_train), columns=col_transf.get_feature_names_out())
    X_test = pd.DataFrame(col_transf.transform(X_test), columns=col_transf.get_feature_names_out())

    return col_transf, X_train, X_test, y_train, y_test


def train(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Churn Prediction Experiment")

    with mlflow.start_run(run_name="Random Forest Run"):
        df = pd.read_csv("dataset/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

        model = train(X_train, y_train)
        joblib.dump(model, r"Models\random_forest.pkl")
        y_pred = model.predict(X_test)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 6)

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

        mlflow.set_tag("developer", "Rowaina")
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="RandomForestChurnModel",
            signature=signature
        )

        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=model.classes_)
        conf_disp.plot()
        plt.savefig("confusion_matrix_rf.png")
        mlflow.log_artifact("confusion_matrix_rf.png")
        plt.show()


if __name__ == "__main__":
    main()
