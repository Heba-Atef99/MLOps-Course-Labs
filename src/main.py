import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.preprocessing import preprocess
from src.train_model import train
from src.visualize import log_confusion_matrix

def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("churn_prediction_experement")

    with mlflow.start_run():
        df = pd.read_csv("dataset/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)

        model = train(X_train, y_train)
        y_pred = model.predict(X_test)

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

        mlflow.set_tag("developer", "Mohamed Alaa")

        log_confusion_matrix(y_test, y_pred, model)

if __name__ == "__main__":
    main()
