from sklearn.linear_model import LogisticRegression
import mlflow.sklearn
from mlflow.models.signature import infer_signature

def train(X_train, y_train):
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    signature = infer_signature(X_train, log_reg.predict(X_train))
    mlflow.sklearn.log_model(log_reg, "model", signature=signature)

    return log_reg
