import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import mlflow.sklearn

def rebalance(data):
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    churn_min = min(len(churn_0), len(churn_1))
    churn_maj_downsample = resample(
        churn_0 if len(churn_0) > len(churn_1) else churn_1,
        n_samples=churn_min,
        replace=False,
        random_state=1234,
    )
    churn_min_df = churn_1 if len(churn_0) > len(churn_1) else churn_0
    return pd.concat([churn_maj_downsample, churn_min_df])

def preprocess(df):
    filter_feat = [
        "CreditScore", "Geography", "Gender", "Age", "Tenure",
        "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
        "EstimatedSalary", "Exited",
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary",
    ]

    data_bal = rebalance(df[filter_feat])
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1912)

    col_transf = make_column_transformer(
        (StandardScaler(), num_cols),
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_test = col_transf.transform(X_test)

    mlflow.sklearn.log_model(col_transf, "preprocessor")

    return col_transf, pd.DataFrame(X_train, columns=col_transf.get_feature_names_out()), \
           pd.DataFrame(X_test, columns=col_transf.get_feature_names_out()), y_train, y_test
