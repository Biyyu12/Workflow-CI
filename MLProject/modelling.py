import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Loan Status Prediction")

data = pd.read_csv(r"C:\Users\Abiyy\ProjectML\MSML_Abiyyu-Rasyiq-Muhadzzib\Membangun_model\loan_data_preprocessed.csv")

X_train, X_test, y_train, y_test = train_test_split(
    data.drop("loan_status", axis=1),
    data["loan_status"],
    random_state=42,
    test_size=0.2,
    stratify=data["loan_status"]
)

input_example = X_train[0:5]

with mlflow.start_run():
    
    # Log parameters
    n_estimators = 100
    learning_rate = 0.1
    max_depth = 5

    mlflow.autolog()

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth
    )

    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        input_example=input_example
    )

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)