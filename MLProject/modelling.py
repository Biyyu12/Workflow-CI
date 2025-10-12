import mlflow
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os
import warnings
import sys
import joblib

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    max_depth = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    file_path = os.path.abspath(sys.argv[4]) if len(sys.argv) > 4 else os.path.join(os.path.dirname(__file__), "loan_data_preprocessed.csv")

    print(f"Current working directory: {os.getcwd()}")
    print(f"Loading data from: {file_path}")

    data = pd.read_csv(file_path)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("loan_status", axis=1),
        data["loan_status"],
        random_state=42,
        test_size=0.2,
        stratify=data["loan_status"]
    )

    input_example = X_train[0:5]

    with mlflow.start_run():
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Log predictions for evaluation
        y_pred = model.predict(X_test)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred, average='weighted'))
        mlflow.log_metric("recall", recall_score(y_test, y_pred, average='weighted'))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average='weighted'))
        
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("max_depth", max_depth)

        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        # Save model to artifacts directory (relative path from MLProject/)
        model_dir = "model_artifacts"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pkl")
        
        joblib.dump(model, model_path)
        print(f"Model saved to: {os.path.abspath(model_path)}")
        
        # Verify file exists
        if os.path.exists(model_path):
            print(f"✓ Model file verified at: {model_path}")
            print(f"  File size: {os.path.getsize(model_path)} bytes")
        else:
            print(f"✗ ERROR: Model file not found at: {model_path}")

