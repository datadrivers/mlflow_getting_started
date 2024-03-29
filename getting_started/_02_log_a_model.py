""" Lets log a simple sklearn model
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
import mlflow.sklearn
from config import mlflow_server_uri

# Hyperparameter settings
max_depth = 2

if __name__ == '__main__':

    # Connect to mlflow server
    mlflow.set_tracking_uri(mlflow_server_uri)

    # Set Experiment
    mlflow.set_experiment("My Wine Classifier")

    with mlflow.start_run(run_name="My run"):

        # Load data
        X, y = load_wine(return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        # Train the model
        clf = DecisionTreeClassifier(max_depth=max_depth)
        clf.fit(x_train, y_train)

        # Evaluate
        y_pred = clf.predict(x_test)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        accuracy = accuracy_score(y_test, y_pred)

        # Log parameter and metrics
        mlflow.log_param("max_dept", max_depth)
        mlflow.log_metrics({
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy

        })

        # Log the model
        mlflow.sklearn.log_model(sk_model=clf, artifact_path="my_model")

        # Save a model under a specific path
        # mlflow.sklearn.save_model(sk_model=clf,
        #                           path="../../cloud_mock/my_specific_bucket/my_specific_saved_model")

