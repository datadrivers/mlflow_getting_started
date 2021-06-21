""" Just some very first steps.
"""
import mlflow
from config import mlflow_server_uri

if __name__ == "__main__":

    # Connect to mlflow server
    mlflow.set_tracking_uri(mlflow_server_uri)

    # Set Experiment
    mlflow.set_experiment("My Basic Experiment1")

    with mlflow.start_run(run_name="My first run"):

        mlflow.log_param("p1", 0.5)
        mlflow.log_metric("m1", 0.89)

        with open("sample_artifact.txt", 'w') as f:
            f.write("Some artifact content.")
        mlflow.log_artifact("sample_artifact.txt", artifact_path="prefix_in_folder")

