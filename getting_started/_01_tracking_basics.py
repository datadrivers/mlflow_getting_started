""" Just some very first steps.
"""
import mlflow
from config import mlflow_server_uri

if __name__ == "__main__":

    # Connect to mlflow server
    mlflow.set_tracking_uri(mlflow_server_uri)

    # Set Experiment
    mlflow.set_experiment("My Basic Experiment")

    with mlflow.start_run(run_name="My first run"):

        mlflow.log_param("my_parameter", 0.5)
        mlflow.log_metric("my_metric", 0.89)

        with open("sample_artifact.txt", 'w') as f:
            f.write("Some artifact content for demo purposes.")
        mlflow.log_artifact("sample_artifact.txt", artifact_path="prefix_in_folder")

