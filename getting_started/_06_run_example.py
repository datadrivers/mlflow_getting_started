""" Run model from registry
"""
import mlflow.pyfunc
from config import mlflow_server_uri
from utils import load_airbnb_data

model_name = "My_airbnb_model"
version = '1'

if __name__ == "__main__":

    # Connect to mlflow server
    mlflow.set_tracking_uri(mlflow_server_uri)
    mlflow.set_registry_uri(mlflow_server_uri)

    # Load model from registry
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{version}"
    )

    # Predict
    try:
        _, X_test, _, _ = load_airbnb_data()
        print(model.predict(X_test))
    except ValueError as e:
      print("ERROR: " + str(e))
