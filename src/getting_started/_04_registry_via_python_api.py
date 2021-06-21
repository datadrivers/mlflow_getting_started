""" Alternative to the UI: Using the python API
Further reading: https://www.mlflow.org/docs/latest/model-registry.html#model-registry-workflows
"""
import mlflow
from config import mlflow_server_uri
from mlflow.tracking import MlflowClient
import pickle

""" Register / Update / Change Stages """

# There are several ways to register a model.
# The recommended way is to log a model and register it via giving it a registered_model_name.

# mlflow.sklearn.log_model(my_fitted_object,
#                          "model_name",
#                          registered_model_name="MyModel02")

# Note that the fitting does not mandatory need to take place in the same environment as the logging/registration process.
# But one can also load a pickled model and safe it.


# with open("my_pickled_fitted_object.pkl", "rb") as file:
#     my_fitted_object = pickle.load(file)
#
# mlflow.sklearn.log_model(my_fitted_object,
#                          "model_name",
#                          registered_model_name="MyModel02")

# One can also register a specific run id
# result = mlflow.register_model(
#     "runs:/d16076a3ec534311817565e6527539c0/sklearn-model",
#     "MyModel02"
# )

# or use the MlflowClient to update already registred model versions and stages

""" Delete """
# mlflow.set_tracking_uri(mlflow_server_uri)
# mlflow.set_registry_uri(mlflow_server_uri)
#
# client = MlflowClient()
# client.delete_registered_model(name="MyModel01")