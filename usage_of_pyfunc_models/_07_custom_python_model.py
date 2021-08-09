""" Build a custom pyfunc model including the preprocessing step.
"""
import mlflow
from config import mlflow_server_uri
from sklearn.ensemble import RandomForestRegressor
from utils import load_airbnb_data, preprocess_airbnb_data
from mlflow.models.signature import infer_signature
import cloudpickle


class MyCustomModel(mlflow.pyfunc.PythonModel):

    def __init__(self, trained_rf):
        self.rf = trained_rf

    def preprocess_input(self, model_input):
        return preprocess_airbnb_data(model_input)

    def predict(self, context, model_input):
        processed_model_input = self.preprocess_input(model_input.copy())
        return self.rf.predict(processed_model_input)


if __name__ == "__main__":

    # Connect to mlflow server
    mlflow.set_tracking_uri(mlflow_server_uri)
    mlflow.set_registry_uri(mlflow_server_uri)

    # Load data and preprocess
    X_train, X_test, y_train, y_test = load_airbnb_data()

    # Preprocess
    X_train_processed = preprocess_airbnb_data(X_train)

    # Init rf and fit
    rf = RandomForestRegressor(n_estimators=100, max_depth=25)
    rf.fit(X_train_processed, y_train)

    # Create customized model: RF incl preprocessing
    my_custom_model = MyCustomModel(trained_rf=rf)
    mlflow.set_experiment("Airbnb Predictor")

    # Save the conda environment for this model.
    conda_env = {
        'channels': ['defaults', 'conda-forge'],
        'dependencies': [
            'python=python=3.8.5',
            'pip'],
        'pip': [
            'mlflow',
            'cloudpickle=={}'.format(cloudpickle.__version__),
            'scikit-learn==0.24.1'
        ],
        'name': 'mlflow-env'
    }

    signature = infer_signature(X_test, y_test)

    mlflow.pyfunc.log_model(python_model=my_custom_model,
                            artifact_path="rf_inc_pp",
                            registered_model_name="My_airbnb_model",
                            # signature=signature,
                            conda_env=conda_env)




