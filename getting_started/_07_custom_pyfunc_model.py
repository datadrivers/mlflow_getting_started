""" Build a custom pyfunc model including the preprocessing step.
"""
import mlflow
from config import mlflow_server_uri
from sklearn.ensemble import RandomForestRegressor
from utils import load_airbnb_data
import cloudpickle


class MyCustomModel(mlflow.pyfunc.PythonModel):

    def __init__(self, trained_rf):
        self.rf = trained_rf

    def preprocess_input(self, model_input):

        model_input["trunc_lat"] = round(model_input["latitude"], 3)
        model_input["trunc_long"] = round(model_input["longitude"], 3)
        model_input["review_scores_sum"] = (
                model_input['review_scores_accuracy'] +
                model_input['review_scores_cleanliness'] +
                model_input['review_scores_checkin'] +
                model_input['review_scores_communication'] +
                model_input['review_scores_location'] +
                model_input['review_scores_value']
        )
        model_input = model_input.drop(["latitude", "longitude"], axis=1)
        return model_input

    def predict(self, context, model_input):
        processed_model_input = self.preprocess_input(model_input.copy())
        return self.rf.predict(processed_model_input)


if __name__ == "__main__":

    # Connect to mlflow server
    mlflow.set_tracking_uri(mlflow_server_uri)
    mlflow.set_registry_uri(mlflow_server_uri)

    # Load data and preprocess
    X_train, X_test, y_train, y_test = load_airbnb_data()

    """ Do some preprocessing """
    cols_to_drop = ["latitude", "longitude"]

    X_train_processed = X_train.copy()
    X_train_processed["trunc_lat"] = round(X_train["latitude"], 3)
    X_train_processed["trunc_long"] = round(X_train["longitude"], 3)
    X_train_processed["review_scores_sum"] = (
            X_train['review_scores_accuracy'] +
            X_train['review_scores_cleanliness'] +
            X_train['review_scores_checkin'] +
            X_train['review_scores_communication'] +
            X_train['review_scores_location'] +
            X_train['review_scores_value']
    )
    X_train_processed = X_train_processed.drop(cols_to_drop, axis=1)

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

    mlflow.pyfunc.log_model(python_model=my_custom_model,
                            artifact_path="rf_inc_pp",
                            registered_model_name="My_airbnb_model",
                            conda_env=conda_env)




