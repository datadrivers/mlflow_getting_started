""" An motivational example
"""
import mlflow
from config import mlflow_server_uri
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from utils import load_airbnb_data, preprocess_airbnb_data

if __name__ == "__main__":

    # Connect to mlflow server
    mlflow.set_tracking_uri(mlflow_server_uri)
    mlflow.set_registry_uri(mlflow_server_uri)

    # Load data
    X_train, X_test, y_train, y_test = load_airbnb_data()
    rf = RandomForestRegressor(n_estimators=100, max_depth=25)

    # Preprocess df
    X_train_processed = preprocess_airbnb_data(X_train)
    X_test_processed = preprocess_airbnb_data(X_test)

    # Fit and evaluate the rf
    rf.fit(X_train_processed, y_train)
    rf_mse = mean_squared_error(y_test, rf.predict(X_test_processed))

    print(f"RF fitted. MSE is {rf_mse}.")

    mlflow.set_experiment("Airbnb Predictor")
    with mlflow.start_run(run_name="RF Model") as run:
        mlflow.sklearn.log_model(rf,
                                 "rf",
                                 registered_model_name="My_airbnb_model")
        mlflow.log_metric("mse", rf_mse)

