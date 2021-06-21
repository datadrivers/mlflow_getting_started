""" An motivational example
"""
import mlflow
from config import mlflow_server_uri
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from utils import load_airbnb_data

if __name__ == "__main__":

    # Connect to mlflow server
    mlflow.set_tracking_uri(mlflow_server_uri)
    mlflow.set_registry_uri(mlflow_server_uri)

    # Load data
    X_train, X_test, y_train, y_test = load_airbnb_data()
    rf = RandomForestRegressor(n_estimators=100, max_depth=25)

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

    X_test_processed = X_test.copy()
    X_test_processed["trunc_lat"] = round(X_test["latitude"], 3)
    X_test_processed["trunc_long"] = round(X_test["longitude"], 3)
    X_test_processed["review_scores_sum"] = (
            X_test['review_scores_accuracy'] +
            X_test['review_scores_cleanliness'] +
            X_test['review_scores_checkin'] +
            X_test['review_scores_communication'] +
            X_test['review_scores_location'] +
            X_test['review_scores_value']
    )
    X_test_processed = X_test_processed.drop(cols_to_drop, axis=1)

    # Fit and evaluate the rf
    rf.fit(X_train_processed, y_train)
    rf_mse = mean_squared_error(y_test, rf.predict(X_test_processed))

    print(f"RF fitted. MSE is {rf_mse}.")

    mlflow.set_experiment("Airbnb Predictor")
    with mlflow.start_run(run_name="RF Model") as run:
        mlflow.sklearn.log_model(rf,
                                 "rf",
                                 registered_model_name="MyModel02")
        mlflow.log_metric("mse", rf_mse)

