""" Just some utils
"""
import mlflow
import pandas as pd
from sklearn.datasets import load_wine
import numpy as np
import random

from sklearn.model_selection import train_test_split


def get_latest_run_id(experiment_id: str) -> str:
    result = mlflow.search_runs(experiment_ids=experiment_id)
    return result.iloc[0, 0]


def get_path_of_run_id(experiment_id: str,
                       run_id: str,
                       suffix: str = "") -> str:
    result = mlflow.search_runs(experiment_ids=experiment_id)
    path = result.loc[result.run_id == run_id, "artifact_uri"].get(0)
    if suffix != "":
        path = path + f"/{suffix}"
    return path


def get_model_path_of_latest_run(experiment_id: str,
                                 suffix: str = "") -> str:
    return get_path_of_run_id(experiment_id,
                              get_latest_run_id(experiment_id),
                              suffix)


def get_random_wine_classifier_input() -> np.array:
    X, _ = load_wine(return_X_y=True)
    return random.choice(X).reshape(1, -1)


def load_airbnb_data():
    df = pd.read_csv("../data/airbnb-cleaned-mlflow.csv")
    X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1),
                                                        df[["price"]].values.ravel(),
                                                        random_state=42)
    return X_train, X_test, y_train, y_test