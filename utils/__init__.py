""" Just some utils
"""
import mlflow
import pandas as pd
from pandas import DataFrame
from sklearn.datasets import load_wine
import numpy as np
import random
from sklearn.model_selection import train_test_split
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mlflow.tracking import MlflowClient
import os
from typing import Tuple


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


def compare_plots(experiment_name: str,
                  run_id_1: str,
                  run_id_2: str,
                  plot_name: str = "my_confusion_matrix.png",
                  figsize: Tuple[int, int] = (12, 4)):

    """
    Workaround to compare two plots given two experiments.
    This is just a quick and dirty workaround.
    The possibility to compare artifacts might arise soon,
    see e.g. https://github.com/mlflow/mlflow/pull/4413/commits.
    """

    # Initialize mlflow client
    client = MlflowClient()
    mlflow.set_experiment(experiment_name)

    # Delete tmp download folder
    try:
        shutil.rmtree(os.getcwd() + "/download")
    except:
        pass

    # Build plot
    fig = plt.figure(figsize=figsize)
    plt.title(f"Comparison of {plot_name}")
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)

    # Build first subplot
    ax1 = fig.add_subplot(1, 2, 1)
    # Download and read
    os.mkdir(os.getcwd() + "/download")
    local_path = client.download_artifacts(run_id=run_id_1,
                                           path="",
                                           dst_path=os.getcwd() + "/download/")
    path_to_picture = local_path + plot_name
    # Show
    ax1.imshow(mpimg.imread(path_to_picture))
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)

    # Build second subplot
    ax1 = fig.add_subplot(1, 2, 2)
    # Download and read
    local_path = client.download_artifacts(run_id=run_id_2,
                                           path="",
                                           dst_path=os.getcwd() + "/download/")
    # Show
    ax1.imshow(mpimg.imread(path_to_picture))
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)

    # Show plot and delete tmp download folder
    plt.show()
    shutil.rmtree(os.getcwd() + "/download")


def get_random_wine_classifier_input() -> np.array:
    X, _ = load_wine(return_X_y=True)
    return random.choice(X).reshape(1, -1)


def load_airbnb_data():
    df = pd.read_csv("../data/airbnb-cleaned-mlflow.csv")
    X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1),
                                                        df[["price"]].values.ravel(),
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


def preprocess_airbnb_data(df: DataFrame) -> DataFrame:

    """ Do some preprocessing """
    cols_to_drop = ["latitude", "longitude"]

    df_processed = df.copy()
    df_processed["trunc_lat"] = round(df["latitude"], 3)
    df_processed["trunc_long"] = round(df["longitude"], 3)
    df_processed["review_scores_sum"] = (
            df['review_scores_accuracy'] +
            df['review_scores_cleanliness'] +
            df['review_scores_checkin'] +
            df['review_scores_communication'] +
            df['review_scores_location'] +
            df['review_scores_value']
    )
    df_processed = df_processed.drop(cols_to_drop, axis=1)

    return df_processed


