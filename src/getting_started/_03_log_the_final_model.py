""" Log the final model
"""
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from config import mlflow_server_uri

# Set final parameter
max_depth = 3


class DecisionTreeModel:

    def __init__(self, max_depth: int):
        self.max_depth = max_depth

    def load_data(self):
        x, y = load_wine(return_X_y=True)
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x, y, test_size=0.33, random_state=42)

    def train(self):
        self.tree = DecisionTreeClassifier(max_depth=self.max_depth)
        self.tree.fit(self.x_train, self.y_train)

    def evaluate(self):
        self.y_pred = self.tree.predict(self.x_test)
        self.precision = precision_score(self.y_test, self.y_pred, average="weighted")
        self.recall = recall_score(self.y_test, self.y_pred, average="weighted")
        self.accuracy = accuracy_score(self.y_test, self.y_pred)


if __name__ == '__main__':

    # Connect to mlflow server
    mlflow.set_tracking_uri(mlflow_server_uri)

    # Set Experiment
    mlflow.set_experiment("My Wine Classifier")

    with mlflow.start_run(run_name="final_run"):

        # Load model, data, train and evaluate
        model = DecisionTreeModel(max_depth=max_depth)
        model.load_data()
        model.train()
        model.evaluate()

        # Log metrics and parameter
        mlflow.log_param("tree_depth", max_depth)
        mlflow.log_metric("precision", model.precision)
        mlflow.log_metric("recall", model.recall)
        mlflow.log_metric("accuracy", model.accuracy)

        # Add input_example and signature as further meta information for the resulting model
        input_example = random.choice(model.x_train)
        signature = infer_signature(model.x_test, model.y_test)

        # Log model
        mlflow.sklearn.log_model(sk_model=model.tree,
                                 artifact_path="my_artifact_path",
                                 signature=signature,
                                 input_example=input_example)
