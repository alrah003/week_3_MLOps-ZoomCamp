import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import pickle

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    lr = data[0]
    dv = data[1]
    print("MLflow Version:", mlflow.__version__)
    mlflow.set_tracking_uri("http://mlflow:5000")
    with mlflow.start_run():
        mlflow.set_tag("Developer", "Michael Mannerow")
        mlflow.sklearn.log_model(lr, "linear_regression_model")

        # Serialize and save the DictVectorizer using pickle
        dv_path = "dict_vectorizer.pkl"  # File path for the serialized DictVectorizer
        with open(dv_path, 'wb') as f:  # Open the file in write-binary mode
            pickle.dump(dv, f)  # Serialize the DictVectorizer and save it to the file

        # Log the file as an artifact in MLflow
        mlflow.log_artifact(dv_path, "model_artifacts")  # Log the file under the "model_artifacts" folder

        #Register the model:
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/linear_regression_model"
        mlflow.register_model(model_uri, "LinearRegressionModel")