import json

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from src.utils.config import load_config


config = load_config()

TRACKING_URI = config["mlflow"]["tracking_uri"]
REGISTERED_MODEL_NAME = config["mlflow"]["registered_model_name"]
MODEL_STAGE = config["mlflow"]["stage"]


def load_artifacts():
    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient()

    versions = client.get_latest_versions(
        REGISTERED_MODEL_NAME,
        stages=[MODEL_STAGE],
    )

    if not versions:
        raise RuntimeError(
            f"No model found in stage '{MODEL_STAGE}' "
            f"for '{REGISTERED_MODEL_NAME}'."
        )

    model_version = str(versions[0].version)
    run_id = versions[0].run_id

    model_uri = (
        f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
    )

    model = mlflow.sklearn.load_model(model_uri)

    preprocessor_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="preprocessing/preprocessor.pkl",
    )

    feature_list_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="preprocessing/feature_list.json",
    )

    preprocessor = joblib.load(preprocessor_path)

    with open(
        feature_list_path,
        "r",
        encoding="utf-8",
    ) as file:
        feature_list = json.load(file)

    return (
        model,
        preprocessor,
        feature_list,
        model_version,
    )