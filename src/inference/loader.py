import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from src.utils.config import load_config

BASE_DIR = Path(__file__).resolve().parents[2]

config = load_config()

PREPROCESSOR_PATH = BASE_DIR / config["model"]["preprocessor_file"]
FEATURE_LIST_PATH = BASE_DIR / config["model"]["feature_list_file"]

TRACKING_URI = config["mlflow"]["tracking_uri"]
REGISTERED_MODEL_NAME = config["mlflow"]["registered_model_name"]
MODEL_STAGE = config["mlflow"]["stage"]


def load_artifacts():
    mlflow.set_tracking_uri(TRACKING_URI)

    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"

    # Load model from MLflow Model Registry
    model = mlflow.sklearn.load_model(model_uri)

    # Get the registered model version
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

    # Load fitted preprocessing artifacts
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as file:
        feature_list = json.load(file)

    return model, preprocessor, feature_list, model_version
