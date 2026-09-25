import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from src.utils.config import load_config
from src.utils.logger import get_logger

BASE_DIR = Path(__file__).resolve().parents[2]

config = load_config()
logger = get_logger(__name__)

MODEL_PATH = BASE_DIR / config["model"]["model_file"]
PREPROCESSOR_PATH = BASE_DIR / config["model"]["preprocessor_file"]
FEATURE_LIST_PATH = BASE_DIR / config["model"]["feature_list_file"]
RESULTS_PATH = BASE_DIR / config["paths"]["results_file"]

TRACKING_URI = config["mlflow"]["tracking_uri"]
EXPERIMENT_NAME = config["mlflow"]["experiment_name"]
REGISTERED_MODEL_NAME = config["mlflow"]["registered_model_name"]
MODEL_STAGE = config["mlflow"]["stage"]


def register_model():
    # Load the already trained model
    model = joblib.load(MODEL_PATH)

    # Load metrics produced by Notebook 6
    with open(RESULTS_PATH, "r", encoding="utf-8") as file:
        results = json.load(file)

    # Configure MLflow
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="logistic_regression_production") as run:

        # Log important parameters
        mlflow.log_param(
            "model_type",
            results["model"],
        )
        mlflow.log_param(
            "threshold",
            config["model"]["threshold"],
        )
        mlflow.log_param(
            "project_version",
            config["project"]["version"],
        )

        # Log model parameters
        mlflow.log_params(model.get_params())

        # Log evaluation metrics
        mlflow.log_metrics(
            {
                "test_accuracy": results["test_accuracy"],
                "test_precision": results["test_precision"],
                "test_recall": results["test_recall"],
                "test_f1": results["test_f1"],
                "test_roc_auc": results["test_roc_auc"],
            }
        )

        # Log preprocessing artifacts
        mlflow.log_artifact(
            str(PREPROCESSOR_PATH),
            artifact_path="preprocessing",
        )

        mlflow.log_artifact(
            str(FEATURE_LIST_PATH),
            artifact_path="preprocessing",
        )

        mlflow.log_artifact(
            str(RESULTS_PATH),
            artifact_path="evaluation",
        )

        # Log the trained model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )

        run_id = run.info.run_id

    # Register the logged model in the Model Registry
    model_uri = f"runs:/{run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    # Move the registered version to the configured stage
    client = MlflowClient()

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=registered_model.version,
        stage=MODEL_STAGE,
    )

    logger.info(
        "Run ID: %s",
        run_id,
    )

    logger.info(
        "Registered model: %s",
        REGISTERED_MODEL_NAME,
    )

    logger.info(
        "Model version: %s",
        registered_model.version,
    )

    logger.info(
        "Stage: %s",
        MODEL_STAGE,
    )

    logger.info("MLflow registration completed successfully.")


if __name__ == "__main__":
    register_model()
