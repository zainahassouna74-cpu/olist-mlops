import os
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Allow Docker/environment variables to override MLflow tracking URI
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if mlflow_tracking_uri:
        config["mlflow"]["tracking_uri"] = mlflow_tracking_uri

    return config


if __name__ == "__main__":
    config = load_config()
    print(config)
