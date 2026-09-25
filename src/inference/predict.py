import pandas as pd

from src.features.preprocessing import transform_features
from src.inference.loader import load_artifacts
from src.utils.config import load_config

config = load_config()

THRESHOLD = float(config["model"]["threshold"])

model, preprocessor, feature_list, model_version = load_artifacts()


def predict_order(order_data: dict):
    df = pd.DataFrame([order_data])

    transformed = transform_features(
        df,
        preprocessor,
    )

    probability = model.predict_proba(transformed)[0][1]

    prediction = int(probability >= THRESHOLD)

    return {
        "prediction": prediction,
        "probability": float(probability),
        "model_version": model_version,
    }
