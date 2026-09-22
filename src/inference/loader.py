from pathlib import Path
import json
import joblib


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "logistic_regression_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
FEATURE_LIST_PATH = MODELS_DIR / "feature_list.json"


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as file:
        feature_list = json.load(file)

    return model, preprocessor, feature_list