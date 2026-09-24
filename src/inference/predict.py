import pandas as pd

from src.inference.loader import load_artifacts
from src.utils.config import load_config

config = load_config()
THRESHOLD = float(config["model"]["threshold"])

model, preprocessor, feature_list, model_version = load_artifacts()


def predict_order(order_data: dict):
    df = pd.DataFrame([order_data])

    transformed = preprocessor.transform(df)

    probability = model.predict_proba(transformed)[0][1]
    prediction = int(probability >= THRESHOLD)

    return {
        "prediction": prediction,
        "probability": float(probability),
        "model_version": model_version,
    }


if __name__ == "__main__":
    sample_order = {
        "customer_zip_code_prefix": 13023,
        "customer_city": "campinas",
        "customer_state": "SP",
        "item_count": 1,
        "total_price": 120.0,
        "total_freight": 18.0,
        "payment_count": 1,
        "total_payment": 138.0,
        "max_installments": 3,
        "unique_products": 1,
        "unique_sellers": 1,
        "unique_categories": 1,
        "purchase_year": 2018,
        "purchase_month": 5,
        "purchase_dayofweek": 2,
        "purchase_hour": 14,
        "estimated_delivery_days": 20,
    }

    print(predict_order(sample_order))
