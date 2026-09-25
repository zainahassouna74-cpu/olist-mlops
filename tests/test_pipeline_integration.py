import pandas as pd

from src.features.preprocessing import transform_features
from src.inference.loader import load_artifacts
from src.inference.predict import predict_order


def test_full_prediction_pipeline():
    order_data = {
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
        "order_purchase_timestamp": "2018-05-10T14:00:00",
        "order_estimated_delivery_date": "2018-05-30T14:00:00",
    }

    result = predict_order(order_data)

    assert result["prediction"] in [0, 1]
    assert 0.0 <= result["probability"] <= 1.0
    assert result["model_version"] is not None


def test_pipeline_matches_manual_prediction():
    order_data = {
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
        "order_purchase_timestamp": "2018-05-10T14:00:00",
        "order_estimated_delivery_date": "2018-05-30T14:00:00",
    }

    pipeline_result = predict_order(order_data)

    model, preprocessor, _, _ = load_artifacts()

    df = pd.DataFrame([order_data])

    transformed = transform_features(
        df,
        preprocessor,
    )

    manual_probability = float(model.predict_proba(transformed)[0][1])

    assert abs(pipeline_result["probability"] - manual_probability) < 1e-10
