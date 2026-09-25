import pandas as pd

from src.features.preprocessing import transform_features
from src.inference.loader import load_artifacts


def test_model_artifacts_load_successfully():
    model, preprocessor, feature_list, model_version = load_artifacts()

    assert model is not None
    assert preprocessor is not None
    assert feature_list is not None
    assert model_version is not None


def test_model_predict_proba_output():
    model, preprocessor, _, _ = load_artifacts()

    df = pd.DataFrame(
        [
            {
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
        ]
    )

    transformed = transform_features(
        df,
        preprocessor,
    )

    probabilities = model.predict_proba(transformed)

    assert len(probabilities) == 1
    assert len(probabilities[0]) == 2

    probability = probabilities[0][1]

    assert 0.0 <= probability <= 1.0
