import pandas as pd

from src.features.preprocessing import transform_features


class DummyPreprocessor:
    def __init__(self):
        self.transform_called = False

    def transform(self, data):
        self.transform_called = True
        return data


def test_preprocessing_uses_transform():
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

    preprocessor = DummyPreprocessor()

    transformed = transform_features(
        df,
        preprocessor,
    )

    assert preprocessor.transform_called is True
    assert transformed is not None
