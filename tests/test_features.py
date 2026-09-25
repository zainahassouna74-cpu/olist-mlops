import pandas as pd

from src.features.build_features import build_features


def test_build_features():
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

    features = build_features(df)

    assert "order_purchase_timestamp" not in features.columns
    assert "order_estimated_delivery_date" not in features.columns

    assert features.loc[0, "purchase_year"] == 2018
    assert features.loc[0, "purchase_month"] == 5
    assert features.loc[0, "purchase_hour"] == 14
    assert features.loc[0, "estimated_delivery_days"] == 20