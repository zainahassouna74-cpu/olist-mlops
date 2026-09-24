from src.inference.predict import predict_order


def test_predict_order_returns_expected_keys():
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

    result = predict_order(sample_order)

    assert "prediction" in result
    assert "probability" in result
    assert result["prediction"] in [0, 1]
    assert 0.0 <= result["probability"] <= 1.0
