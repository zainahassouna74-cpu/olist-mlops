from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_endpoint():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "logistic_regression"
    assert data["model_version"] == "1"
    assert data["task"] == "late_delivery_classification"


def test_predict_endpoint():
    payload = {
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

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0
    assert data["model_version"] == "1"


def test_batch_predict_endpoint():
    payload = [
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
            "purchase_year": 2018,
            "purchase_month": 5,
            "purchase_dayofweek": 2,
            "purchase_hour": 14,
            "estimated_delivery_days": 20,
        },
        {
            "customer_zip_code_prefix": 22041,
            "customer_city": "rio de janeiro",
            "customer_state": "RJ",
            "item_count": 2,
            "total_price": 180.0,
            "total_freight": 25.0,
            "payment_count": 1,
            "total_payment": 205.0,
            "max_installments": 4,
            "unique_products": 2,
            "unique_sellers": 1,
            "unique_categories": 2,
            "purchase_year": 2018,
            "purchase_month": 6,
            "purchase_dayofweek": 4,
            "purchase_hour": 10,
            "estimated_delivery_days": 15,
        },
    ]

    response = client.post("/batch-predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    assert data["model_version"] == "1"

    for prediction in data["predictions"]:
        assert prediction["prediction"] in [0, 1]
        assert 0.0 <= prediction["probability"] <= 1.0
        assert data["model_version"] == "1"


def test_predict_invalid_payload():
    payload = {"customer_state": "SP"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
