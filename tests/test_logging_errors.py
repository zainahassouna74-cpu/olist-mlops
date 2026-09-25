import logging

from fastapi.testclient import TestClient

from app.main import app
from src.utils.logger import get_logger

client = TestClient(app)


def test_logger_has_file_and_console_handlers():
    logger = get_logger("test_logger")

    has_file_handler = any(
        isinstance(handler, logging.FileHandler) for handler in logger.handlers
    )

    has_console_handler = any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    )

    assert has_file_handler
    assert has_console_handler


def test_logger_level_is_info():
    logger = get_logger("test_logger_level")

    assert logger.level == logging.INFO


def test_missing_fields_return_422():
    payload = {"customer_state": "SP"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_invalid_numeric_value_returns_422():
    payload = {
        "customer_zip_code_prefix": 13023,
        "customer_city": "campinas",
        "customer_state": "SP",
        "item_count": -1,
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

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_invalid_date_returns_422():
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
        "order_purchase_timestamp": "not-a-date",
        "order_estimated_delivery_date": "2018-05-30T14:00:00",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
