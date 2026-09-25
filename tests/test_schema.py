import pytest
from pydantic import ValidationError

from app.schemas import OrderInput


def test_valid_order_schema():
    order = OrderInput(
    customer_zip_code_prefix=13023,
    customer_city="campinas",
    customer_state="SP",
    item_count=1,
    total_price=120.0,
    total_freight=18.0,
    payment_count=1,
    total_payment=138.0,
    max_installments=3,
    unique_products=1,
    unique_sellers=1,
    unique_categories=1,
    order_purchase_timestamp="2018-05-10T14:00:00",
    order_estimated_delivery_date="2018-05-30T14:00:00",
)

    assert order.customer_city == "campinas"


def test_invalid_purchase_month():
    with pytest.raises(ValidationError):
        OrderInput(
            customer_zip_code_prefix=13023,
            customer_city="campinas",
            customer_state="SP",
            item_count=1,
            total_price=120.0,
            total_freight=18.0,
            payment_count=1,
            total_payment=138.0,
            max_installments=3,
            unique_products=1,
            unique_sellers=1,
            unique_categories=1,
            purchase_year=2018,
            purchase_month=13,
            purchase_dayofweek=2,
            purchase_hour=14,
            estimated_delivery_days=20,
        )


def test_negative_total_price():
    with pytest.raises(ValidationError):
        OrderInput(
            customer_zip_code_prefix=13023,
            customer_city="campinas",
            customer_state="SP",
            item_count=1,
            total_price=-10.0,
            total_freight=18.0,
            payment_count=1,
            total_payment=138.0,
            max_installments=3,
            unique_products=1,
            unique_sellers=1,
            unique_categories=1,
            purchase_year=2018,
            purchase_month=5,
            purchase_dayofweek=2,
            purchase_hour=14,
            estimated_delivery_days=20,
        )
