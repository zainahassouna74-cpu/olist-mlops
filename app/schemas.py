from datetime import datetime

from pydantic import BaseModel, Field


class OrderInput(BaseModel):
    customer_zip_code_prefix: int
    customer_city: str
    customer_state: str

    item_count: int = Field(ge=1)
    total_price: float = Field(ge=0)
    total_freight: float = Field(ge=0)

    payment_count: int = Field(ge=1)
    total_payment: float = Field(ge=0)
    max_installments: int = Field(ge=1)

    unique_products: int = Field(ge=1)
    unique_sellers: int = Field(ge=1)
    unique_categories: int = Field(ge=0)

    order_purchase_timestamp: datetime
    order_estimated_delivery_date: datetime


class PredictionResponse(BaseModel):
    prediction: int
    probability: float | None = None
    model_version: str


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]
    model_version: str
