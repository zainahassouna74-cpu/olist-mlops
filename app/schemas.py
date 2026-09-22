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
    unique_categories: int = Field(ge=1)

    purchase_year: int
    purchase_month: int = Field(ge=1, le=12)
    purchase_dayofweek: int = Field(ge=0, le=6)
    purchase_hour: int = Field(ge=0, le=23)

    estimated_delivery_days: int = Field(ge=0)

from typing import List, Optional


class PredictionResponse(BaseModel):
    prediction: int
    probability: Optional[float] = None
    model_version: str


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: List[PredictionResponse]
    model_version: str