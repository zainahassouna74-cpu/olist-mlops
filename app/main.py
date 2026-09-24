import time

from fastapi import FastAPI, HTTPException

from app.schemas import BatchPredictionResponse, OrderInput, PredictionResponse
from src.inference.predict import model_version, predict_order
from src.utils.logger import get_logger

app = FastAPI(
    title="Olist Late Delivery Prediction API",
    version="1.0.0",
)

logger = get_logger(__name__)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(order: OrderInput):
    start_time = time.perf_counter()

    try:
        result = predict_order(order.model_dump())

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "prediction_request | input=%s | output=%s | latency_ms=%.2f | model_version=%s",
            order.model_dump(),
            result,
            latency_ms,
            result["model_version"],
        )

        return {
            "prediction": result["prediction"],
            "probability": result["probability"],
            "model_version": result["model_version"],
        }

    except Exception as exc:
        logger.exception("prediction_failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        ) from exc


@app.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict(orders: list[OrderInput]):
    if not orders:
        raise HTTPException(
            status_code=400,
            detail="At least one order is required",
        )

    start_time = time.perf_counter()

    try:
        results = []

        for order in orders:
            result = predict_order(order.model_dump())

            results.append(
                {
                    "prediction": result["prediction"],
                    "probability": result["probability"],
                    "model_version": result["model_version"],
                }
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "batch_prediction_request | count=%d | input=%s | output=%s | latency_ms=%.2f | model_version=%s",
            len(orders),
            [order.model_dump() for order in orders],
            results,
            latency_ms,
            model_version,
        )

        return {
            "count": len(results),
            "predictions": results,
            "model_version": model_version,
        }

    except Exception as exc:
        logger.exception("prediction_failed")

        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed",
        ) from exc


@app.get("/model-info")
def model_info():
    return {
        "model_name": "logistic_regression",
        "model_version": model_version,
        "task": "late_delivery_classification",
    }
