import json
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException

from app.schemas import BatchPredictionResponse, OrderInput, PredictionResponse
from src.inference.predict import model_version, predict_order
from src.utils.logger import get_logger

app = FastAPI(
    title="Olist Late Delivery Prediction API",
    version="1.0.0",
)

logger = get_logger(__name__)

# -------------------------------------------------------------------
# Monitoring state
# -------------------------------------------------------------------

metrics_lock = Lock()

monitoring = {
    "request_count": 0,
    "prediction_count": 0,
    "error_count": 0,
    "total_latency_ms": 0.0,
    "late_predictions": 0,
    "on_time_predictions": 0,
}

PREDICTION_LOG_PATH = Path("logs/predictions.jsonl")
PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def save_prediction_log(order_data: dict, result: dict, latency_ms: float) -> None:
    """Store predictions so they can be evaluated later when ground truth arrives."""

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": order_data,
        "prediction": result["prediction"],
        "probability": result["probability"],
        "model_version": result["model_version"],
        "latency_ms": round(latency_ms, 2),
        "actual_delivery_status": None,
    }

    with PREDICTION_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, default=str) + "\n")


def update_monitoring(result: dict, latency_ms: float) -> None:
    with metrics_lock:
        monitoring["request_count"] += 1
        monitoring["prediction_count"] += 1
        monitoring["total_latency_ms"] += latency_ms

        if result["prediction"] == 1:
            monitoring["late_predictions"] += 1
        else:
            monitoring["on_time_predictions"] += 1


def register_error() -> None:
    with metrics_lock:
        monitoring["request_count"] += 1
        monitoring["error_count"] += 1


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    with metrics_lock:
        request_count = monitoring["request_count"]
        prediction_count = monitoring["prediction_count"]
        error_count = monitoring["error_count"]
        total_latency_ms = monitoring["total_latency_ms"]
        late_predictions = monitoring["late_predictions"]
        on_time_predictions = monitoring["on_time_predictions"]

    average_latency_ms = (
        total_latency_ms / prediction_count if prediction_count else 0.0
    )

    error_rate = error_count / request_count if request_count else 0.0

    late_prediction_rate = (
        late_predictions / prediction_count if prediction_count else 0.0
    )

    return {
        "request_count": request_count,
        "prediction_count": prediction_count,
        "error_count": error_count,
        "error_rate": round(error_rate, 4),
        "average_latency_ms": round(average_latency_ms, 2),
        "prediction_distribution": {
            "late": late_predictions,
            "on_time": on_time_predictions,
            "late_rate": round(late_prediction_rate, 4),
        },
        "model_version": model_version,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(order: OrderInput):
    start_time = time.perf_counter()

    try:
        order_data = order.model_dump()
        result = predict_order(order_data)

        latency_ms = (time.perf_counter() - start_time) * 1000

        update_monitoring(result, latency_ms)
        save_prediction_log(order_data, result, latency_ms)

        logger.info(
            "prediction_request | input=%s | output=%s | latency_ms=%.2f | "
            "model_version=%s",
            order_data,
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
        register_error()
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
            order_data = order.model_dump()
            result = predict_order(order_data)

            results.append(
                {
                    "prediction": result["prediction"],
                    "probability": result["probability"],
                    "model_version": result["model_version"],
                }
            )

        latency_ms = (time.perf_counter() - start_time) * 1000
        latency_per_prediction = latency_ms / len(results)

        for order, result in zip(orders, results, strict=True):
            update_monitoring(result, latency_per_prediction)
            save_prediction_log(
                order.model_dump(),
                result,
                latency_per_prediction,
            )

        logger.info(
            "batch_prediction_request | count=%d | input=%s | output=%s | "
            "latency_ms=%.2f | model_version=%s",
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
        register_error()
        logger.exception("batch_prediction_failed")

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
