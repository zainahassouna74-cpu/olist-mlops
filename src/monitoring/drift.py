import json
from pathlib import Path


def calculate_prediction_drift(
    log_path: str | Path,
    baseline_late_rate: float,
    alert_threshold: float = 0.10,
    min_samples: int = 20,
):
    log_path = Path(log_path)

    if not log_path.exists():
        return {
            "status": "no_data",
            "sample_count": 0,
            "alert": False,
        }

    predictions = []

    with log_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            prediction = record.get("prediction")

            if prediction in [0, 1]:
                predictions.append(prediction)

    sample_count = len(predictions)

    if sample_count < min_samples:
        return {
            "status": "insufficient_data",
            "sample_count": sample_count,
            "alert": False,
        }

    current_late_rate = sum(predictions) / sample_count

    drift = abs(current_late_rate - baseline_late_rate)

    alert = drift >= alert_threshold

    return {
        "status": "alert" if alert else "ok",
        "sample_count": sample_count,
        "baseline_late_rate": round(
            baseline_late_rate,
            4,
        ),
        "current_late_rate": round(
            current_late_rate,
            4,
        ),
        "drift": round(
            drift,
            4,
        ),
        "threshold": alert_threshold,
        "alert": alert,
    }
