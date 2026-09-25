import json

from src.monitoring.drift import (
    calculate_prediction_drift,
)


def test_drift_returns_no_data_for_missing_file(
    tmp_path,
):
    log_file = tmp_path / "missing.jsonl"

    result = calculate_prediction_drift(
        log_path=log_file,
        baseline_late_rate=0.09,
    )

    assert result["status"] == "no_data"
    assert result["alert"] is False


def test_drift_detects_large_distribution_change(
    tmp_path,
):
    log_file = tmp_path / "predictions.jsonl"

    with log_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        for _ in range(20):
            record = {"prediction": 1}

            file.write(json.dumps(record) + "\n")

    result = calculate_prediction_drift(
        log_path=log_file,
        baseline_late_rate=0.09,
        alert_threshold=0.10,
        min_samples=20,
    )

    assert result["status"] == "alert"
    assert result["alert"] is True


def test_drift_no_alert_when_distribution_is_close(
    tmp_path,
):
    log_file = tmp_path / "predictions.jsonl"

    predictions = [1, 1] + [0] * 18

    with log_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        for prediction in predictions:
            record = {"prediction": prediction}

            file.write(json.dumps(record) + "\n")

    result = calculate_prediction_drift(
        log_path=log_file,
        baseline_late_rate=0.09,
        alert_threshold=0.10,
        min_samples=20,
    )

    assert result["status"] == "ok"
    assert result["alert"] is False
