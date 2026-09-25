import pandas as pd
import pytest

from src.data.validate import validate_training_data


def test_validation_rejects_missing_file(tmp_path):
    missing_file = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        validate_training_data(missing_file)


def test_validation_rejects_non_numeric_column(tmp_path):
    data = {
        "customer_zip_code_prefix": [13023],
        "item_count": [1],
        "total_price": ["invalid"],
        "total_freight": [18.0],
        "payment_count": [1],
        "total_payment": [138.0],
        "max_installments": [3],
        "review_count": [1],
        "avg_review_score": [5.0],
        "unique_products": [1],
        "unique_sellers": [1],
        "unique_categories": [1],
        "is_late": [0],
    }

    df = pd.DataFrame(data)

    file_path = tmp_path / "invalid.csv"
    df.to_csv(file_path, index=False)

    with pytest.raises(ValueError):
        validate_training_data(file_path)