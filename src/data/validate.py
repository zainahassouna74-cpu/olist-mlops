from pathlib import Path

import great_expectations as gx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "notebooks" / "train.csv"


def validate_training_data():
    # Load training data
    df = pd.read_csv(DATA_PATH)

    # Load GX project context
    context = gx.get_context(mode="file")

    # Create or retrieve Pandas datasource
    try:
        data_source = context.data_sources.get("olist_pandas")
    except Exception:  # noqa: BLE001
        data_source = context.data_sources.add_pandas("olist_pandas")

    # Create or retrieve dataframe asset
    try:
        data_asset = data_source.get_asset("training_data")
    except Exception:  # noqa: BLE001
        data_asset = data_source.add_dataframe_asset(name="training_data")

    # Create or retrieve batch definition
    try:
       batch_definition = data_asset.get_batch_definition("whole_training_data")
    except Exception:  # noqa: BLE001
     batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "whole_training_data"
    )

     batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    expectations = [
        gx.expectations.ExpectTableColumnsToMatchSet(
            column_set=[
                "order_id",
                "customer_id",
                "order_status",
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "customer_unique_id",
                "customer_zip_code_prefix",
                "customer_city",
                "customer_state",
                "item_count",
                "total_price",
                "total_freight",
                "payment_count",
                "total_payment",
                "max_installments",
                "review_count",
                "avg_review_score",
                "unique_products",
                "unique_sellers",
                "unique_categories",
                "is_late",
            ],
            exact_match=True,
        ),
        # Important columns should not be missing
        gx.expectations.ExpectColumnValuesToNotBeNull(column="order_id"),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="customer_state"),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="total_price"),
        # Numeric ranges
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="item_count",
            min_value=1,
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="total_price",
            min_value=0,
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="total_freight",
            min_value=0,
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="payment_count",
            min_value=1,
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="total_payment",
            min_value=0,
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="max_installments",
            min_value=1,
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="unique_products",
            min_value=1,
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="unique_sellers",
            min_value=1,
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="unique_categories",
            min_value=0,
        ),
        # Review score should be between 1 and 5
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="avg_review_score",
            min_value=1,
            max_value=5,
            mostly=0.95,
        ),
        # Target must be binary
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="is_late",
            value_set=[0, 1],
        ),
    ]
    all_success = True

    for expectation in expectations:
        result = batch.validate(expectation)

        print(
            f"{expectation.__class__.__name__}: "
            f"{'PASSED' if result.success else 'FAILED'}"
        )

        if not result.success:
            all_success = False

    if not all_success:
        raise ValueError("Great Expectations validation failed.")

    print("\nAll Great Expectations checks passed.")


if __name__ == "__main__":
    validate_training_data()
