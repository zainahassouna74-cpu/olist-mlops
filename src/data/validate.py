from pathlib import Path

import great_expectations as gx
import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "notebooks" / "train.csv"


EXPECTED_COLUMNS = [
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
]


VALID_ORDER_STATUS = [
    "delivered",
    "shipped",
    "canceled",
    "unavailable",
    "invoiced",
    "processing",
    "created",
    "approved",
]


VALID_BRAZILIAN_STATES = [
    "AC",
    "AL",
    "AP",
    "AM",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MT",
    "MS",
    "MG",
    "PA",
    "PB",
    "PR",
    "PE",
    "PI",
    "RJ",
    "RN",
    "RS",
    "RO",
    "RR",
    "SC",
    "SP",
    "SE",
    "TO",
]


def validate_training_data(data_path=None):
    if data_path is None:
        data_path = DEFAULT_DATA_PATH

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {data_path}"
        )

    logger.info("Loading training data from %s", data_path)

    df = pd.read_csv(data_path)

    # Basic dtype validation
    numeric_columns = [
        "customer_zip_code_prefix",
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
    ]

    for column in numeric_columns:
        if column in df.columns and not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            raise ValueError(
                f"Column '{column}' must be numeric."
            )

    # Load Great Expectations project context
    context = gx.get_context(mode="file")

    # Create or retrieve Pandas datasource
    try:
        data_source = context.data_sources.get(
            "olist_pandas"
        )
    except Exception:  # noqa: BLE001
        data_source = context.data_sources.add_pandas(
            "olist_pandas"
        )

    # Create or retrieve dataframe asset
    try:
        data_asset = data_source.get_asset(
            "training_data"
        )
    except Exception:  # noqa: BLE001
        data_asset = data_source.add_dataframe_asset(
            name="training_data"
        )

    # Create or retrieve batch definition
    try:
        batch_definition = (
            data_asset.get_batch_definition(
                "whole_training_data"
            )
        )
    except Exception:  # noqa: BLE001
        batch_definition = (
            data_asset.add_batch_definition_whole_dataframe(
                "whole_training_data"
            )
        )

    batch = batch_definition.get_batch(
        batch_parameters={"dataframe": df}
    )

    expectations = [
        # Schema
        gx.expectations.ExpectTableColumnsToMatchSet(
            column_set=EXPECTED_COLUMNS,
            exact_match=True,
        ),

        # Required values
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="order_id"
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="customer_id"
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="customer_state"
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="total_price"
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="is_late"
        ),

        # Missing-rate expectations
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="total_freight",
            mostly=0.99,
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="total_payment",
            mostly=0.99,
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="customer_city",
            mostly=0.99,
        ),

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

        # Review score
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="avg_review_score",
            min_value=1,
            max_value=5,
            mostly=0.95,
        ),

        # Allowed categorical values
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="customer_state",
            value_set=VALID_BRAZILIAN_STATES,
        ),
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="order_status",
            value_set=VALID_ORDER_STATUS,
        ),

        # Binary target
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="is_late",
            value_set=[0, 1],
        ),
    ]

    all_success = True

    for expectation in expectations:
        result = batch.validate(expectation)

        if result.success:
            logger.info(
                "%s: PASSED",
                expectation.__class__.__name__,
            )
        else:
            logger.error(
                "%s: FAILED",
                expectation.__class__.__name__,
            )
            all_success = False

    if not all_success:
        logger.error(
            "Great Expectations validation failed."
        )

        # Our selected policy = REJECT invalid data
        raise ValueError(
            "Great Expectations validation failed."
        )

    logger.info(
        "All Great Expectations checks passed."
    )

    return True


if __name__ == "__main__":
    validate_training_data()