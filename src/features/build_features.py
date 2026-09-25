import pandas as pd

TARGET_COLUMN = "is_late"

DROP_COLUMNS = [
    "order_id",
    "customer_id",
    "customer_unique_id",
    "order_status",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "review_count",
    "avg_review_score",
]

PAYMENT_COLUMNS = [
    "payment_count",
    "total_payment",
    "max_installments",
]


def build_features(df):
    df = df.copy()

    columns_to_drop = [
        column for column in DROP_COLUMNS + [TARGET_COLUMN] if column in df.columns
    ]

    df = df.drop(columns=columns_to_drop)

    if "order_purchase_timestamp" in df.columns:
        df["order_purchase_timestamp"] = pd.to_datetime(
            df["order_purchase_timestamp"],
            errors="coerce",
        )

    if "order_estimated_delivery_date" in df.columns:
        df["order_estimated_delivery_date"] = pd.to_datetime(
            df["order_estimated_delivery_date"],
            errors="coerce",
        )

    if (
        "order_purchase_timestamp" in df.columns
        and "order_estimated_delivery_date" in df.columns
    ):
        df["purchase_year"] = df["order_purchase_timestamp"].dt.year

        df["purchase_month"] = df["order_purchase_timestamp"].dt.month

        df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek

        df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour

        df["estimated_delivery_days"] = (
            df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
        ).dt.total_seconds() / 86400

        df = df.drop(
            columns=[
                "order_purchase_timestamp",
                "order_estimated_delivery_date",
            ]
        )

    existing_payment_columns = [
        column for column in PAYMENT_COLUMNS if column in df.columns
    ]

    if existing_payment_columns:
        df[existing_payment_columns] = df[existing_payment_columns].fillna(0)

    return df
