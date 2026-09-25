import pandas as pd


def create_late_delivery_label(df):
    df = df.copy()

    df["order_delivered_customer_date"] = pd.to_datetime(
        df["order_delivered_customer_date"],
        errors="coerce",
    )

    df["order_estimated_delivery_date"] = pd.to_datetime(
        df["order_estimated_delivery_date"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
    ).copy()

    df["is_late"] = (
        df["order_delivered_customer_date"]
        > df["order_estimated_delivery_date"]
    ).astype(int)

    return df