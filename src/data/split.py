import pandas as pd


def time_based_split(
    df,
    train_ratio=0.70,
    validation_ratio=0.15,
):
    df = df.copy()

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"],
        errors="coerce",
    )

    df = df.sort_values(
        "order_purchase_timestamp"
    ).reset_index(drop=True)

    n_rows = len(df)

    train_end = int(n_rows * train_ratio)

    validation_end = int(
        n_rows * (train_ratio + validation_ratio)
    )

    train_df = df.iloc[:train_end].copy()

    validation_df = df.iloc[
        train_end:validation_end
    ].copy()

    test_df = df.iloc[
        validation_end:
    ].copy()

    return train_df, validation_df, test_df