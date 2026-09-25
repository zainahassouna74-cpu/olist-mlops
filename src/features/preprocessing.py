import pandas as pd

from src.features.build_features import build_features


def transform_features(
    df: pd.DataFrame,
    preprocessor,
):
    features = build_features(df)

    transformed = preprocessor.transform(features)

    return transformed
