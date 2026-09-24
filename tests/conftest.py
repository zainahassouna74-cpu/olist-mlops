import src.inference.loader as loader


class DummyPreprocessor:
    def transform(self, df):
        return df


class DummyModel:
    def predict_proba(self, data):
        return [[0.2, 0.8]]


def fake_load_artifacts():
    return DummyModel(), DummyPreprocessor(), [], "1"


loader.load_artifacts = fake_load_artifacts