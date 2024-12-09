import pandas as pd
import pytest

from custom_transformers.transformers import FeatureSelector


@pytest.fixture
def sample_data():
    return pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})


def test_feature_selector_select_columns(sample_data: pd.DataFrame):
    transformer = FeatureSelector(columns=["A", "B"])

    transformer.fit(sample_data)

    transformed = transformer.transform(sample_data)

    assert list(transformed.columns) == ["A", "B"]
    assert transformed.shape == (3, 2)


def test_feature_selector_invalid_column(sample_data: pd.DataFrame):
    # 'Z' doesn't exist
    transformer = FeatureSelector(columns=["A", "Z"])

    transformer.fit(sample_data)

    with pytest.raises(KeyError):
        _ = transformer.transform(sample_data)
