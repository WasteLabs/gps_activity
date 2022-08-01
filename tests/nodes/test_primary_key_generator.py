import pandas as pd
import pytest

from gps_activity.abstract import AbstractNode
from gps_activity.models import DefaultValues
from gps_activity.nodes import PrimaryKeyGenerator


_KEYS_DELIMITER = DefaultValues().pk_delimiter


@pytest.fixture
def df_expected():
    return pd.DataFrame(
        {
            "x": [0, 1, 2],
            "y": [0, 1, 2],
            "xy": [f"0{_KEYS_DELIMITER}0", f"1{_KEYS_DELIMITER}1", f"2{_KEYS_DELIMITER}2"],
        },
    )


@pytest.fixture
def df_input(df_expected: pd.DataFrame):
    return df_expected.drop(columns=["xy"])


@pytest.fixture
def primary_key_generator():
    return PrimaryKeyGenerator(target_column="xy", source_columns=["x", "y"])


class TestPrimaryKeyGenerator:
    """
    Module adding unixtime to dataframe
    """

    def test_instance(self, primary_key_generator: PrimaryKeyGenerator):
        assert isinstance(primary_key_generator, AbstractNode)

    def test_fit_transform(
        self,
        primary_key_generator: PrimaryKeyGenerator,
        df_input: pd.DataFrame,
        df_expected: pd.DataFrame,
    ):
        df_computed = primary_key_generator.fit_transform(df_input)
        assert df_computed.equals(df_expected)

    def test_validation_source_columns(
        self,
        primary_key_generator: PrimaryKeyGenerator,
        df_input: pd.DataFrame,
        df_expected: pd.DataFrame,
    ):
        try:
            PrimaryKeyGenerator(target_column="xy", source_columns="DUMMY_INPUT")
            raise AssertionError("Pipeline must fail")
        except ValueError:
            assert True
