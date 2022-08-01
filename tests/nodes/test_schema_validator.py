import pandas as pd
from pandera.errors import SchemaError
import pytest


from gps_activity.abstract import AbstractNode
from gps_activity.models import DataFramePivotFields
from gps_activity.nodes import PanderaValidator


def bare_minimum_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lat": [22.446138, 22.44614],
            "lon": [114.006416, 114.00642],
            "datetime": ["2022-05-10 00:12:37", "2022-05-10 00:12:23"],
            "plate_no": ["123", "123"],
        },
    )


def missing_coordinate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lon": [114.006416, 114.00642],
            "datetime": ["2022-05-10 00:12:37", "2022-05-10 00:12:23"],
            "plate_no": ["123", "123"],
        },
    )


def missing_datetime() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lat": [22.446138, 22.44614],
            "lon": [114.006416, 114.00642],
            "plate_no": ["123", "123"],
        },
    )


def missing_plate_no() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lat": [22.446138, 22.44614],
            "lon": [114.006416, 114.00642],
            "datetime": ["2022-05-10 00:12:37", "2022-05-10 00:12:23"],
        },
    )


def redundant_fields() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lat": [22.446138, 22.44614],
            "lon": [114.006416, 114.00642],
            "datetime": ["2022-05-10 00:12:37", "2022-05-10 00:12:23"],
            "random1": [114.006416, 114.00642],
            "random2": [114.006416, 114.00642],
            "plate_no": ["123", "123"],
        },
    )


@pytest.fixture
def schema_validator(gps_pivot_fields: DataFramePivotFields) -> PanderaValidator:
    return PanderaValidator(gps_pivot_fields)


class TestPanderaValidator:
    def test_instance(self, schema_validator: PanderaValidator):
        assert isinstance(schema_validator, AbstractNode)

    @pytest.mark.parametrize(
        "input_df",
        [
            bare_minimum_dataframe(),
            redundant_fields(),
        ],
    )
    def test_succesfull_run(
        self,
        schema_validator: PanderaValidator,
        input_df: pd.DataFrame,
    ):
        schema_validator.fit_transform(input_df)

    @pytest.mark.parametrize(
        "input_df",
        [
            missing_coordinate(),
            missing_datetime(),
            missing_plate_no(),
        ],
    )
    def test_schema_error(
        self,
        schema_validator: PanderaValidator,
        input_df: pd.DataFrame,
    ):
        try:
            schema_validator.transform(input_df)
            raise AssertionError("Failure is expected")
        except SchemaError:
            assert True
