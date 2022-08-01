import pandas as pd
import pytest

from gps_activity.abstract import AbstractNode
from gps_activity.models import DataFramePivotFields
from gps_activity.nodes import UniqueVehicleConstrain


@pytest.fixture
def gps_valid() -> pd.DataFrame:
    content = {"plate_no": ["123", "123", "123"]}
    return pd.DataFrame(content)


@pytest.fixture
def gps_invalid() -> pd.DataFrame:
    content = {"plate_no": ["123", "1234", "12345"]}
    return pd.DataFrame(content)


@pytest.fixture
def constrain_module(gps_pivot_fields: DataFramePivotFields) -> pd.DataFrame:
    return UniqueVehicleConstrain(pivot_fields=gps_pivot_fields)


class TestUniqueVehicleConstrain:
    def test_validation_failure(
        self,
        constrain_module: AbstractNode,
        gps_invalid: pd.DataFrame,
    ):
        try:
            constrain_module.transform(gps_invalid)
            raise AssertionError("Execution must fail")
        except ValueError:
            assert True

    def test_validation_success(
        self,
        constrain_module: AbstractNode,
        gps_valid: pd.DataFrame,
    ):
        constrain_module.transform(gps_valid)
        assert True
