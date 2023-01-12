import pandas as pd
import pytest

from gps_activity.abstract import AbstractNode
from gps_activity.models import DataFramePivotFields
from gps_activity.nodes import UniqueTimestampConstraint


@pytest.fixture
def data_duplicated(gps_pivot_fields: DataFramePivotFields) -> pd.DataFrame:
    content = {gps_pivot_fields.source_datetime: ["123", "123", "123"]}
    return pd.DataFrame(content)


@pytest.fixture
def data_unique(gps_pivot_fields: DataFramePivotFields) -> pd.DataFrame:
    content = {gps_pivot_fields.source_datetime: ["123", "124", "125"]}
    return pd.DataFrame(content)


@pytest.fixture
def constraint_node(gps_pivot_fields: DataFramePivotFields) -> pd.DataFrame:
    return UniqueTimestampConstraint(
        source_datetime=gps_pivot_fields.source_datetime,
    )


class TestUniqueTimestampConstraint:
    def test_instance(self, constraint_node: UniqueTimestampConstraint):
        assert isinstance(constraint_node, AbstractNode)

    def test_duplicated_timestamp(
        self,
        constraint_node: UniqueTimestampConstraint,
        data_duplicated: pd.DataFrame,
    ):
        try:
            constraint_node.transform(X=data_duplicated)
            raise AssertionError("Failure is expected due to duplicates")
        except ValueError:
            assert True

    def test_unique_timestamp(
        self,
        constraint_node: UniqueTimestampConstraint,
        data_unique: pd.DataFrame,
    ):
        constraint_node.transform(X=data_unique)
