import pandas as pd
import pytest

from gps_activity.abstract import AbstractNode
from gps_activity.models import DataFramePivotFields
from gps_activity.nodes import SorterGPS


@pytest.fixture
def input_data(gps_pivot_fields: DataFramePivotFields):
    return pd.DataFrame(
        {
            gps_pivot_fields.computed_unixtime: [3, 2, 1],
        },
    )


@pytest.fixture
def expected_data(gps_pivot_fields: DataFramePivotFields):
    return pd.DataFrame(
        {
            gps_pivot_fields.computed_unixtime: [1, 2, 3],
        },
    )


@pytest.fixture
def sort_node(gps_pivot_fields: DataFramePivotFields):
    return SorterGPS(
        unixtime_column=gps_pivot_fields.computed_unixtime,
    )


class TestSorterGPS:
    def test_instance(self, sort_node: SorterGPS):
        assert isinstance(sort_node, AbstractNode)

    def test_crs_projection_model(
        self,
        sort_node: SorterGPS,
        input_data: pd.DataFrame,
        expected_data: pd.DataFrame,
    ):
        output_data = sort_node.fit_transform(input_data)
        assert output_data.equals(expected_data)
