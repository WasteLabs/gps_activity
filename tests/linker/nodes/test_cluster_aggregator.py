import pandas as pd
import pytest

from gps_activity.abstract import AbstractNode
from gps_activity.linker.nodes import ClusterAggregator
from gps_activity.models import DataFramePivotFields


@pytest.fixture
def gps_input() -> pd.DataFrame:
    sample = {
        "lat": [0, 0, 0, 0, 2, 2, 3],
        "lon": [0, 0, 0, 2, 0, 2, 3],
        "x": [0, 0, 0, 2, 0, 2, 3],
        "y": [0, 0, 0, 0, 2, 2, 3],
        "cluster_id": [-1, -1, 0, 0, 0, 0, 1],
    }
    sample["plate_no"] = ["123"] * len(sample["lat"])
    sample["date"] = ["2022-07-22"] * len(sample["lat"])
    return pd.DataFrame(sample)


@pytest.fixture
def clusters_expected() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plate_no": ["123", "123"],
            "date": ["2022-07-22", "2022-07-22"],
            "cluster_id": [0, 1],
            "lat": [1.0, 3.0],
            "lon": [1.0, 3.0],
            "x": [1.0, 3.0],
            "y": [1.0, 3.0],
        },
    )


@pytest.fixture
def cluster_aggregator(gps_pivot_fields: DataFramePivotFields) -> ClusterAggregator:
    return ClusterAggregator(pivot_fields=gps_pivot_fields)


class TestClusterAggregator:
    def __check_expectations(self, expected: pd.DataFrame, computed: pd.DataFrame):
        columns_sequence = list(expected.columns)
        computed = computed[columns_sequence].astype(str)
        expected = expected[columns_sequence].astype(str)
        assert expected.equals(computed)

    def test_instance(self, cluster_aggregator: ClusterAggregator):
        assert isinstance(cluster_aggregator, AbstractNode)

    def test_transform_aggregate(
        self,
        cluster_aggregator: ClusterAggregator,
        gps_input: pd.DataFrame,
        clusters_expected: pd.DataFrame,
    ):
        computed_clusters = cluster_aggregator.transform(gps_input)
        self.__check_expectations(computed_clusters, clusters_expected)

    def test_fit_transform_aggregate(
        self,
        cluster_aggregator: ClusterAggregator,
        gps_input: pd.DataFrame,
        clusters_expected: pd.DataFrame,
    ):
        computed_clusters = cluster_aggregator.fit_transform(gps_input)
        self.__check_expectations(computed_clusters, clusters_expected)
