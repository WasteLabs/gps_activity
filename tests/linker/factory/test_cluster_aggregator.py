import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from gps_activity.linker.factory import ClusterAggregationFactory
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
            "cluster_primary_key": ["123_0", "123_1"],
        },
    )


@pytest.fixture
def aggregation_pipeline(
    gps_pivot_fields: DataFramePivotFields,
    wsg_84: str,
    hk_crs: str,
) -> Pipeline:
    return ClusterAggregationFactory.factory_pipeline(
        source_lat_column=gps_pivot_fields.source_lat,
        source_lon_column=gps_pivot_fields.source_lon,
        source_datetime=gps_pivot_fields.source_datetime,
        source_vehicle_id=gps_pivot_fields.source_vehicle_id,
        source_crs=wsg_84,
        target_crs=hk_crs,
    )


class TestClusterAggregationFactory:
    def __check_expectations(self, expected: pd.DataFrame, computed: pd.DataFrame):
        columns_sequence = list(expected.columns)
        computed = computed[columns_sequence]
        expected = expected[columns_sequence]
        assert expected.equals(computed)

    def test_instance(self, aggregation_pipeline: Pipeline):
        assert isinstance(aggregation_pipeline, Pipeline)

    def test_transform_aggregate(
        self,
        aggregation_pipeline: Pipeline,
        gps_input: pd.DataFrame,
        clusters_expected: pd.DataFrame,
    ):
        computed_clusters = aggregation_pipeline.transform(gps_input)
        self.__check_expectations(clusters_expected, computed_clusters)

    def test_fit_transform_aggregate(
        self,
        aggregation_pipeline: Pipeline,
        gps_input: pd.DataFrame,
        clusters_expected: pd.DataFrame,
    ):
        computed_clusters = aggregation_pipeline.fit_transform(gps_input)
        self.__check_expectations(clusters_expected, computed_clusters)
