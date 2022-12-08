import pandas as pd
import pytest

from gps_activity.extraction.factory.clustering import FDBSCANFactory
from gps_activity.extraction.factory.clustering import STCMFactory
from gps_activity.models import DataFramePivotFields
from gps_activity.models import DefaultValues

default_values = DefaultValues()
pivot_fields = DataFramePivotFields()


@pytest.fixture
def clustering_gps_points():
    x = list(range(10)) + [40, 80, 120] + list(range(150, 160))
    y = list(range(len(x)))
    unixtime = list(range(len(x)))
    cluster_candidates = [True] * 5 + [False] * 5 + [True] * 13
    df = pd.DataFrame(
        {
            pivot_fields.projected_lat: x,
            pivot_fields.projected_lon: y,
            pivot_fields.computed_unixtime: unixtime,
            pivot_fields.fragmentation_output: cluster_candidates,
        },
    )
    df["garbage1"] = "123"
    df["garbage2"] = "123"
    df[pivot_fields.source_vehicle_id] = "123"
    return df


@pytest.fixture
def expected_cluster_count() -> int:
    return 3


@pytest.fixture
def many_vehicle_gps(clustering_gps_points: pd.DataFrame):
    df = clustering_gps_points
    plate_no = df[pivot_fields.source_vehicle_id].to_list()
    plate_no[-1] = "321"
    df[pivot_fields.source_vehicle_id] = plate_no
    return df


@pytest.fixture
def fdbscan_pipeline(gps_pivot_fields: DataFramePivotFields):
    return FDBSCANFactory.factory_pipeline(
        source_vehicle_id_column=gps_pivot_fields.source_vehicle_id,
        eps=2,
        min_samples=3,
    )


@pytest.fixture
def stcm_pipeline(gps_pivot_fields: DataFramePivotFields):
    return STCMFactory.factory_pipeline(
        source_vehicle_id_column=gps_pivot_fields.source_vehicle_id,
        eps=1,
        min_duration_sec=2,
    )


@pytest.fixture
def clustering_output_column(gps_pivot_fields: DataFramePivotFields):
    return DataFramePivotFields().clustering_output


@pytest.fixture
def noise_impute_value(gps_pivot_fields: DataFramePivotFields):
    return DefaultValues().noise_gps_cluster_id


@pytest.fixture
def sctm_test_candidate() -> pd.DataFrame:
    """
    corner case with skip point at the middle
    """
    return pd.DataFrame(
        {
            pivot_fields.projected_lat: [0, 0, 0, 4, 5, 5, 5],
            pivot_fields.projected_lon: [0, 1, 2, 6, 8, 9, 10],
            pivot_fields.computed_unixtime: [0, 1, 2, 4, 5, 6, 7],
            pivot_fields.clustering_output: [0] * 3 + [-1] + [1] * 3,
            pivot_fields.fragmentation_output: [True] * 7,
            pivot_fields.source_vehicle_id: ["123"] * 7,
            "dummy_field_1": ["123"] * 7,
            "dummy_field_2": ["123"] * 7,
        },
    )


def ensure_ignore_non_cluster_candidates(
    gps: pd.DataFrame,
    fragmentation_flag: str,
    cluster_cand_col: str,
    noise_impute_value: int,
):
    cluster_ids = gps.loc[~gps[fragmentation_flag], cluster_cand_col]
    assert (cluster_ids == noise_impute_value).all()
