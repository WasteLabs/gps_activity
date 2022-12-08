import pandas as pd
import pytest

from gps_activity.extraction.factory.clustering import FDBSCANFactory
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
