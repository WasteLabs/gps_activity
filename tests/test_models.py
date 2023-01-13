import pandas as pd
import pytest

from gps_activity.models import LinkerDataContainer
from gps_activity.models import DataFramePivotFields


@pytest.fixture
def gps(gps_pivot_fields: DataFramePivotFields) -> pd.DataFrame:
    return pd.DataFrame(
        {
            gps_pivot_fields.source_lat: [1, 1, 1, 1, 1],
            gps_pivot_fields.source_lon: [1, 1, 1, 1, 1],
            gps_pivot_fields.projected_date: [
                "2022-05-10",
                "2022-05-10",
                "2022-05-10",
                "2022-05-10",
                "2022-05-10",
            ],
            gps_pivot_fields.clustering_output: [0, 0, 0, 0, 0],
            gps_pivot_fields.source_vehicle_id: ["0", "0", "0", "0", "0"],
        },
    )


@pytest.fixture
def clusters() -> pd.DataFrame:
    pivots = DataFramePivotFields()
    return pd.DataFrame(
        {
            pivots.source_lat: [1, 2],
            pivots.source_lon: [1, 2],
            pivots.projected_date: [
                "2022-05-10",
                "2022-05-10",
            ],
            pivots.clustering_output: [0, 1],
            pivots.clusters_pk: ["0_0", "0_1"],
            pivots.source_vehicle_id: ["0", "0"],
        },
    )


@pytest.fixture
def plan() -> pd.DataFrame:
    pivots = DataFramePivotFields()
    return pd.DataFrame(
        {
            pivots.source_lat: [1.05],
            pivots.source_lon: [1.003],
            pivots.projected_date: [
                "2022-05-10",
            ],
            pivots.plans_pk: ["123"],
            pivots.source_vehicle_id: ["0"],
        },
    )


@pytest.fixture
def clusters_plan_join() -> pd.DataFrame:
    pivots = DataFramePivotFields()
    return pd.DataFrame(
        {
            pivots.clusters_pk: ["0_0"],
            pivots.plans_pk: ["123"],
        },
    )


@pytest.fixture
def data_container(
    gps: pd.DataFrame,
    clusters: pd.DataFrame,
    plan: pd.DataFrame,
    clusters_plan_join: pd.DataFrame,
) -> pd.DataFrame:
    return LinkerDataContainer(
        gps=gps,
        plan=plan,
        clusters=clusters,
        clusters_plan_join=clusters_plan_join,
    )


class TestLinkerDataContainer:
    def test_function(
        self,
        data_container: LinkerDataContainer,
        gps: pd.DataFrame,
    ):
        concated_gps = data_container.get_concatenated_gps()
        assert isinstance(concated_gps, pd.DataFrame)
        assert gps.shape[0] == concated_gps.shape[0]
        assert gps.shape[1] < concated_gps.shape[1]
