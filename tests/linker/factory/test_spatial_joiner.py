import geopandas as gpd
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from gps_activity.linker.factory import SpatialJoinerFactory
from gps_activity.linker.models import SpatialJoinArguments


@pytest.fixture
def left_table():
    x = [0, 2]
    y = [0, 2]
    df = pd.DataFrame({"x": x, "y": y})
    geometry = gpd.points_from_xy(x=x, y=y)
    return gpd.GeoDataFrame(data=df, geometry=geometry)


@pytest.fixture
def right_table():
    x = [3, 5]
    y = [3, 5]
    df = pd.DataFrame({"x": x, "y": y})
    geometry = gpd.points_from_xy(x=x, y=y)
    return gpd.GeoDataFrame(data=df, geometry=geometry)


@pytest.fixture
def joiner() -> Pipeline:
    return SpatialJoinerFactory.factory_pipeline(how="inner", max_distance=2)


@pytest.fixture
def sjoin_args(
    left_table: gpd.GeoDataFrame,
    right_table: gpd.GeoDataFrame,
) -> SpatialJoinArguments:
    return SpatialJoinArguments(
        clustered_gps=left_table,
        route_plan=right_table,
    )


class TestSpatialJoiner:
    def test_correctness_spatial_join(
        self,
        joiner: Pipeline,
        sjoin_args: SpatialJoinArguments,
    ):
        sjoin = joiner.fit_transform(sjoin_args)
        assert sjoin.shape[0] == 1

    def test_non_geodataframe_instance_sjoin(
        self,
        joiner: Pipeline,
        sjoin_args: SpatialJoinArguments,
    ):
        try:
            sjoin_args.route_plan = None
            joiner.fit_transform(sjoin_args)
            raise AssertionError("Must fail because of non gpd.GeoDataFrame instance")
        except ValueError:
            assert True

    def test_broken_sjoin(
        self,
        joiner: Pipeline,
        sjoin_args: SpatialJoinArguments,
    ):
        try:
            sjoin_args.route_plan = gpd.GeoDataFrame({"x": [0], "y": [1]})
            joiner.fit_transform(sjoin_args)
            raise AssertionError("Must fail because missing geometry")
        except ValueError:
            assert True
