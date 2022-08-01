import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from gps_activity import ActivityLinkageSession
from gps_activity.abstract import AbstractNode
from gps_activity.linker import factory
from gps_activity.models import DataFramePivotFields


@pytest.fixture
def gps() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": [
                "2022-05-03 08:23:47",
                "2022-05-03 08:44:05",
                "2022-05-03 08:44:12",
                "2022-05-03 08:44:22",
                "2022-05-03 08:44:32",
                "2022-05-03 09:10:56",
                "2022-05-03 09:11:06",
                "2022-05-03 09:11:16",
                "2022-05-03 09:11:26",
                "2022-05-03 09:11:36",
                "2022-05-03 09:11:46",
                "2022-05-03 09:11:56",
            ],
            "plate_no": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "lat": [
                22.353788,
                22.353788,
                22.353765,
                22.353766,
                22.35377,
                22.339838,
                22.34035,
                22.341065,
                22.34135,
                22.341401,
                22.341423,
                22.341415,
            ],
            "lon": [
                114.1262,
                114.1262,
                114.12616,
                114.12616,
                114.12616,
                114.139576,
                114.138798,
                114.138313,
                114.138165,
                114.138183,
                114.138218,
                114.13823,
            ],
            "cluster_id": [
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                -1.0,
                -1.0,
                2.0,
                2.0,
                2.0,
                2.0,
            ],
        },
    )


@pytest.fixture
def plans() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CRN#": [0, 1, 2, 3, 4],
            "lat": [22.3068475, 22.3062673, 22.3055948, 22.3079302, 22.3043566],
            "lng": [114.1882672, 114.1872456, 114.1904037, 114.1935618, 114.1845771],
            "datetime": [
                "2022-05-03",
                "2022-05-03",
                "2022-05-03",
                "2022-05-03",
                "2022-05-03",
            ],
            "plate_no": [1, 1, 1, 1, 1],
        },
    )


@pytest.fixture
def pivot_fields() -> DataFramePivotFields:
    return DataFramePivotFields()


@pytest.fixture
def gps_link_preprocess_pipeline(wsg_84: str, hk_crs: str) -> Pipeline:
    return factory.PreprocessingFactory.factory_pipeline(
        source_lat_column="lat",
        source_lon_column="lon",
        source_datetime="datetime",
        source_vehicle_id="plate_no",
        source_crs=wsg_84,
        target_crs=hk_crs,
        generate_primary_key_for="gps",
        source_composite_keys=["plate_no", "datetime", "lat", "lon"],
    )


@pytest.fixture
def plans_link_preprocess_pipeline(wsg_84: str, hk_crs: str) -> Pipeline:
    return factory.PreprocessingFactory.factory_pipeline(
        source_lat_column="lat",
        source_lon_column="lng",
        source_datetime="datetime",
        source_vehicle_id="plate_no",
        source_crs=wsg_84,
        target_crs=hk_crs,
        generate_primary_key_for="plan",
        source_composite_keys=["CRN#"],
    )


@pytest.fixture
def cluster_agg_pipeline(wsg_84: str, hk_crs: str) -> Pipeline:
    return factory.ClusterAggregationFactory.factory_pipeline(
        source_lat_column="lat",
        source_lon_column="lon",
        source_datetime="datetime",
        source_vehicle_id="plate_no",
        source_crs=wsg_84,
        target_crs=hk_crs,
    )


@pytest.fixture
def spatial_joiner():
    return factory.SpatialJoinerFactory.factory_pipeline(
        how="inner",
        max_distance=80,
    )


@pytest.fixture
def coverage_stats_extractor():
    return factory.CoverageStatisticsFactory.factory_pipeline()


@pytest.fixture
def spatial_validator():
    return factory.JoinValidatorFactory.factory_pipeline(
        max_days_distance=1,
        ensure_vehicle_overlap=True,
    )


@pytest.fixture
def linkage_session(
    gps_link_preprocess_pipeline: Pipeline,
    plans_link_preprocess_pipeline: Pipeline,
    cluster_agg_pipeline: Pipeline,
    spatial_joiner: Pipeline,
    spatial_validator: Pipeline,
    coverage_stats_extractor: Pipeline,
) -> ActivityLinkageSession:
    return ActivityLinkageSession(
        gps_preprocessor=gps_link_preprocess_pipeline,
        plan_preprocessor=plans_link_preprocess_pipeline,
        cluster_aggregator=cluster_agg_pipeline,
        spatial_joiner=spatial_joiner,
        spatial_validator=spatial_validator,
        coverage_stats_extractor=coverage_stats_extractor,
    )


class TestActivityLinkageSession:
    def test_instance(
        self,
        linkage_session: ActivityLinkageSession,
        gps: pd.DataFrame,
        plans: pd.DataFrame,
    ):
        assert isinstance(linkage_session, AbstractNode)

    def test_linkage_session_run(
        self,
        linkage_session: ActivityLinkageSession,
        gps: pd.DataFrame,
        plans: pd.DataFrame,
    ):
        linkage_data = linkage_session.fit_transform(X={"gps": gps, "plan": plans})
        assert linkage_data.clusters_plan_join.shape[0] == 0

    def test_compute_coverage_stats(
        self,
        linkage_session: ActivityLinkageSession,
        gps: pd.DataFrame,
        plans: pd.DataFrame,
    ):
        coverage_stats = linkage_session.compute_coverage_stats(X={"gps": gps, "plan": plans})
        assert isinstance(coverage_stats, pd.DataFrame)
        assert coverage_stats.shape[0] == 1
