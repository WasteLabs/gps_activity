from typing import List
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from gps_activity import ActivityExtractionSession
from gps_activity.abstract import AbstractPredictor
from gps_activity.extraction.factory.clustering import FDBSCANFactory
from gps_activity.extraction.factory.fragmentation import VelocityFragmentationFactory
from gps_activity.extraction.factory.preprocessing import PreprocessingFactory
from gps_activity.models import DataFramePivotFields


@pytest.fixture
def preprocessing(
    gps_pivot_fields: DataFramePivotFields,
    wsg_84: str,
    hk_crs: str,
) -> Pipeline:
    return PreprocessingFactory.factory_pipeline(
        source_lat_column=gps_pivot_fields.source_lat,
        source_lon_column=gps_pivot_fields.source_lon,
        source_datetime=gps_pivot_fields.source_datetime,
        source_vehicle_id=gps_pivot_fields.source_vehicle_id,
        source_crs=wsg_84,
        target_crs=hk_crs,
    )


@pytest.fixture
def fragmentation() -> Pipeline:
    return VelocityFragmentationFactory.factory_pipeline(max_velocity_hard_limit=4)


@pytest.fixture
def clustering(gps_pivot_fields: DataFramePivotFields) -> Pipeline:
    return FDBSCANFactory.factory_pipeline(
        source_vehicle_id_column=gps_pivot_fields.source_vehicle_id,
        eps=30,
        min_samples=3,
    )


@pytest.fixture
def vfhdbscan(
    preprocessing: Pipeline,
    fragmentation: Pipeline,
    clustering: Pipeline,
) -> ActivityExtractionSession:
    return ActivityExtractionSession(
        preprocessing=preprocessing,
        fragmentation=fragmentation,
        clustering=clustering,
    )


@pytest.fixture
def expected_columns(gps_pivot_fields: DataFramePivotFields) -> List[str]:
    return [
        gps_pivot_fields.source_datetime,
        gps_pivot_fields.source_lat,
        gps_pivot_fields.source_lon,
        gps_pivot_fields.source_vehicle_id,
        gps_pivot_fields.computed_unixtime,
        gps_pivot_fields.computed_velocity,
        gps_pivot_fields.clustering_output,
        gps_pivot_fields.fragmentation_output,
    ]


class TestActivityExtractionSession:
    def __assert_columns(
        self,
        expected_columns: List[str],
        gps_clustered: pd.DataFrame,
    ):
        for expected_column in expected_columns:
            assert expected_column in gps_clustered.columns

    def test_instance(
        self,
        vfhdbscan: ActivityExtractionSession,
    ):
        assert isinstance(vfhdbscan, AbstractPredictor)

    def test_pipeline_run(
        self,
        vfhdbscan: ActivityExtractionSession,
        gps_sample_alba_scl_weee: pd.DataFrame,
        expected_columns: List[str],
    ):
        gps_clustered = vfhdbscan.predict(gps_sample_alba_scl_weee)
        self.__assert_columns(expected_columns, gps_clustered)

    def test_fragmentation_preprocessing(
        self,
        vfhdbscan: ActivityExtractionSession,
        gps_sample_alba_scl_weee: pd.DataFrame,
    ):
        vfhdbscan.get_fragmentation_input(gps_sample_alba_scl_weee)

    def test_clustering_preprocessing(
        self,
        vfhdbscan: ActivityExtractionSession,
        gps_sample_alba_scl_weee: pd.DataFrame,
    ):
        vfhdbscan.get_clustering_input(gps_sample_alba_scl_weee)

    def test_classification_preprocessing(
        self,
        vfhdbscan: ActivityExtractionSession,
        gps_sample_alba_scl_weee: pd.DataFrame,
    ):
        vfhdbscan.get_classification_input(gps_sample_alba_scl_weee)
