import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

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
def clustering_pipeline(gps_pivot_fields: DataFramePivotFields):
    return FDBSCANFactory.factory_pipeline(
        source_vehicle_id_column=gps_pivot_fields.source_vehicle_id,
        eps=2,
        min_samples=3,
    )


class TestPreprocessingPipelineFactory:

    CLUSTER_OUPUT_COLUMNS = pivot_fields.clustering_output
    NOISE_IMPUTE_VALUE = default_values.noise_gps_cluster_id

    def __test_cluster_count(self, gps: pd.DataFrame, cluster_id_col: str):
        assert gps[cluster_id_col].nunique() == 3

    def __ensure_ignore_non_cluster_candidates(
        self,
        gps: pd.DataFrame,
        fragmentation_flag: str,
        cluster_cand_col: str,
    ):
        cluster_ids = gps.loc[~gps[fragmentation_flag], cluster_cand_col]
        assert (cluster_ids == self.NOISE_IMPUTE_VALUE).all()

    def test_instance(self, clustering_pipeline: Pipeline):
        assert isinstance(clustering_pipeline, Pipeline)

    def test_pipeline(
        self,
        clustering_gps_points: pd.DataFrame,
        clustering_pipeline: Pipeline,
    ):
        y_pred = clustering_pipeline.fit_predict(clustering_gps_points)
        clustering_gps_points[self.CLUSTER_OUPUT_COLUMNS] = y_pred
        self.__test_cluster_count(
            clustering_gps_points,
            self.CLUSTER_OUPUT_COLUMNS,
        )
        self.__ensure_ignore_non_cluster_candidates(
            clustering_gps_points,
            pivot_fields.fragmentation_output,
            self.CLUSTER_OUPUT_COLUMNS,
        )

    def test_unique_vehicle_constrain(
        self,
        many_vehicle_gps: pd.DataFrame,
        clustering_pipeline: Pipeline,
    ):
        try:
            clustering_pipeline.fit_predict(many_vehicle_gps)
            raise AssertionError("Must fail due to single vehicle constrain")
        except ValueError:
            assert True
