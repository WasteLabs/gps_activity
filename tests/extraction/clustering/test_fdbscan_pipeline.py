import pandas as pd
from sklearn.pipeline import Pipeline

from gps_activity.models import DataFramePivotFields
from gps_activity.models import DefaultValues


default_values = DefaultValues()
pivot_fields = DataFramePivotFields()


class TestFDBSCAN:

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

    def test_instance(self, fdbscan_pipeline: Pipeline):
        assert isinstance(fdbscan_pipeline, Pipeline)

    def test_pipeline(
        self,
        clustering_gps_points: pd.DataFrame,
        fdbscan_pipeline: Pipeline,
    ):
        y_pred = fdbscan_pipeline.fit_predict(clustering_gps_points)
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
        fdbscan_pipeline: Pipeline,
    ):
        try:
            fdbscan_pipeline.fit_predict(many_vehicle_gps)
            raise AssertionError("Must fail due to single vehicle constrain")
        except ValueError:
            assert True
