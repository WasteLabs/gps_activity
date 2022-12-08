import pandas as pd
from sklearn.pipeline import Pipeline

from .conftest import ensure_ignore_non_cluster_candidates


class TestFDBSCAN:
    def __test_cluster_count(
        self,
        gps: pd.DataFrame,
        cluster_id_col: str,
        expected_cluster_count: int,
    ):
        assert gps[cluster_id_col].nunique() == expected_cluster_count

    def test_instance(self, fdbscan_pipeline: Pipeline):
        assert isinstance(fdbscan_pipeline, Pipeline)

    # flake8: noqa: CFQ002
    def test_pipeline(
        self,
        clustering_gps_points: pd.DataFrame,
        fdbscan_pipeline: Pipeline,
        clustering_output_column: str,
        gps_pivot_fields,
        noise_impute_value: int,
        expected_cluster_count: int,
    ):
        y_pred = fdbscan_pipeline.fit_predict(clustering_gps_points)
        clustering_gps_points[clustering_output_column] = y_pred
        self.__test_cluster_count(
            clustering_gps_points,
            clustering_output_column,
            expected_cluster_count=expected_cluster_count,
        )
        ensure_ignore_non_cluster_candidates(
            clustering_gps_points,
            gps_pivot_fields.fragmentation_output,
            clustering_output_column,
            noise_impute_value,
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
