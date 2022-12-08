import logging
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .conftest import ensure_ignore_non_cluster_candidates


class TestSTCM:
    def test_instance(self, stcm_pipeline: Pipeline):
        assert isinstance(stcm_pipeline, Pipeline)

    def test_pipeline(
        self,
        clustering_gps_points: pd.DataFrame,
        stcm_pipeline: Pipeline,
        clustering_output_column: str,
        gps_pivot_fields,
        noise_impute_value: int,
    ):
        y_pred = stcm_pipeline.fit_predict(clustering_gps_points)
        clustering_gps_points[clustering_output_column] = y_pred
        ensure_ignore_non_cluster_candidates(
            clustering_gps_points,
            gps_pivot_fields.fragmentation_output,
            clustering_output_column,
            noise_impute_value,
        )

    def test_unique_vehicle_constrain(
        self,
        many_vehicle_gps: pd.DataFrame,
        stcm_pipeline: Pipeline,
    ):
        try:
            stcm_pipeline.fit_predict(many_vehicle_gps)
            raise AssertionError("Must fail due to single vehicle constrain")
        except ValueError:
            assert True

    def test_clustering_correctness(
        self,
        sctm_test_candidate: pd.DataFrame,
        stcm_pipeline: Pipeline,
        gps_pivot_fields,
        round_tolerance: float,
    ):
        predictions = stcm_pipeline.fit_predict(sctm_test_candidate.copy())
        logging.info(f"predictions: {predictions}")
        logging.info(f"expected: {sctm_test_candidate[gps_pivot_fields.clustering_output]}")
        assert np.allclose(
            predictions,
            sctm_test_candidate[gps_pivot_fields.clustering_output],
            atol=round_tolerance,
            rtol=round_tolerance,
        )
