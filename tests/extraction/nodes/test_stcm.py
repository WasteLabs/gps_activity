import numpy as np
import pandas as pd
import pytest

from gps_activity.abstract import AbstractPredictor
from gps_activity.models import DataFramePivotFields
from gps_activity.models import DefaultValues
from gps_activity.extraction.nodes import STCM


pivot_fields = DataFramePivotFields()
defaults = DefaultValues()


def gps_sample_1() -> pd.DataFrame:
    """
    general test
    """
    return pd.DataFrame(
        {
            pivot_fields.projected_lat: [0, 0, 0, 6, 6, 8, 9],
            pivot_fields.projected_lon: [0, 1, 2, 6, 7, 8, 9],
            pivot_fields.computed_unixtime: [0, 1, 2, 4, 5, 6, 7],
            pivot_fields.fragmentation_output: [True] * 7,
            pivot_fields.clustering_output: [0] * 3 + [-1] * 4,
        },
    )


def gps_sample_2() -> pd.DataFrame:
    """
    corner case with skip point at the middle
    """
    return pd.DataFrame(
        {
            pivot_fields.projected_lat: [0, 0, 0, 4, 5, 5, 5],
            pivot_fields.projected_lon: [0, 1, 2, 6, 8, 9, 10],
            pivot_fields.computed_unixtime: [0, 1, 2, 4, 5, 6, 7],
            pivot_fields.fragmentation_output: [True] * 7,
            pivot_fields.clustering_output: [0] * 3 + [-1] + [1] * 3,
        },
    )


def gps_sample_3() -> pd.DataFrame:
    """
    test fragmentation flag
    """
    return pd.DataFrame(
        {
            pivot_fields.projected_lat: [0, 0, 0, 4, 5, 5, 5],
            pivot_fields.projected_lon: [0, 1, 2, 6, 8, 9, 10],
            pivot_fields.computed_unixtime: [0, 1, 2, 4, 5, 6, 7],
            pivot_fields.fragmentation_output: [True, True, True] + [False] * 4,
            pivot_fields.clustering_output: [0] * 3 + [-1] * 4,
        },
    )


@pytest.fixture
def stcm() -> STCM:
    """
    test fragmentation flag
    """
    return STCM(eps=1, min_duration_sec=2)


class TestSTCM:
    def test_instance(self, stcm):
        assert isinstance(stcm, AbstractPredictor)

    @pytest.mark.parametrize("gps", [gps_sample_1(), gps_sample_2(), gps_sample_3()])
    def test_stcm(self, stcm: STCM, gps: pd.DataFrame, round_tolerance: float):
        clusters_predicted = stcm.fit_predict(gps)
        clusters_expected = gps[pivot_fields.clustering_output].values
        print(f"{clusters_predicted} vs {clusters_expected}")
        assert np.allclose(
            clusters_expected,
            clusters_predicted,
            atol=round_tolerance,
            rtol=round_tolerance,
        )
