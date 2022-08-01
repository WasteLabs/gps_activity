import pandas as pd
import pytest

from gps_activity.abstract import AbstractPredictor
from gps_activity.extraction.nodes import FDBSCAN
from gps_activity.models import DataFramePivotFields
from gps_activity.models import DefaultValues


internal_columns = DataFramePivotFields()
defaults = DefaultValues()


@pytest.fixture
def cluster_candidate_flag() -> str:
    return "is_cluster_candidate"


@pytest.fixture
def clustering_gps_points(cluster_candidate_flag: str):
    x = list(range(10)) + [40, 80, 120] + list(range(150, 160))
    y = list(range(len(x)))
    unixtime = list(range(len(x)))
    cluster_candidates = [True] * 5 + [False] * 5 + [True] * 13
    return pd.DataFrame(
        {
            "x": x,
            "y": y,
            "unixtime": unixtime,
            cluster_candidate_flag: cluster_candidates,
        },
    )


@pytest.fixture
def fdbscan(cluster_candidate_flag: str) -> FDBSCAN:
    return FDBSCAN(
        clustering_candidate_col=cluster_candidate_flag,
        eps=2,
        min_samples=3,
    )


class TestFDBSCAN:

    CLUSTER_ID = internal_columns.clustering_output
    NOISE_IMPUTE_VALUE = defaults.noise_gps_cluster_id

    def __test_cluster_count(self, gps: pd.DataFrame):
        assert gps[self.CLUSTER_ID].nunique() == 3

    def __ensure_ignore_non_cluster_candidates(self, gps: pd.DataFrame, cluster_cand_col: str):
        cluster_ids = gps.loc[~gps[cluster_cand_col], self.CLUSTER_ID]
        assert (cluster_ids == self.NOISE_IMPUTE_VALUE).all()

    def test_instance(self, fdbscan: FDBSCAN):
        assert isinstance(fdbscan, AbstractPredictor)

    def test_expectations(
        self,
        clustering_gps_points: pd.DataFrame,
        fdbscan: FDBSCAN,
        cluster_candidate_flag: str,
    ):
        clustering_gps_points[self.CLUSTER_ID] = fdbscan.fit_predict(clustering_gps_points)
        self.__test_cluster_count(clustering_gps_points)
        self.__ensure_ignore_non_cluster_candidates(
            clustering_gps_points,
            cluster_candidate_flag,
        )
