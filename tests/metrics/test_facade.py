from typing import List

import numpy as np
import pandas as pd
import pytest

from gps_activity import ActivityMetricsSession
from gps_activity.abstract import AbstractNode
from gps_activity.models import DataContainer


@pytest.fixture
def clusters_plan_join() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cluster_primary_key": ["1", "2"],
            "plans_primary_key": ["1", "2"],
            "sjoin_temporal_dist": [0, 0],
            "sjoin_spatial_dist": [
                2.9736081149615674,
                8.572556426019199,
            ],
            "sjoin_overall_dist": [
                2.9736081149615674,
                8.572556426019199,
            ],
        },
    )


@pytest.fixture
def empty_clusters_plan_join(clusters_plan_join: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([], columns=clusters_plan_join.columns)


@pytest.fixture
def correct_coverage_stats() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plate_no": ["1", "1", "1", "1", "1"],
            "date": ["2022-05-10", "2022-05-10", "2022-05-10", "2022-05-10", "2022-05-10"],
            "n_records_gps": [2434, 2352, 1838, 2534, 2628],
            "n_records_plan": [57, 75, 16, 18, 14],
            "action_required": [
                "Keep as is",
                "Keep as is",
                "Keep as is",
                "Keep as is",
                "Keep as is",
            ],
        },
    )


@pytest.fixture
def incorrect_coverage_stats() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plate_no": ["1", "1", "1", "1", "1"],
            "date": ["2022-05-10", "2022-05-10", "2022-05-10", "2022-05-10", "2022-05-10"],
            "n_records_gps": [np.NaN, 2352, 1838, 2534, 2628],
            "n_records_plan": [57, 75, 16, 18, 14],
            "action_required": [
                "Keep as is",
                "Keep as is",
                "Keep as is",
                "Keep as is",
                "Keep as is",
            ],
        },
    )


@pytest.fixture
def clusters() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cluster_primary_key": [
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
            ],
        },
    )


@pytest.fixture
def plan() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plans_primary_key": [
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
            ],
        },
    )


@pytest.fixture
def metrics_session() -> ActivityMetricsSession:
    return ActivityMetricsSession(beta=1)


@pytest.fixture
def correct_data_container(
    clusters: pd.DataFrame,
    plan: pd.DataFrame,
    clusters_plan_join: pd.DataFrame,
    correct_coverage_stats: pd.DataFrame,
) -> pd.DataFrame:
    return DataContainer(
        gps=pd.DataFrame({}),
        clusters=clusters,
        plan=plan,
        clusters_plan_join=clusters_plan_join,
        coverage_stats=correct_coverage_stats,
    )


@pytest.fixture
def incorrect_data_container(
    clusters: pd.DataFrame,
    plan: pd.DataFrame,
    clusters_plan_join: pd.DataFrame,
    incorrect_coverage_stats: pd.DataFrame,
) -> pd.DataFrame:
    return DataContainer(
        gps=pd.DataFrame({}),
        clusters=clusters,
        plan=plan,
        clusters_plan_join=clusters_plan_join,
        coverage_stats=incorrect_coverage_stats,
    )


@pytest.fixture
def no_cluster_plan_join_data_container(
    clusters: pd.DataFrame,
    plan: pd.DataFrame,
    empty_clusters_plan_join: pd.DataFrame,
    correct_coverage_stats: pd.DataFrame,
) -> pd.DataFrame:
    return DataContainer(
        gps=pd.DataFrame({}),
        clusters=clusters,
        plan=plan,
        clusters_plan_join=empty_clusters_plan_join,
        coverage_stats=incorrect_coverage_stats,
    )


@pytest.fixture
def expected_results() -> List[float]:
    return [0.2, 0.2, 0.08 / 0.4]


@pytest.fixture
def tol() -> List[float]:
    return 10 ** (-5)


class TestActivityMetricsSession:
    def test_instance(
        self,
        metrics_session: ActivityMetricsSession,
    ):
        assert isinstance(metrics_session, AbstractNode)

    def test_fit_transform_activity_metrics(
        self,
        correct_data_container: DataContainer,
        metrics_session: ActivityMetricsSession,
        expected_results: List[float],
        tol: float,
    ):
        metrics = metrics_session.fit_transform(correct_data_container)
        assert np.allclose(
            [metrics.recall, metrics.precision, metrics.fbeta_score],
            expected_results,
            atol=tol,
            rtol=tol,
        )

    def test_incorrect_coverage_stats(
        self,
        incorrect_data_container: DataContainer,
        metrics_session: ActivityMetricsSession,
        expected_results: List[float],
        tol: float,
    ):
        try:
            metrics_session.transform(incorrect_data_container)
            raise AssertionError("Must fail because non-overlaps in coverage stats")
        except ValueError:
            assert True

    def test_corner_case_no_cluster_plan_join(
        self,
        no_cluster_plan_join_data_container: DataContainer,
        metrics_session: ActivityMetricsSession,
        expected_results: List[float],
        tol: float,
    ):
        metrics = metrics_session.transform(no_cluster_plan_join_data_container)
        assert metrics.status == "fail"
        assert ActivityMetricsSession.NO_JOINS_STATUS_DETAILS == metrics.status_details
