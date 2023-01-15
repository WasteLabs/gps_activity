import pandas as pd
import pytest

from gps_activity.abstract import AbstractNode
from gps_activity.metrics.nodes import Recall
from gps_activity.models import LinkerDataContainer


@pytest.fixture
def clusters_plan_join() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cluster_primary_key": ["1", "1"],
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
def data_container(plan: pd.DataFrame, clusters_plan_join: pd.DataFrame) -> pd.DataFrame:
    return LinkerDataContainer(
        gps=pd.DataFrame({}),
        plan=plan,
        clusters_plan_join=clusters_plan_join,
    )


@pytest.fixture
def recall() -> pd.DataFrame:
    return Recall()


@pytest.fixture
def expected_recall() -> pd.DataFrame:
    return 0.2


class TestRecall:
    def test_instance(self, recall: Recall):
        assert isinstance(recall, AbstractNode)

    def test_estimate(
        self,
        recall: Recall,
        data_container: LinkerDataContainer,
        expected_recall: float,
    ):

        computed_recall = recall.fit_transform(data_container)
        assert computed_recall == expected_recall
