import pandas as pd
import pytest

from gps_activity.abstract import AbstractNode
from gps_activity.metrics.nodes import Precision
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
def data_container(clusters: pd.DataFrame, clusters_plan_join: pd.DataFrame) -> pd.DataFrame:
    return DataContainer(
        gps=pd.DataFrame({}),
        clusters=clusters,
        plan=pd.DataFrame({}),
        clusters_plan_join=clusters_plan_join,
    )


@pytest.fixture
def precision() -> Precision():
    return Precision()


@pytest.fixture
def expected_precision() -> pd.DataFrame:
    return 0.2


class TestPrecision:
    def test_instance(self, precision: Precision):
        assert isinstance(precision, AbstractNode)

    def test_estimate(
        self,
        precision: Precision,
        data_container: DataContainer,
        expected_precision: float,
    ):

        computed_precision = precision.fit_transform(data_container)
        assert computed_precision == expected_precision
