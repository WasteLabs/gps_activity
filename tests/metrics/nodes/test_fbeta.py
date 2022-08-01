import numpy as np
import pandas as pd
import pytest

from gps_activity.abstract import AbstractNode
from gps_activity.metrics.nodes import Fbeta
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
def data_container(
    clusters: pd.DataFrame,
    plan: pd.DataFrame,
    clusters_plan_join: pd.DataFrame,
) -> pd.DataFrame:
    return DataContainer(
        gps=pd.DataFrame({}),
        clusters=clusters,
        plan=plan,
        clusters_plan_join=clusters_plan_join,
    )


@pytest.fixture
def fbeta() -> Fbeta:
    return Fbeta(beta=1)


@pytest.fixture
def expected_fbeta() -> pd.DataFrame:
    return 0.08 / 0.4


class TestPrecision:
    def test_instance(self, fbeta: Fbeta):
        assert isinstance(fbeta, AbstractNode)

    def test_estimate(
        self,
        fbeta: Fbeta,
        data_container: DataContainer,
        expected_fbeta: float,
    ):

        computed_fbeta = fbeta.fit_transform(data_container)
        assert np.allclose(
            [computed_fbeta],
            [expected_fbeta],
            rtol=10 ** (-5),
            atol=10 ** (-5),
        )
