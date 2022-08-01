import numpy as np
import pandas as pd
import pytest

from gps_activity.abstract import AbstractNode
from gps_activity.linker.nodes import CoverageStatistics
from gps_activity.models import DataContainer


@pytest.fixture
def gps() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["1", "2"],
            "plate_no": ["1", "4"],
        },
    )


@pytest.fixture
def plan() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["1", "2"],
            "plate_no": ["1", "1"],
        },
    )


@pytest.fixture
def expected_statistics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plate_no": ["1", "4", "1"],
            "date": ["1", "2", "2"],
            "n_records_gps": [1.0, 1.0, np.NaN],
            "n_records_plan": [1.0, np.NaN, 1.0],
            "action_required": [
                "Keep as is",
                "Drop vehicle-date in  _gps",
                "Drop vehicle-date in  _plan",
            ],
        },
    )


@pytest.fixture
def data_container(gps: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    return DataContainer.factory_instance(X={"gps": gps, "plan": plan})


@pytest.fixture
def coverage_stats() -> CoverageStatistics:
    return CoverageStatistics()


class TestCoverageStatistics:
    """
    Table describes how provided gps covers provided route plan
    """

    def test_instance(self, coverage_stats: CoverageStatistics):
        assert isinstance(coverage_stats, AbstractNode)

    def __preprocess_for_compare(self, X):
        return X.fillna("").astype(str)

    def test_fit_transform_coverage_stats(
        self,
        coverage_stats: CoverageStatistics,
        data_container: DataContainer,
        expected_statistics: pd.DataFrame,
    ):
        computed_stats = coverage_stats.fit_transform(data_container)
        computed_stats = self.__preprocess_for_compare(computed_stats)
        expected_statistics = self.__preprocess_for_compare(expected_statistics)
        assert expected_statistics.equals(computed_stats)
