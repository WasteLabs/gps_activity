import logging
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from gps_activity.linker.factory import CoverageStatisticsFactory
from gps_activity.models import LinkerDataContainer


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
def expected_statistics(gps_pivot_fields, default_values) -> pd.DataFrame:
    _gps_col = (
        f"{default_values.sjoin_cov_stat_agg_column}{default_values.pk_delimiter}" f"{default_values.sjoin_gps_suffix}"
    )
    _plan_col = (
        f"{default_values.sjoin_cov_stat_agg_column}{default_values.pk_delimiter}" f"{default_values.sjoin_plan_suffix}"
    )
    return pd.DataFrame(
        {
            gps_pivot_fields.source_vehicle_id: ["1", "4", "1"],
            gps_pivot_fields.projected_date: ["1", "2", "2"],
            _gps_col: [1.0, 1.0, np.NaN],
            _plan_col: [1.0, np.NaN, 1.0],
            default_values.sjoin_cov_stat_action_field: [
                default_values.sjoin_cov_stat_action_default,
                f"{default_values.sjoin_cov_stat_action_required} {default_values.sjoin_gps_suffix}",
                f"{default_values.sjoin_cov_stat_action_required} {default_values.sjoin_plan_suffix}",
            ],
        },
    )


@pytest.fixture
def data_container(gps: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    return LinkerDataContainer(gps=gps, plan=plan)


@pytest.fixture
def coverage_stats() -> Pipeline:
    return CoverageStatisticsFactory.factory_pipeline()


class TestCoverageStatistics:
    """
    Table describes how provided gps covers provided route plan
    """

    def test_instance(self, coverage_stats: Pipeline):
        assert isinstance(coverage_stats, Pipeline)

    def __preprocess_for_compare(self, X):
        return X.fillna("").astype(str)

    def test_fit_transform_coverage_stats(
        self,
        coverage_stats: Pipeline,
        data_container: LinkerDataContainer,
        expected_statistics: pd.DataFrame,
    ):
        computed_stats = coverage_stats.fit_transform(data_container)
        computed_stats = self.__preprocess_for_compare(computed_stats)
        expected_statistics = self.__preprocess_for_compare(expected_statistics)
        logging.debug(f"\n\nComputed:\n{computed_stats.columns}\n{computed_stats}")
        logging.debug(f"\n\nExpected:\n{expected_statistics.columns}\n{expected_statistics}")
        assert expected_statistics.equals(computed_stats)
