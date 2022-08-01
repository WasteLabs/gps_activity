from typing import List

import pandas as pd
import pytest

from gps_activity.abstract import AbstractNode
from gps_activity.nodes import UnixtimeExtractor


@pytest.fixture
def gps_timing_source_points() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": [
                pd.Timestamp("2022-04-25 00:02:18"),
                pd.Timestamp("2022-04-25 01:02:07"),
                pd.Timestamp("2022-04-25 02:01:57"),
            ],
        },
    )


@pytest.fixture
def exptected_delta_sec() -> List[float]:
    return [3589.0, 3590.0]


@pytest.fixture
def gps_timing_src_col() -> str:
    return "datetime"


@pytest.fixture
def gps_timing_trgt_col() -> str:
    return "unixtime"


@pytest.fixture
def unixtime_extractor(
    gps_timing_src_col: str,
    gps_timing_trgt_col: str,
) -> UnixtimeExtractor:
    return UnixtimeExtractor(
        source_column=gps_timing_src_col,
        target_column=gps_timing_trgt_col,
    )


class TestUnixtimeExtractor:
    def __check_expected_time_delta(
        self,
        computed_gps: pd.DataFrame,
        exptected_delta_sec: List[float],
    ):
        delta_sec = computed_gps["unixtime"] - computed_gps["unixtime"].shift(1)
        delta_sec = delta_sec[1:].to_list()
        assert delta_sec == exptected_delta_sec

    def __conduct_test(
        self,
        computed_gps: pd.DataFrame,
        unixtime_extractor: UnixtimeExtractor,
        exptected_delta_sec: pd.Series,
    ):
        assert isinstance(unixtime_extractor, AbstractNode)
        self.__check_expected_time_delta(computed_gps, exptected_delta_sec)

    def test_transform_unixtime_extractor(
        self,
        unixtime_extractor: UnixtimeExtractor,
        gps_timing_source_points: pd.DataFrame,
        exptected_delta_sec: pd.Series,
    ):
        computed_gps = unixtime_extractor.transform(gps_timing_source_points)
        self.__conduct_test(
            computed_gps=computed_gps,
            unixtime_extractor=unixtime_extractor,
            exptected_delta_sec=exptected_delta_sec,
        )

    def test_fit_transform_unixtime_extractor(
        self,
        unixtime_extractor: UnixtimeExtractor,
        gps_timing_source_points: pd.DataFrame,
        exptected_delta_sec: pd.Series,
    ):
        computed_gps = unixtime_extractor.fit_transform(gps_timing_source_points)
        self.__conduct_test(
            computed_gps=computed_gps,
            unixtime_extractor=unixtime_extractor,
            exptected_delta_sec=exptected_delta_sec,
        )
