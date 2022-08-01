from typing import List

import pandas as pd
import pytest

from gps_activity.abstract import AbstractPredictor
from gps_activity.extraction.nodes import VelocityFragmentator


@pytest.fixture
def velocity_gps_points() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "velocity": [1, 2, 3, 4, 5, 6],
        },
    )


@pytest.fixture
def expected_results() -> List[bool]:
    return [True, True, True, True, False, False]


@pytest.fixture
def max_velocity_hard_limit() -> int:
    return 4


@pytest.fixture
def source_velocity_column() -> str:
    return "velocity"


@pytest.fixture
def velocity_fragmentator(
    source_velocity_column: str,
    max_velocity_hard_limit: int,
) -> VelocityFragmentator:
    return VelocityFragmentator(
        source_velocity_column=source_velocity_column,
        max_velocity_hard_limit=max_velocity_hard_limit,
    )


@pytest.fixture
def broken_velocity_fragmentator(
    source_velocity_column: str,
    max_velocity_hard_limit: int,
) -> VelocityFragmentator:
    return VelocityFragmentator(
        source_velocity_column="not_existing_source_column",
        max_velocity_hard_limit=max_velocity_hard_limit,
    )


class TestVelocityFragmentator:
    def test_instance(self, velocity_fragmentator):
        assert isinstance(velocity_fragmentator, AbstractPredictor)

    def test_prediction(
        self,
        velocity_fragmentator: VelocityFragmentator,
        velocity_gps_points: pd.DataFrame,
        expected_results: List[bool],
    ):
        computed_results = velocity_fragmentator.predict(velocity_gps_points)
        assert expected_results == list(computed_results)

    def test_fit_prediction(
        self,
        velocity_fragmentator: VelocityFragmentator,
        velocity_gps_points: pd.DataFrame,
        expected_results: List[bool],
    ):
        computed_results = velocity_fragmentator.fit_predict(velocity_gps_points)
        assert expected_results == list(computed_results)

    def test_validation(
        self,
        broken_velocity_fragmentator: VelocityFragmentator,
        velocity_gps_points: pd.DataFrame,
    ):
        try:
            broken_velocity_fragmentator.fit_predict(velocity_gps_points)
            raise AssertionError("Valiation failure")
        except KeyError:
            assert True
