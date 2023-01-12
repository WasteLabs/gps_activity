from typing import List

import pandas as pd
import pytest

from gps_activity.abstract import AbstractNode
from gps_activity.models import DataFramePivotFields
from gps_activity.nodes import VelocityCalculator


@pytest.fixture
def df_expected(gps_pivot_fields: DataFramePivotFields):
    return pd.DataFrame(
        {
            gps_pivot_fields.projected_lon: [
                818721.6578177342,
                818721.6578177342,
                818721.6578177342,
                818721.6578177342,
                818721.6578177342,
                818721.6578177342,
                818721.6578177342,
                818721.6578177342,
            ],
            gps_pivot_fields.projected_lat: [
                834088.352552602,
                835088.352552602,
                836088.352552602,
                837088.352552602,
                834088.352552602,
                835088.352552602,
                836088.352552602,
                837088.352552602,
            ],
            gps_pivot_fields.computed_unixtime: [
                1650844938.0,
                1650848527.0,
                1650852117.0,
                1650855706.0,
                1650844938.0,
                1650848527.0,
                1650852117.0,
                1650855706.0,
            ],
            gps_pivot_fields.source_vehicle_id: ["1", "1", "1", "1", "2", "2", "2", "2"],
            gps_pivot_fields.computed_velocity: [
                0.27862914460852606,
                0.27862914460852606,
                0.2785515320334262,
                0.27862914460852606,
                0.27862914460852606,
                0.27862914460852606,
                0.2785515320334262,
                0.27862914460852606,
            ],
        },
    )


@pytest.fixture
def df_input(
    df_expected: pd.DataFrame,
    gps_pivot_fields: DataFramePivotFields,
):
    return df_expected.drop(columns=[gps_pivot_fields.computed_velocity])


@pytest.fixture
def df_columns(df_expected: pd.DataFrame):
    return list(df_expected.columns)


@pytest.fixture
def velocity_calculator(gps_pivot_fields: DataFramePivotFields):
    return VelocityCalculator(pivot_fields=gps_pivot_fields)


class TestVelocityCalculator:
    def test_instance(self, velocity_calculator: VelocityCalculator):
        assert isinstance(velocity_calculator, AbstractNode)

    def test_transform_velocity_calculator(
        self,
        velocity_calculator: VelocityCalculator,
        df_input: pd.DataFrame,
        df_expected: pd.DataFrame,
        df_columns: List[str],
        gps_pivot_fields: DataFramePivotFields,
    ):
        df_computed = velocity_calculator.transform(df_input)
        assert df_expected[df_columns].equals(df_computed[df_columns])

    def test_fit_transform_velocity_calculator(
        self,
        velocity_calculator: VelocityCalculator,
        df_input: pd.DataFrame,
        df_expected: pd.DataFrame,
        df_columns: List[str],
        gps_pivot_fields: DataFramePivotFields,
    ):
        df_computed = velocity_calculator.fit_transform(df_input)
        assert df_expected[df_columns].equals(df_computed[df_columns])
