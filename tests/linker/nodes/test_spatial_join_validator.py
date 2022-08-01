import pandas as pd
from pandas import Timestamp
import pytest

from gps_activity.linker.nodes import SpatialJoinValidator


@pytest.fixture
def full_df():
    return pd.DataFrame(
        {
            "date_gps": [
                Timestamp("2022-05-10"),
                Timestamp("2022-05-12"),
                Timestamp("2022-05-13"),
                Timestamp("2022-05-10"),
            ],
            "date_plan": [
                Timestamp("2022-05-10"),
                Timestamp("2022-05-10"),
                Timestamp("2022-05-10"),
                Timestamp("2022-05-10"),
            ],
            "plate_no_gps": ["123", "123", "123", "321"],
            "plate_no_plan": ["123", "123", "123", "123"],
            "gps_primary_key": ["123", "123", "123", "123"],
            "plans_primary_key": ["123", "123", "123", "123"],
            "sjoin_spatial_dist": [10, 10, 10, 10],
            "sjoin_valid_flag": [True, True, False, False],
            "sjoin_temporal_dist": [0, 2, 3, 0],
            "sjoin_overall_dist": [10.0, 30.0, 40.0, 10.0],
        },
    )


@pytest.fixture
def expected_df(full_df: pd.DataFrame):
    return (
        full_df[full_df["sjoin_valid_flag"]]
        .reset_index(drop=True)
        .sort_values(by=["sjoin_overall_dist"])
        .drop_duplicates(subset=["plans_primary_key"])
        .reset_index(drop=True)
    )


@pytest.fixture
def input_df(full_df: pd.DataFrame):
    return full_df.drop(
        columns=["sjoin_valid_flag", "sjoin_temporal_dist"],
    )


@pytest.fixture
def empty_dataframe(input_df: pd.DataFrame):
    return pd.DataFrame([], columns=input_df.columns)


@pytest.fixture
def spatial_validator(expected_df: pd.DataFrame):
    sjoin_validator = SpatialJoinValidator(
        max_days_distance=2,
        ensure_vehicle_overlap=True,
    )
    return sjoin_validator


class TestSpatialJoinValidator:
    def test_instance(self, spatial_validator: SpatialJoinValidator):
        assert isinstance(spatial_validator, SpatialJoinValidator)

    def test_sjoin_validator(
        self,
        spatial_validator: SpatialJoinValidator,
        input_df: pd.DataFrame,
        expected_df: pd.DataFrame,
    ):
        computed_df = spatial_validator.fit_transform(input_df)
        expected_df = expected_df.astype(str)
        computed_df = computed_df[expected_df.columns].astype(str)
        assert expected_df.equals(computed_df)

    def test_corner_case_with_no_sjoin(
        self,
        spatial_validator: SpatialJoinValidator,
        empty_dataframe: pd.DataFrame,
    ):
        spatial_validator.fit_transform(empty_dataframe)
