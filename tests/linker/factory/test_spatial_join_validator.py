import pandas as pd
from pandas import Timestamp
import pytest

from sklearn.pipeline import Pipeline
from gps_activity.linker.factory import JoinValidatorFactory


@pytest.fixture
def input_df():
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
            "lat_gps": [1, 1, 1, 1],
            "lon_gps": [1, 1, 1, 1],
            "cluster_primary_key": ["1", "1", "1", "1"],
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
def expected_df():
    return pd.DataFrame(
        {
            "cluster_primary_key": ["1"],
            "plans_primary_key": ["123"],
            "sjoin_temporal_dist": [0],
            "sjoin_spatial_dist": [10.0],
            "sjoin_overall_dist": [10.0],
        },
    )


@pytest.fixture
def spatial_validator(expected_df: pd.DataFrame):
    sjoin_validator = JoinValidatorFactory.factory_pipeline(
        max_days_distance=2,
        ensure_vehicle_overlap=True,
    )
    return sjoin_validator


class TestSpatialJoinValidatorFactory:
    def test_instance(self, spatial_validator: Pipeline):
        assert isinstance(spatial_validator, Pipeline)

    def test_sjoin_validator(
        self,
        spatial_validator: Pipeline,
        input_df: pd.DataFrame,
        expected_df: pd.DataFrame,
    ):
        computed_df = spatial_validator.fit_transform(input_df)
        expected_df = expected_df.astype(str)
        computed_df = computed_df[expected_df.columns].astype(str)
        assert expected_df.equals(computed_df)
