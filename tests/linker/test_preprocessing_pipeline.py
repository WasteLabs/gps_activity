from typing import List

import numpy as np
import pandas as pd
from pandas import Timestamp
from pandera.errors import SchemaError
import pytest
from sklearn.pipeline import Pipeline

from gps_activity.linker.factory.preprocessing import PreprocessingFactory
from gps_activity.models import DataFramePivotFields


@pytest.fixture
def expected_gps() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": [
                Timestamp("2022-05-10 09:14:46"),
                Timestamp("2022-05-10 09:14:56"),
                Timestamp("2022-05-10 09:15:05"),
                Timestamp("2022-05-10 09:15:15"),
                Timestamp("2022-06-07 09:39:48"),
                Timestamp("2022-06-07 09:39:58"),
                Timestamp("2022-06-07 09:40:08"),
                Timestamp("2022-06-07 09:40:18"),
            ],
            "plate_no": [
                "ST3274",
                "ST3274",
                "ST3274",
                "ST3274",
                "ST3274",
                "ST3274",
                "ST3274",
                "ST3274",
            ],
            "lat": [
                22.342915,
                22.342853,
                22.342801,
                22.342716,
                22.351981,
                22.352263,
                22.35226,
                22.352255,
            ],
            "lon": [
                114.135501,
                114.135516,
                114.13552,
                114.135545,
                114.098151,
                114.098265,
                114.098268,
                114.09827,
            ],
            "x": [
                832006.0419431562,
                832007.5852430757,
                832007.9957134096,
                832010.5684404044,
                828159.0620010251,
                828170.820916482,
                828171.1297591918,
                828171.3354810979,
            ],
            "y": [
                822648.5026237525,
                822641.6366286246,
                822635.8782944878,
                822626.4651119314,
                823653.9593036388,
                823685.1805030879,
                823684.8481363123,
                823684.294352846,
            ],
        },
    )


@pytest.fixture
def input_gps(expected_gps) -> pd.DataFrame:
    return expected_gps.drop(columns=["x", "y"]).astype(str)


@pytest.fixture
def column_sequence(expected_gps: pd.DataFrame):
    return set(expected_gps.columns) - set(["plate_no", "datetime"])


@pytest.fixture
def preprocessing_pipeline(
    gps_pivot_fields: DataFramePivotFields,
    wsg_84: str,
    hk_crs: str,
):
    return PreprocessingFactory.factory_pipeline(
        source_lat_column=gps_pivot_fields.source_lat,
        source_lon_column=gps_pivot_fields.source_lon,
        source_datetime=gps_pivot_fields.source_datetime,
        source_vehicle_id=gps_pivot_fields.source_vehicle_id,
        source_crs=wsg_84,
        target_crs=hk_crs,
        generate_primary_key_for="gps",
        source_composite_keys=[
            gps_pivot_fields.source_lat,
            gps_pivot_fields.source_lon,
            gps_pivot_fields.source_vehicle_id,
        ],
    )


@pytest.fixture
def incomplete_dataframe(
    expected_gps: pd.DataFrame,
    gps_pivot_fields: DataFramePivotFields,
):
    return expected_gps.drop(
        columns=[gps_pivot_fields.source_datetime],
    )


class TestPreprocessingPipelineFactory:
    def test_instance(self, preprocessing_pipeline: Pipeline):
        assert isinstance(preprocessing_pipeline, Pipeline)

    def test_pipeline_transform(
        self,
        input_gps: pd.DataFrame,
        expected_gps: pd.DataFrame,
        preprocessing_pipeline: Pipeline,
        column_sequence: List[str],
        round_tolerance: float,
    ):
        preprocessed_gps = preprocessing_pipeline.transform(input_gps)
        preprocessed_gps = preprocessed_gps[column_sequence].values
        expected_gps = expected_gps[column_sequence].values
        assert np.allclose(
            preprocessed_gps,
            expected_gps,
            atol=round_tolerance,
            rtol=round_tolerance,
        )

    def test_incomplete_data_validation(
        self,
        incomplete_dataframe: pd.DataFrame,
        preprocessing_pipeline: Pipeline,
    ):
        try:
            preprocessing_pipeline.transform(incomplete_dataframe)
            raise AssertionError("Passing broken datetime column")
        except SchemaError:
            assert True

    def test_broken_literal_init(self):
        try:
            _dummy_input = "123"
            PreprocessingFactory.factory_pipeline(
                source_lat_column=_dummy_input,
                source_lon_column=_dummy_input,
                source_datetime=_dummy_input,
                source_vehicle_id=_dummy_input,
                source_crs=_dummy_input,
                target_crs=_dummy_input,
                generate_primary_key_for=_dummy_input,
                source_composite_keys=[_dummy_input],
            )
            raise AssertionError("Passing broken datetime column")
        except ValueError:
            assert True
