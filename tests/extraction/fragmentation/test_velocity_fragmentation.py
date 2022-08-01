from typing import Union

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from gps_activity.extraction.factory.fragmentation import VelocityFragmentationFactory
from gps_activity.models import DataFramePivotFields


@pytest.fixture
def expected_results(gps_pivot_fields: DataFramePivotFields) -> pd.DataFrame:
    return pd.DataFrame(
        {
            gps_pivot_fields.computed_velocity: [1, 2, 3, 4, 5, 6],
            gps_pivot_fields.fragmentation_output: [True, True, True, True, False, False],
        },
    )


@pytest.fixture
def input_data(
    expected_results: pd.DataFrame,
    gps_pivot_fields: DataFramePivotFields,
) -> pd.DataFrame:
    return expected_results.drop(columns=[gps_pivot_fields.fragmentation_output])


@pytest.fixture
def max_velocity_hard_limit() -> Union[float, int]:
    return 4


@pytest.fixture
def fragmentation_pipeline(
    max_velocity_hard_limit: Union[float, int],
):
    return VelocityFragmentationFactory.factory_pipeline(
        max_velocity_hard_limit=max_velocity_hard_limit,
    )


class TestVelocityFragmentationPipelineFactory:
    def test_instance(self, fragmentation_pipeline: Pipeline):
        assert isinstance(fragmentation_pipeline, Pipeline)

    def test_pipeline_transform(
        self,
        input_data: pd.DataFrame,
        expected_results: pd.DataFrame,
        fragmentation_pipeline: Pipeline,
        gps_pivot_fields: DataFramePivotFields,
    ):
        y_pred = fragmentation_pipeline.predict(input_data)
        input_data[gps_pivot_fields.fragmentation_output] = y_pred
        assert input_data.equals(expected_results)
