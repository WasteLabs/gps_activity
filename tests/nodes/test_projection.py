from typing import List

import numpy as np
import pandas as pd
import pytest

from gps_activity.abstract import AbstractNode
from gps_activity.models import CRSProjectionModel
from gps_activity.models import DataFramePivotFields
from gps_activity.nodes import CRSTransformer


@pytest.fixture
def expected_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lat": [22.446138, 22.44614],
            "lon": [114.006416, 114.00642],
            "y": [834088.352552602, 834088.5735585009],
            "x": [818721.6578177342, 818722.0698150198],
        },
    )


@pytest.fixture
def columns_order(expected_dataframe: pd.DataFrame) -> pd.DataFrame:
    return list(expected_dataframe.columns)


@pytest.fixture
def input_dataframe(expected_dataframe: pd.DataFrame) -> pd.DataFrame:
    return expected_dataframe.drop(columns=["x", "y"])


@pytest.fixture
def crs_transformer(
    projection_model: CRSProjectionModel,
    gps_pivot_fields: DataFramePivotFields,
):
    return CRSTransformer(
        crs_projection=projection_model,
        pivot_fields=gps_pivot_fields,
    )


class TestCRSProjector:
    def test_transformer_instance(self, crs_transformer: CRSTransformer):
        assert isinstance(crs_transformer, AbstractNode)

    def test_fit_transform_projection(
        self,
        input_dataframe: pd.DataFrame,
        expected_dataframe: pd.DataFrame,
        crs_transformer: CRSTransformer,
        columns_order: List[str],
        round_tolerance: float,
    ):
        computed_dataframe = crs_transformer.fit_transform(input_dataframe.copy())
        computed_dataframe = computed_dataframe[columns_order]
        expected_dataframe = expected_dataframe[columns_order]
        assert np.allclose(
            computed_dataframe.values,
            expected_dataframe.values,
            atol=round_tolerance,
            rtol=round_tolerance,
        )
