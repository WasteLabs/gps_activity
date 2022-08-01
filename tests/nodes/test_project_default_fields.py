import pandas as pd
import pytest

from gps_activity.abstract import AbstractNode
from gps_activity.models import DataFramePivotFields
from gps_activity.nodes import DefaultSchemaProjector


@pytest.fixture
def df_expected() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lng": [1, 2, 3],
            "latitude": [1, 2, 3],
            "vehicle_id": [1, 2, 3],
            "dtime": [1, 2, 3],
            "lon": [1, 2, 3],
            "lat": [1, 2, 3],
            "plate_no": [1, 2, 3],
            "datetime": [1, 2, 3],
        },
    )


@pytest.fixture
def df_input(df_expected: pd.DataFrame) -> pd.DataFrame:
    return df_expected.drop(columns=["lon", "lat", "plate_no", "datetime"])


@pytest.fixture
def original_pivot_fields() -> DataFramePivotFields:
    return DataFramePivotFields(
        source_lat="latitude",
        source_lon="lng",
        source_datetime="dtime",
        source_vehicle_id="vehicle_id",
    )


@pytest.fixture
def schema_projector(original_pivot_fields: DataFramePivotFields) -> DefaultSchemaProjector:
    return DefaultSchemaProjector(original_pivot_fields)


class TestDefaultSchemaProjector:
    def test_instance(self, schema_projector):
        assert isinstance(schema_projector, AbstractNode)

    def test_fit_transform_schema_project(
        self,
        schema_projector: DefaultSchemaProjector,
        df_expected: pd.DataFrame,
        df_input: pd.DataFrame,
    ):
        df_projected = schema_projector.transform(df_input)
        assert df_projected[list(df_expected.columns)].equals(df_expected)
