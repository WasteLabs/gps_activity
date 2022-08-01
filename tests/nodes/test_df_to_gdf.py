import geopandas as gpd
import pandas as pd
import pytest

from gps_activity.abstract import AbstractNode
from gps_activity.models import CRSProjectionModel
from gps_activity.models import DataFramePivotFields
from gps_activity.nodes import ConverterDataFrame2GeoDataFrame


@pytest.fixture
def df_input(gps_pivot_fields: DataFramePivotFields):
    return pd.DataFrame(
        {
            gps_pivot_fields.projected_lon: [818721.6578177342],
            gps_pivot_fields.projected_lat: [834088.352552602],
        },
    )


@pytest.fixture
def df2gdf_converter(
    gps_pivot_fields: DataFramePivotFields,
    projection_model: CRSProjectionModel,
):
    return ConverterDataFrame2GeoDataFrame(
        pivot_fields=gps_pivot_fields,
        crs_projection=projection_model,
    )


class TestConverterDataFrame2GeoDataFrame:
    def test_instance(self, df2gdf_converter: ConverterDataFrame2GeoDataFrame):
        assert isinstance(df2gdf_converter, AbstractNode)

    def test_crs_projection_model(
        self,
        df2gdf_converter: ConverterDataFrame2GeoDataFrame,
        df_input: pd.DataFrame,
    ):
        gdf_output = df2gdf_converter.transform(df_input)
        assert "geometry" in gdf_output.columns
        assert isinstance(gdf_output, gpd.GeoDataFrame)
