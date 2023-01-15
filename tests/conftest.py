import pandas as pd
from pandas import Timestamp
import pytest

from gps_activity.models import CRSProjectionModel
from gps_activity.models import DataFramePivotFields
from gps_activity.models import DefaultValues


@pytest.fixture
def hk_crs() -> str:
    return "EPSG:2326"


@pytest.fixture
def wsg_84() -> str:
    return "EPSG:4326"


@pytest.fixture
def gps_sample_alba_scl_weee() -> pd.DataFrame:
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
            "unixtime": [
                1652174086.0,
                1652174096.0,
                1652174105.0,
                1652174115.0,
                1654594788.0,
                1654594798.0,
                1654594808.0,
                1654594818.0,
            ],
            "computed_velocity": [
                0.7037305147360688,
                0.7037305147360688,
                0.6414383713749385,
                0.9758428666008413,
                3.3362184996898088,
                3.3362184996898088,
                0.04537085990963587,
                0.05907602138465724,
            ],
        },
    )


@pytest.fixture
def projection_model(wsg_84: str, hk_crs: str) -> CRSProjectionModel:
    return CRSProjectionModel(source_crs=wsg_84, target_crs=hk_crs)


@pytest.fixture
def gps_pivot_fields() -> DataFramePivotFields:
    return DataFramePivotFields(
        source_lat="lat",
        source_lon="lon",
        source_datetime="datetime",
        source_vehicle_id="plate_no",
    )


@pytest.fixture
def default_values() -> DefaultValues:
    return DefaultValues()


@pytest.fixture
def round_tolerance() -> float:
    return 10 ** (-4)
