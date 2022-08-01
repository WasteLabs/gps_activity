import pandas as pd
import pytest

from gps_activity.abstract import AbstractNode
from gps_activity.models import DataFramePivotFields
from gps_activity.nodes import DateExtractor


@pytest.fixture
def df_expected():
    return pd.DataFrame(
        {
            "datetime": ["2022-05-10 00:25:32", "2022-05-10 00:26:32"],
            "date": ["2022-05-10", "2022-05-10"],
        },
    )


@pytest.fixture
def df_input(df_expected: pd.DataFrame):
    return df_expected.drop(columns=["date"])


@pytest.fixture
def date_extractor():
    return DateExtractor(
        pivot_fields=DataFramePivotFields(source_datetime="datetime"),
    )


class TestDateExtractor:
    def test_instance(self, date_extractor: DateExtractor):
        assert isinstance(date_extractor, AbstractNode)

    def test_fit_transform_date_extraction(
        self,
        date_extractor: DateExtractor,
        df_input: pd.DataFrame,
        df_expected: pd.DataFrame,
    ):
        df_output = date_extractor.fit_transform(df_input)
        df_output.astype(str).equals(df_expected.astype(str))
