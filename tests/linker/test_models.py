import pandas as pd
from pydantic import BaseModel
import pytest

# from gps_activity.linker.models import ActivityLinkageLinkerDataContainer
from gps_activity.models import LinkerDataContainer


@pytest.fixture
def gps():
    return pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]})


@pytest.fixture
def plan():
    return pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]})


class TestLinkerDataContainer:
    def test_correct_init(self, gps: pd.DataFrame, plan: pd.DataFrame):
        instance = LinkerDataContainer.factory_instance({"gps": gps, "plan": plan})
        assert isinstance(instance, BaseModel)

    def test_incorrect_init(self, plan: pd.DataFrame):
        try:
            LinkerDataContainer.factory_instance({"dummy_input": plan})
            raise AssertionError("Incorrect data")
        except KeyError:
            assert True
