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
    def test_init(self, gps: pd.DataFrame, plan: pd.DataFrame):
        instance = LinkerDataContainer(gps=gps, plan=plan)
        assert isinstance(instance, BaseModel)
