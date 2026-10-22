import importlib
import runpy

import pytest
import ctlearn.tools as tools
from ctapipe.core import Tool


@pytest.mark.parametrize(
    ("module_name", "tool_name"),
    [
        ("ctlearn.tools.predict_mono_model", "MonoPredictCTLearnModel"),
        ("ctlearn.tools.predict_stereo_model", "StereoPredictCTLearnModel"),
        ("ctlearn.tools.predict_LST1", "LST1PredictionTool"),
        ("ctlearn.tools.keras.train_model", "TrainCTLearnKerasModel"),
        ("ctlearn.tools.pytorch.train_model", "TrainCTLearnPyTorchModel"),
    ],
)
def test_module_main_guard_runs_tool(monkeypatch, module_name, tool_name):
    calls = []

    def record_run(tool, *args, **kwargs):
        calls.append(tool)

    monkeypatch.setattr(Tool, "run", record_run)
    runpy.run_module(module_name, run_name="__main__")

    assert len(calls) == 1
    assert type(calls[0]).__name__ == tool_name


@pytest.mark.parametrize(
    ("name", "module_name"),
    [
        ("TrainCTLearnModel", "ctlearn.tools.train_model"),
        ("TrainCTLearnKerasModel", "ctlearn.tools.keras.train_model"),
        ("TrainCTLearnPyTorchModel", "ctlearn.tools.pytorch.train_model"),
        ("LoggingModelCheckpoint", "ctlearn.tools.keras.train_model"),
        ("LST1PredictionTool", "ctlearn.tools.predict_LST1"),
        ("MonoPredictCTLearnModel", "ctlearn.tools.predict_mono_model"),
        ("StereoPredictCTLearnModel", "ctlearn.tools.predict_stereo_model"),
    ],
)
def test_lazy_tool_exports(name, module_name):
    implementation = importlib.import_module(module_name)

    assert getattr(tools, name) is getattr(implementation, name)


def test_unknown_tool_export_raises_attribute_error():
    with pytest.raises(AttributeError, match="unknown_tool"):
        tools.unknown_tool
