import runpy

import pytest
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
