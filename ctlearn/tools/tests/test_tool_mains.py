from importlib import import_module
from types import SimpleNamespace

import pytest


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
def test_main_runs_tool(monkeypatch, module_name, tool_name):
    """ Test that the main function of a tool runs the tool's run method once. """
    module = import_module(module_name)
    calls = []
    monkeypatch.setattr(
        module, tool_name, lambda: SimpleNamespace(run=lambda: calls.append(True))
    )

    module.main()

    assert calls == [True]
