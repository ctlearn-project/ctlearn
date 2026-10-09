"""ctlearn command line tools.
"""

__all__ = [
    "TrainCTLearnModel",
    "TrainCTLearnKerasModel",
    "TrainCTLearnPyTorchModel",
    "LST1PredictionTool",
    "MonoPredictCTLearnModel",
    "StereoPredictCTLearnModel"
]

def __getattr__(name):
    if name == "TrainCTLearnModel":
        from .train_model import TrainCTLearnModel
        return TrainCTLearnModel
    if name == "TrainCTLearnKerasModel":
        from .keras.train_model import TrainCTLearnKerasModel
        return TrainCTLearnKerasModel
    if name == "TrainCTLearnPyTorchModel":
        from .pytorch.train_model import TrainCTLearnPyTorchModel
        return TrainCTLearnPyTorchModel
    if name == "LST1PredictionTool":
        from .predict_LST1 import LST1PredictionTool
        return LST1PredictionTool
    if name == "MonoPredictCTLearnModel":
        from .predict_mono_model import MonoPredictCTLearnModel
        return MonoPredictCTLearnModel
    if name == "StereoPredictCTLearnModel":
        from .predict_stereo_model import StereoPredictCTLearnModel
        return StereoPredictCTLearnModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")