"""ctlearn command line tools.
"""

import sys
import os
import warnings

is_debug = '--debug' in sys.argv or any(arg.startswith('--log-level=DEBUG') for arg in sys.argv)
if not is_debug:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['NCCL_DEBUG'] = 'WARN'
    warnings.filterwarnings("ignore", ".*NoneDefaultNotAllowedWarning.*")
    warnings.filterwarnings("ignore", ".*MergeConflictWarning.*")
    warnings.filterwarnings("ignore", ".*'ctlearn.tools.train_model' found in sys.modules.*")

from .train_model import DLFrameWork
try:
    from .predict_LST1 import LST1PredictionTool
except ImportError:
    pass
try:
    from .predict_model import MonoPredictCTLearnModel, StereoPredictCTLearnModel
except ImportError:
    pass

__all__ = [
    "DLFrameWork",
]
try:
    __all__.append("MonoPredictCTLearnModel")
    __all__.append("StereoPredictCTLearnModel")
except NameError:
    pass
try:
    __all__.append("LST1PredictionTool")
except NameError:
    pass