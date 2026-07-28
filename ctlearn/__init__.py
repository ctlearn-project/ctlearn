import sys
import os
import warnings

# Suppress noisy logs globally before any other imports occur
is_debug = '--debug' in sys.argv or any(arg.startswith('--log-level=DEBUG') for arg in sys.argv)
if not is_debug:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['NCCL_DEBUG'] = 'WARN'
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", ".*NoneDefaultNotAllowedWarning.*")
    warnings.filterwarnings("ignore", ".*MergeConflictWarning.*")
    warnings.filterwarnings("ignore", ".*'ctlearn.tools.train_model' found in sys.modules.*")

from ._version import __version__
import importlib.util


def is_package_available(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None

__all__ = ["__version__", "is_package_available"]
