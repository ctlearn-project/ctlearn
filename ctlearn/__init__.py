import os
import sys
if os.environ.get("TF_USE_LEGACY_KERAS") == "1":
    try:
        import tf_keras
        sys.modules["keras"] = tf_keras
    except ImportError:
        pass

from ._version import __version__

__all__ = ["__version__"]