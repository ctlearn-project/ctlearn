from ._version import __version__
import importlib.util


def is_package_available(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None

__all__ = ["__version__", "is_package_available"]
