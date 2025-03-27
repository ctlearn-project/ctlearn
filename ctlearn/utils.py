from importlib.resources import files, as_file

from ctapipe.core import Provenance
from ctapipe.core.traits import TraitError
from ctapipe.instrument import SubarrayDescription

__all__ = ["get_LST1_SubarrayDescription", "validate_trait_dict"]

def get_LST1_SubarrayDescription():
    """Load subarray description from bundled file"""
    with as_file(files("ctlearn") / "resources/LST-1_SubarrayDescription.h5") as path:
        Provenance().add_input_file(path, role="SubarrayDescription")
        return SubarrayDescription.from_hdf(path)

def validate_trait_dict(dict, required_keys):
    """
    Validate that a dictionary contains all required keys.

    Parameters
    ----------
    dict : dict
        Dictionary to validate.
    required_keys : set
        Set of required keys.

    Returns
    -------
    bool
        True if the dictionary contains all required keys.  Otherwise, raises a TraitError.
    """
    missing_keys = required_keys - dict.keys()
    if missing_keys:
        raise TraitError(f"Dict is missing required key(s): {', '.join(missing_keys)}")
    return True
