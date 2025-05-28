from importlib.resources import files, as_file

from ctapipe.core import Provenance
from ctapipe.core.traits import TraitError
from ctapipe.instrument import SubarrayDescription
from ctapipe.instrument.optics import FocalLengthKind

import os
import time
from tqdm import tqdm

__all__ = ["get_lst1_subarray_description", "validate_trait_dict", "monitor_progress"]

def monitor_progress(src_path, dst_path, stop_event, logger):
    try:
        total_size = os.path.getsize(src_path)
    except OSError:
        logger.error(f"Unable to access source file '{src_path}'.")
        return

    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Copy Progress") as pbar:
        while not stop_event.is_set():
            try:
                current_size = os.path.getsize(dst_path)
            except OSError:
                current_size = 0
            pbar.n = current_size
            pbar.refresh()
            time.sleep(0.5)
        # Ensure the progress bar reaches the end
        try:
            final_size = os.path.getsize(dst_path)
            pbar.n = final_size
            pbar.refresh()
        except OSError:
            pass



def get_lst1_subarray_description(focal_length_choice=FocalLengthKind.EFFECTIVE):
    """
    Load subarray description from bundled file
    
    Parameters
    ----------
    focal_length_choice : FocalLengthKind
        Choice of focal length to use.  Options are ``FocalLengthKind.EQUIVALENT``
        and ``FocalLengthKind.EFFECTIVE``. Default is ``FocalLengthKind.EFFECTIVE``.

    Returns
    -------
    SubarrayDescription
        Subarray description of the LST-1 telescope.
    """
    with as_file(files("ctlearn") / "resources/LST-1_SubarrayDescription.h5") as path:
        Provenance().add_input_file(path, role="SubarrayDescription")
        return SubarrayDescription.from_hdf(path, focal_length_choice=focal_length_choice)
    
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
