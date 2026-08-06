"""
Utility functions for the CTLearn tools.
"""

from enum import Enum
import pathlib
from importlib.resources import files, as_file
import os
import time
from tqdm import tqdm

from ctapipe.core import Provenance
from ctapipe.core.traits import TraitError
from ctapipe.instrument.optics import FocalLengthKind
from ctapipe.instrument import SubarrayDescription


__all__ = [
    "monitor_progress",
    "validate_trait_dict",
    "get_lst1_subarray_description",
    "FrameworkType",
    "detect_framework",
]

def monitor_progress(src_path, dst_path, stop_event, logger):
    try:
        total_size = os.path.getsize(src_path)
    except OSError:
        logger.error(f"Unable to access source file '{src_path}'.")
        return

    last_logged_percent = -1

    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Copy Progress") as pbar:
        while not stop_event.is_set():
            try:
                current_size = os.path.getsize(dst_path)
            except OSError:
                current_size = 0

            pbar.n = current_size
            pbar.refresh()
            
            # Logging cada 10%
            if total_size > 0:
                percent = int((current_size / total_size) * 100)
                if percent // 10 != last_logged_percent // 10:
                    logger.info(f"Progress: {percent}%")
                    last_logged_percent = percent

            time.sleep(0.5)
        # Ensure the progress bar reaches the end
        try:
            final_size = os.path.getsize(dst_path)
            pbar.n = final_size
            pbar.refresh()
            logger.info("Copy completed.")
        except OSError:
            logger.warning("Could not get final size of output file.")
            
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

class FrameworkType(Enum):
    """
    Deep learning framework type enumeration.
    
    This enumeration specifies which deep learning framework to use for
    model training and inference. CTLearn supports both Keras (TensorFlow backend)
    and PyTorch frameworks.
    
    Attributes:
        KERAS (int): Use Keras/TensorFlow framework (value: 1)
            - Advantages: High-level API, easy to use, good for prototyping
            - TensorFlow 2.x with Keras API
            - Suitable for production deployment
            
        PYTORCH (int): Use PyTorch framework (value: 2)
            - Advantages: Dynamic computation graphs, flexible, research-friendly
            - PyTorch 1.x or 2.x
            - Better for custom architectures and experimental models
    
    Example:
        >>> from ctlearn.core.ctlearn_enum import FrameworkType
        >>> framework = FrameworkType.PYTORCH
        >>> print(framework.name)  # 'PYTORCH'
        >>> print(framework.value)  # 'PyTorch'
    """
    KERAS = "Keras"
    PYTORCH = "PyTorch"

def detect_framework(path_val):
    """
    Determines framework based on file extension.
    Returns 'Keras', 'PyTorch', or raises TraitError.
    """
    if path_val is None:
        return None

    path = pathlib.Path(path_val)
    ext = path.suffix.lower()

    if ext in [".keras", ".h5"]:
        return FrameworkType["KERAS"]
    elif ext in [".pt", ".pth"]:
        return FrameworkType["PYTORCH"]
    else:
        raise TraitError(
            f"Invalid model extension '{ext}' for file '{path}'. "
            "Expected '.keras' or '.h5' for Keras, or '.pt' or '.pth' for PyTorch."
        )
