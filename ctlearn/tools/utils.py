"""
Utility functions for the CTLearn tools.
"""

from enum import Enum
import pathlib
from importlib.resources import files, as_file
import os
import time
from tqdm import tqdm

import tensorflow as tf
import torch

from ctapipe.core import Provenance
from ctapipe.core.tool import ToolConfigurationError
from ctapipe.core.traits import TraitError
from ctapipe.instrument.optics import FocalLengthKind
from ctapipe.instrument import SubarrayDescription


__all__ = [
    "monitor_progress",
    "validate_trait_dict",
    "get_lst1_subarray_description",
    "FrameworkType",
    "setup_framework",
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


def setup_framework(model_paths):
    """
    Detects the deep learning framework from model paths, ensures consistency, 
    and configures the hardware devices for distributed or single-device execution.

    This function iterates through the provided model paths, infers the framework 
    (Keras or PyTorch) based on file extensions, and ensures all models belong 
    to the same framework. It then initializes the appropriate hardware setup 
    (e.g., MirroredStrategy for Keras, CUDA/CPU device for PyTorch).

    Parameters
    ----------
    model_paths : list of str or pathlib.Path or None
        A list of paths pointing to the saved model files. `None` values are ignored.

    Returns
    -------
    tuple
        A tuple containing:
        - framework_type (FrameworkType): The identified framework enum (FrameworkType.KERAS or FrameworkType.PYTORCH).
        - num_devices (int): The number of devices available/configured for the framework.
        - strategy (tf.distribute.Strategy or None): The TensorFlow distribution strategy if Keras is detected, else None.
        - device (torch.device or None): The PyTorch target device if PyTorch is detected, else None.

    Raises
    ------
    ToolConfigurationError
        If no valid model paths are provided, or if multiple inconsistent frameworks 
        are detected across the provided paths.
    """

    def _detect_framework(path_val):
        """
        Determines framework based on file extension.
        Returns 'Keras', 'PyTorch', or raises TraitError.
        """
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

    # Detect frameworks from all non-None paths
    detected_frameworks = {}
    for path in model_paths:
        if path is not None:
            detected_frameworks[path] = _detect_framework(path)
    # Fail immediately if no valid model paths are provided
    if not detected_frameworks:
        raise ToolConfigurationError(
            "No model paths were specified. At least one valid model path "
            "(.keras/.h5 for Keras, or .pt/.pth for PyTorch) must be provided."
        )
    # Verify consistency across specified paths
    unique_frameworks = set(detected_frameworks.values())
    if len(unique_frameworks) > 1:
        details = ", ".join(f"{p}: {fw.value}" for p, fw in detected_frameworks.items())
        raise ToolConfigurationError(
            f"Inconsistent model frameworks detected across paths: {details}. "
            "All specified model files must belong to the same framework."
        )
    # Set framework directly from the detected FrameworkType enum
    framework_type = next(iter(unique_frameworks))
    # Configure framework-specific hardware/device setup
    strategy, device = None, None
    if framework_type == FrameworkType.KERAS:
        strategy = tf.distribute.MirroredStrategy()
        num_devices = strategy.num_replicas_in_sync
    elif framework_type == FrameworkType.PYTORCH:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            num_devices = torch.cuda.device_count()
        else:
            device = torch.device("cpu")
            num_devices = 1
    return framework_type, num_devices, strategy, device