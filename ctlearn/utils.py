from importlib.resources import files, as_file

from ctapipe.core import Provenance
from ctapipe.core.traits import TraitError
from ctapipe.instrument import SubarrayDescription
from ctapipe.instrument.optics import FocalLengthKind


__all__ = [
    "get_lst1_subarray_description",
    "validate_trait_dict",
    "validate_conv_backend",
    "model_conv_backend",
]

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

def validate_conv_backend(image_mappers, conv_backend):
    """
    Validate that the image mapper(s) used to build the input and the
    model's convolution backend agree with each other.

    ``HexagdlyMapper`` output is hex-addressed and must be paired with a
    model built with ``conv_backend="hexagdly"``; a square-mapped input
    (e.g. from ``BilinearMapper``) must be paired with
    ``conv_backend="square"``. Nothing else currently enforces this pairing,
    so a mismatch would otherwise silently train/predict on geometrically
    meaningless input.

    Parameters
    ----------
    image_mappers : dict
        Dictionary of ``dl1_data_handler.image_mapper.ImageMapper`` instances,
        as built by ``DLDataReader.image_mappers``.
    conv_backend : str
        The model's ``conv_backend`` trait value (``"square"`` or
        ``"hexagdly"``).

    Returns
    -------
    bool
        True if the mapper(s) and conv backend agree. Otherwise, raises a
        ValueError.
    """
    mapper_is_hex = any(
        type(mapper).__name__ == "HexagdlyMapper" for mapper in image_mappers.values()
    )
    model_is_hex = conv_backend == "hexagdly"
    if mapper_is_hex != model_is_hex:
        raise ValueError(
            "Mismatch between image_mapper_type and the model's conv_backend: "
            f"HexagdlyMapper in use = {mapper_is_hex}, conv_backend = {conv_backend!r}. "
            "HexagdlyMapper output must be paired with a model built with "
            "conv_backend='hexagdly', and a square-mapped input (e.g. from "
            "BilinearMapper) must be paired with conv_backend='square'."
        )
    return True

def model_conv_backend(model):
    """
    Determine the convolution backend a (possibly loaded-from-disk) Keras
    model actually uses, by inspecting its layers directly.

    Used at prediction time, where models are loaded via
    ``keras.saving.load_model`` rather than built through
    ``CTLearnModel.from_name``, so no ``conv_backend`` trait is available to
    read -- the loaded layers are the only ground truth for what the model
    was actually built with.

    Parameters
    ----------
    model : keras.Model
        A (possibly nested) Keras model to inspect.

    Returns
    -------
    str
        ``"hexagdly"`` if any layer in the model (recursing into nested
        sub-models) is a ``keras_hexagdly`` layer, otherwise ``"square"``.
    """
    import keras_hexagdly as hgly

    def _uses_hexagdly(layer):
        if isinstance(layer, (hgly.Conv2d, hgly.MaxPool2d)):
            return True
        sublayers = getattr(layer, "layers", None)
        return sublayers is not None and any(_uses_hexagdly(sub) for sub in sublayers)

    return "hexagdly" if _uses_hexagdly(model) else "square"
