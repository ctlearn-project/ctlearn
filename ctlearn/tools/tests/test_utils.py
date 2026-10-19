import pytest

from ctapipe.core.traits import TraitError
from ctlearn.tools.utils import (
    setup_framework,
    validate_trait_dict,
)


def test_validate_trait_dict_rejects_missing_keys():
    with pytest.raises(TraitError, match="missing required key"):
        validate_trait_dict({"hello": "world"}, {"required"})


@pytest.mark.parametrize(
    ("model_paths", "error", "message"),
    [
        ([], IOError, "No model paths"),
        (["model.invalid"], TraitError, "Invalid model extension"),
        (
            ["model.keras", "model.pt"],
            IOError,
            "Inconsistent model frameworks",
        ),
    ],
)
def test_setup_framework_rejects_invalid_model_paths(model_paths, error, message):
    with pytest.raises(error, match=message):
        setup_framework(model_paths)
