import pytest
from torch.utils.data import DataLoader
from traitlets.config.loader import Config

from dl1_data_handler.reader import DLImageReader
from ctlearn.core.keras.sequence import KerasSequence
from ctlearn.core.pytorch.dataset import PyTorchDataset


@pytest.mark.parametrize(
    ("dataloader_cls", "expected_features_shape", "channel_axis"),
    [
        (KerasSequence, (1, 112, 112, 2), -1),
        (PyTorchDataset, (1, 2, 112, 112), 1),
    ],
    ids=["Keras", "PyTorch"],
)
@pytest.mark.parametrize(
    ("mode", "allowed_tels", "num_telescopes"),
    [
        ("mono", [7], 1),
        ("stereo", [7, 13, 15], 3),
    ],
    ids=["mono", "stereo"],
)
def test_data_loading(
    dl1_gamma_file,
    dataloader_cls,
    expected_features_shape,
    channel_axis,
    mode,
    allowed_tels,
    num_telescopes,
):
    """Check the data loading for the Keras and PyTorch frameworks"""
    expected_features_shape = list(expected_features_shape)
    expected_features_shape[channel_axis] *= num_telescopes
    # Create a configuration suitable for the test
    config = Config(
        {
            "DLImageReader": {
                "mode": mode,
                "allowed_tels": allowed_tels,
                "focal_length_choice": "EQUIVALENT",
            },
        }
    )
    # Create an image reader
    dl1_reader = DLImageReader(input_url_signal=[dl1_gamma_file], config=config)
    # Initialize the PyTorch Dataset or the Keras Sequence
    dl1_dataset = dataloader_cls(
        DLDataReader=dl1_reader,
        indices=[0],
        tasks=["type", "energy", "cameradirection", "skydirection"],
        stack_telescope_images=mode == "stereo",
    )
    # For PyTorch, wrap in DataLoader to apply batch dimension (dim=0)
    if issubclass(dataloader_cls, PyTorchDataset):
        batch_loader = DataLoader(dl1_dataset, batch_size=1)
        features, labels = next(iter(batch_loader))
        # Optional: convert PyTorch tensor to numpy if labels/features are Tensors
        if hasattr(features, "numpy"):
            features = features.numpy()
    else:
        features, labels = dl1_dataset[0]
    #  Check that all the correct labels are present
    assert (
        "type" in labels
        and "energy" in labels
        and "cameradirection" in labels
        and "skydirection" in labels
    )
    #  Check the shape of the features match the expected ones
    assert features.shape == tuple(expected_features_shape)
