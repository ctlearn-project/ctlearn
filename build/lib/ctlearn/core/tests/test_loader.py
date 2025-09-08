import pytest
from traitlets.config.loader import Config

from dl1_data_handler.reader import DLImageReader
from ctlearn.core.loader import DLDataLoader


def test_data_loader(dl1_tmp_path, dl1_gamma_file):
    """check"""
    # Create a configuration suitable for the test
    config = Config(
        {
            "DLImageReader": {
                "allowed_tels": [4],
            },
        }
    )
    # Create an image reader
    dl1_reader = DLImageReader(input_url_signal=[dl1_gamma_file], config=config)
    # Create a data loader
    dl1_loader = DLDataLoader(
        DLDataReader=dl1_reader,
        indices=[0],
        tasks=["type", "energy", "cameradirection", "skydirection"],
        batch_size=1,
    )
    # Get the features and labels fgrom the data loader for one batch
    features, labels = dl1_loader[0]
    #  Check that all the correct labels are present
    assert (
        "type" in labels
        and "energy" in labels
        and "cameradirection" in labels
        and "skydirection" in labels
    )
    #  Check the shape of the features
    assert features["input"].shape == (1, 110, 110, 2)
