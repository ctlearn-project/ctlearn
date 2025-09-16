import pytest
from traitlets.config.loader import Config

from dl1_data_handler.reader import DLImageReader
from ctlearn.core.data_loader.loader import DLDataLoader

from ctlearn.tools.train.pytorch.utils import read_configuration
from ctlearn.tools.train.pytorch.utils import get_absolute_config_path


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
    config_file_dir = get_absolute_config_path()
    print(config_file_dir)
    parameters = read_configuration(config_file_dir)

    dl1_loader = DLDataLoader.create(
        framework = "pytorch",
        DLDataReader=dl1_reader,
        indices=[0],
        tasks=["type", "energy", "cameradirection", "skydirection"],
        batch_size=1,
        parameters=parameters,
        use_augmentation=parameters["augmentation"]["use_augmentation"],
        is_training=True
    )
    # Get the features and labels fgrom the data loader for one batch
    print(len(dl1_loader[0]))
    print(dl1_loader[0][0])
    print(dl1_loader[0][1])
    print(dl1_loader[0][2])


    features, labels, _ = dl1_loader[0]
    #  Check that all the correct labels are present
    assert (
        "type" in labels
        and "energy" in labels
        and "cameradirection" in labels
        and "skydirection" in labels
    )
    #  Check the shape of the features
    assert features["image"].shape == (0, 1, 110, 110)

if __name__ == "__main__":
    test_data_loader()