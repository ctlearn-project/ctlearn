"""
Predict from pixel-wise image data
"""

import pathlib
import atexit

import numpy as np
from astropy import units as u
import shutil
import tables
import tensorflow as tf
import keras
from astropy.coordinates.earth import EarthLocation
from astropy.coordinates import SkyCoord

from ctapipe.core import Tool
from ctapipe.core.tool import ToolConfigurationError
from ctapipe.core.traits import (
    Bool,
    Int,
    Path,
    flag,
    Set,
    Dict,
    List,
    CaselessStrEnum,
    ComponentName,
    Unicode,
    classes_with_traits,
)
from astropy.table import (
    Table,
    hstack,
    vstack,
    join,
)
from ctapipe.io import read_table, write_table, HDF5Merger
from ctlearn.core.model import LoadedModel
from dl1_data_handler.reader import DLDataReader, ProcessType, get_unmapped_image, get_unmapped_waveform
from dl1_data_handler.loader import DLDataLoader


class MonoPredictionTool(Tool):
    """
    Perform statistics calculation for pixel-wise image data
    """

    name = "MonoPredictionTool"
    description = "Perform statistics calculation for pixel-wise image data"

    examples = """
    To calculate statistics of pixel-wise image data files:

    > ctapipe-calculate-pixel-statistics --input_url input.dl1.h5 --overwrite

    """

    input_url = Path(
        help="Input LST-1 HDF5 files including pixel-wise image data",
        allow_none=True,
        exists=True,
        directory_ok=False,
        file_ok=True,
    ).tag(config=True)

    tel_id = Int(
        default_value=1,
        allow_none=False,
        help="Telescope ID to process.",
    ).tag(config=True)

    dl1dh_reader_type = ComponentName(DLDataReader, default_value="DLImageReader").tag(
        config=True
    )

    reco_algo = Unicode(
        default_value="CTLearn",
        allow_none=False,
        help="Name of the reconstruction algorithm used to generate the dl2 data.",
    ).tag(config=True)

    reco_tasks = List(
        trait=CaselessStrEnum(["type", "energy", "direction"]),
        allow_none=False,
        help=(
            "List of reconstruction tasks to perform. "
            "'type': classification of the primary particle type "
            "'energy': regression of the primary particle energy "
            "'direction': regression of the primary particle arrival direction "
        ),
    ).tag(config=True)

    batch_size = Int(
        default_value=64,
        allow_none=False,
        help="Size of the batch to perform inference of the neural network.",
    ).tag(config=True)

    output_path = Path(
        default_value="./dl2_prediction.h5",
        allow_none=False,
        help="Output path to save the dl2 prediction results",
    ).tag(config=True)

    aliases = {
        ("i", "input_url"): "MonoPredictionTool.input_url",
        "reco": "MonoPredictionTool.reco_tasks",
        ("o", "output"): "MonoPredictionTool.output_path",
    }

    flags = {
        **flag(
            "dl1-parameters",
            "HDF5Merger.dl1_parameters",
            "Include dl1 parameters",
            "Exclude dl1 parameters",
        ),
        **flag(
            "dl1-images",
            "HDF5Merger.dl1_images",
            "Include dl1 images",
            "Exclude dl1 images",
        ),
        **flag(
            "true-parameters",
            "HDF5Merger.true_parameters",
            "Include true parameters",
            "Exclude true parameters",
        ),
        **flag(
            "true-images",
            "HDF5Merger.true_images",
            "Include true images",
            "Exclude true images",
        ),
    }

    classes = classes_with_traits(DLDataReader) + classes_with_traits(LoadedModel)

    def setup(self):

        if not self.output_path.exists():
            self.log.info("Copying to output destination.")
            with HDF5Merger(self.output_path, parent=self) as merger:
                merger(self.input_url)


        # Set up the data reader
        self.log.info("Loading data reader:")
        self.dl1dh_reader = DLDataReader.from_name(
            self.dl1dh_reader_type,
            input_url_signal=[self.input_url],
            parent=self,
        )

        self.table_name = f"/dl1/event/telescope/images/tel_{self.tel_id:03d}"

        # Get the number of rows in the table
        with tables.open_file(self.input_url) as input_file:
            self.table_length = len(input_file.get_node(self.table_name))

        # Load the model from the specified path
        self.log.info("Loading the model.")
        image_shape = self.dl1dh_reader.image_mappers[
            self.dl1dh_reader.cam_name
        ].image_shape
        if self.dl1dh_reader_type == "DLImageReader":
            channel = len(self.dl1dh_reader.img_channels)
        elif self.dl1dh_reader_type == "DLWaveformReader":
            channel = self.dl1dh_reader.sequnce_length
        input_shape = (image_shape, image_shape, channel)
        self.model = LoadedModel(
            input_shape=input_shape, tasks=self.reco_tasks, parent=self
        ).model

    def start(self):
        self.log.info("Starting the prediction...")

        data, predict_data = [], []
        # Iterate over the data in chunks based on the batch size
        for start in range(0, self.table_length, self.batch_size):
            stop = min(start + self.batch_size, self.table_length)
            self.log.debug(f"Processing chunk from {start} to {stop - 1}")
            # Read the data
            dl1_table = read_table(self.input_url, self.table_name, start=start, stop=stop)
            image_data = []
            for event in dl1_table:
                # Get the unmapped image
                image = get_unmapped_image(
                    event, self.dl1dh_reader.img_channels, self.dl1dh_reader.transforms
                )
                image_data.append(self.dl1dh_reader.image_mappers[self.dl1dh_reader.cam_name].map_image(image))
            input_data = {"input": np.array(image_data)}
            # Temp fix for supporting keras2 & keras3
            if int(keras.__version__.split(".")[0]) >= 3:
                input_data = input_data["input"]
            # Predict the data using the loaded model
            predict_data.append(Table(self.model.predict_on_batch(input_data)))
            dl1_table.keep_columns(["obs_id", "event_id", "tel_id", "is_valid"])
            data.append(dl1_table)

        self.predict_data = vstack(predict_data)
        self.data = vstack(data)

    def finish(self):

        # Fill the prediction table with the prediction results based on the different tasks
        if "type" in self.reco_tasks:
            self.data.add_column(self.predict_data["col1"], name="prediction")
            # Save the prediction to the output file
            write_table(
                self.data,
                self.output_path,
                f"/dl2/event/telescope/classification/{self.reco_algo}/tel_{self.tel_id:03d}",
            )
        if "direction" in self.reco_tasks:
            if self.dl1dh_reader.process_type == ProcessType.Simulation:
                tel_pointing = read_table(
                    self.input_url,
                    f"/configuration/telescope/pointing/tel_{self.tel_id:03d}",
                )
                self.data = join(
                    left=self.data,
                    right=tel_pointing,
                    keys=["obs_id", "tel_id"],
                )
            # Save the prediction to the output file
            reco_spherical_offset_az = u.Quantity(self.predict_data["direction"].T[0], unit=u.deg)
            reco_spherical_offset_alt = u.Quantity(self.predict_data["direction"].T[1], unit=u.deg)

            # Set the telescope pointing of the SkyOffsetSeparation tranform to the fix pointing
            pointing = SkyCoord(
                self.data["telescope_pointing_azimuth"],
                self.data["telescope_pointing_altitude"],
                frame="altaz",
            )
            reco_direction = pointing.spherical_offsets_by(reco_spherical_offset_az, reco_spherical_offset_alt)
            self.data = hstack([self.data, reco_direction.to_table()], join_type="exact")
            self.data.remove_columns(
                [
                    "telescope_pointing_azimuth",
                    "telescope_pointing_altitude",
                ]
            )
            self.data.sort(["obs_id", "event_id"])
            write_table(
                self.data,
                self.output_path,
                f"/dl2/event/telescope/geometry/{self.reco_algo}/tel_{self.tel_id:03d}",
            )
        if "energy" in self.reco_tasks:
            reco_energy = u.Quantity(
                np.power(10, np.squeeze(self.predict_data["energy"])), unit=u.TeV
            )
            self.data.add_column(reco_energy, name="energy")
            # Save the prediction to the output file
            write_table(
                self.data,
                self.output_path,
                f"/dl2/event/telescope/energy/{self.reco_algo}/tel_{self.tel_id:03d}",
            )

        self.log.info("Tool is shutting down")


def main():
    # Run the tool
    tool = MonoPredictionTool()
    tool.run()


if __name__ == "main":
    main()
