"""
Predict from pixel-wise image data
"""

import pathlib

import numpy as np
from astropy import units as u
from astropy.table import Table
import tables
import keras
from astropy.coordinates.earth import EarthLocation
from astropy.time import Time
from traitlets.config.loader import Config

from ctapipe.core import Tool
from ctapipe.core.tool import ToolConfigurationError
from ctapipe.core.traits import (
    Bool,
    Int,
    Path,
    Set,
    Dict,
    List,
    CaselessStrEnum,
    Unicode,
    classes_with_traits,
)
from ctapipe.instrument import SubarrayDescription, CameraGeometry
from ctapipe.io import read_table, write_table
from ctlearn.core.model import LoadedModel
from dl1_data_handler.image_mapper import ImageMapper, BilinearMapper
from dl1_data_handler.reader import get_unmapped_image, get_unmapped_waveform

class LST1PredictionTool(Tool):
    """
    Perform statistics calculation for pixel-wise image data
    """

    name = "StatisticsCalculatorTool"
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
    
    reco_algo = Unicode(
        default_value="CTLearn",
        allow_none=False,
        help="Name of the reconstruction algorithm used to generate the dl2 data.",
    ).tag(config=True)

    load_type_model_from = Path(
        default_value=None,
        help=(
            "Path to a Keras model file (Keras3) or directory (Keras2) "
            "for the classification of the primary particle type."
        ),
        allow_none=True,
        exists=True,
        directory_ok=True,
        file_ok=True,
    ).tag(config=True)

    load_energy_model_from = Path(
        default_value=None,
        help=(
            "Path to a Keras model file (Keras3) or directory (Keras2) "
            "for the regression of the primary particle energy."
        ),
        allow_none=True,
        exists=True,
        directory_ok=True,
        file_ok=True,
    ).tag(config=True)

    load_direction_model_from = Path(
        default_value=None,
        help=(
            "Path to a Keras model file (Keras3) or directory (Keras2) "
            "for the regression of the primary particle arrival direction."
        ),
        allow_none=True,
        exists=True,
        directory_ok=True,
        file_ok=True,
    ).tag(config=True)

    batch_size = Int(
        default_value=64,
        allow_none=False,
        help="Size of the batch to perform inference of the neural network.",
    ).tag(config=True)

    channels = List(
        trait=CaselessStrEnum(
            [
                "image",
                "cleaned_image",
                "peak_time",
                "relative_peak_time",
                "cleaned_peak_time",
                "cleaned_relative_peak_time"
            ]
        ),
        default_value=["image", "peak_time"],
        allow_none=False,
        help=(
            "Set the input channels to be loaded from the DL1 event data. "
            "image: integrated charges, "
            "cleaned_image: integrated charges cleaned with the DL1 cleaning mask, "
            "peak_time: extracted peak arrival times, "
            "relative_peak_time: extracted relative peak arrival times, "
            "cleaned_peak_time: extracted peak arrival times cleaned with the DL1 cleaning mask,"
            "cleaned_relative_peak_time: extracted relative peak arrival times cleaned with the DL1 cleaning mask."
        ),
    ).tag(config=True)

    store_event_wise_pointing = Bool(
        default_value=True,
        allow_none=False,
        help="Store the event-wise telescope pointing in the output file.",
    ).tag(config=True)

    output_path = Path(
        default_value="./output.dl2.h5",
        allow_none=False,
        help="Output path to save the dl2 prediction results",
    ).tag(config=True)

    overwrite = Bool(help="Overwrite output file if it exists").tag(config=True)

    aliases = {
        ("i", "input_url"): "LST1PredictionTool.input_url",
        ("t", "type_model"): "LST1PredictionTool.load_type_model_from",
        ("e", "energy_model"): "LST1PredictionTool.load_energy_model_from",
        ("d", "direction_model"): "LST1PredictionTool.load_direction_model_from",
        ("o", "output"): "LST1PredictionTool.output_path",
    }

    flags = {
        "overwrite": (
            {"LST1PredictionTool": {"overwrite": True}},
            "Overwrite existing files",
        ),
    }

    classes = classes_with_traits(BilinearMapper)

    def setup(self):

        # Save dl1 image and parameters tree schemas and tel id for easy access
        self.image_table_name = "/dl1/event/telescope/image/LST_LSTCam"
        self.parameter_table_name = "/dl1/event/telescope/parameters/LST_LSTCam"
        self.tel_id = 1

        # Get the number of rows in the table
        with tables.open_file(self.input_url) as input_file:
            self.table_length = len(input_file.get_node(self.image_table_name))

        # Load the models from the specified paths
        if self.load_type_model_from is not None:
            self.log.info("Loading the type model from %s.", self.load_type_model_from)
            self.keras_model_type = keras.saving.load_model(self.load_type_model_from)
            self.input_shape = self.keras_model_type.input_shape[1:]
        if self.load_energy_model_from is not None:
            self.log.info("Loading the energy model from %s.", self.load_energy_model_from)
            self.keras_model_energy = keras.saving.load_model(self.load_energy_model_from)
            self.input_shape = self.keras_model_type.input_shape[1:]
        if self.load_direction_model_from is not None:
            self.log.info("Loading the direction model from %s.", self.load_direction_model_from)
            self.keras_model_direction = keras.saving.load_model(self.load_direction_model_from)
            self.input_shape = self.keras_model_type.input_shape[1:]

        # Create the image mappers
        self.epoch = Time('1970-01-01T00:00:00', scale='utc')
        pos = {1 : [50.0, 50.0, 16.0] * u.m}
        tel = {1 : "LST_LST_LSTCam"}
        LOCATION = EarthLocation(lon=-17 * u.deg, lat=28 * u.deg, height=2200 * u.m)
        self.subarray = SubarrayDescription(
            "LST-1 of CTAO-North",
            tel_positions=pos,
            tel_descriptions=tel,
            reference_location=LOCATION,
        )
        self.image_mappers = {}
        cam_geom = {}
        self.camera_name = "LSTCam"
        with tables.open_file(self.input_url) as input_file:
            cam_geom_table = input_file.root.configuration.instrument.telescope.camera._f_get_child("geometry_0")
            cam_geom[self.camera_name] = CameraGeometry(
                name=self.camera_name,
                pix_id=cam_geom_table.cols.pix_id[:],
                pix_type="hexagon",
                pix_x=u.Quantity(cam_geom_table.cols.pix_x[:], u.cm),
                pix_y=u.Quantity(cam_geom_table.cols.pix_y[:], u.cm),
                pix_area=u.Quantity(cam_geom_table.cols.pix_area[:], u.cm**2),
                pix_rotation="100.893deg",
                cam_rotation="0deg",
            )
        self.image_mappers[self.camera_name] = BilinearMapper(
            geometry=cam_geom[self.camera_name], subarray=self.subarray, parent=self
        )

        if self.input_shape[0] != self.image_mappers[self.camera_name].image_shape[0]:
            raise ToolConfigurationError(
                f"The input shape of the model ('{self.input_shape[0]}') does not match "
                f"the image shape of the ImageMapper ('{self.image_mappers[self.camera_name].image_shape[0]'). "
                f"Use '--BilinearMapper.interpolation_image_shape={self.input_shape[0]}' ."
            )

        # Get offset and scaling of images
        self.transforms = {}
        self.transforms["image_scale"] = 0.0
        self.transforms["image_offset"] = 0
        self.transforms["peak_time_scale"] = 0.0
        self.transforms["peak_time_offset"] = 0
        # Get the number of rows in the table
        with tables.open_file(self.input_url) as input_file:
            img_table_v_attrs = input_file.get_node(self.image_table_name)._v_attrs
            print(img_table_v_attrs)
            
        # Check the transform value used for the file compression
        if "CTAFIELD_3_TRANSFORM_SCALE" in img_table_v_attrs:
            self.transforms["image_scale"] = img_table_v_attrs[
                "CTAFIELD_3_TRANSFORM_SCALE"
            ]
            self.transforms["image_offset"] = img_table_v_attrs[
                "CTAFIELD_3_TRANSFORM_OFFSET"
            ]
        if "CTAFIELD_4_TRANSFORM_SCALE" in img_table_v_attrs:
            self.transforms["peak_time_scale"] = img_table_v_attrs[
                "CTAFIELD_4_TRANSFORM_SCALE"
            ]
            self.transforms["peak_time_offset"] = img_table_v_attrs[
                "CTAFIELD_4_TRANSFORM_OFFSET"
            ]


    def start(self):
        self.log.info("Starting the prediction...")

        output_identifiers = read_table(self.input_url, self.parameter_table_name)
        tel_az = u.Quantity(output_identifiers["az_tel"], unit=u.rad)
        tel_alt = u.Quantity(output_identifiers["alt_tel"], unit=u.rad)
        event_type = output_identifiers["event_type"]
        time = (Time(output_identifiers["dragon_time"] * u.s, format='unix') - self.epoch).to('mjd')
        print(time)
        output_identifiers.keep_columns(["obs_id", "event_id", "tel_id"])
        output_identifiers.sort(["obs_id", "event_id", "tel_id"])
        prediction, energy, az, alt = [], [], [], []
        print(self.table_length)
        # Iterate over the data in chunks based on the batch size
        for start in range(0, self.table_length, self.batch_size):
            stop = min(start + self.batch_size, self.table_length)
            self.log.debug("Processing chunk from '%d' to '%d'.", start, stop - 1)
            # Read the data
            dl1_table = read_table(self.input_url, self.image_table_name, start=start, stop=stop)
            data = []
            for event in dl1_table:
                # Get the unmapped image
                image = get_unmapped_image(
                    event, self.channels, self.transforms
                )
                data.append(self.image_mappers[self.camera_name].map_image(image))
            input_data = {"input": np.array(data)}
            # Temp fix for supporting keras2 & keras3
            if int(keras.__version__.split(".")[0]) >= 3:
                input_data = input_data["input"]

            if self.load_type_model_from is not None:
                predict_data = self.keras_model_type.predict_on_batch(input_data)
                #print(predict_data)
                #print(predict_data[:, 1])
                prediction.extend(predict_data[:, 1])
            if self.load_energy_model_from is not None:
                predict_data = self.keras_model_energy.predict_on_batch(input_data)
                #print(predict_data)
                energy.extend(predict_data["energy"])
            if self.load_direction_model_from is not None:
                predict_data = self.keras_model_direction.predict_on_batch(input_data)
                #print(predict_data)

                az.extend(predict_data["direction"].T[0])
                alt.extend(predict_data["direction"].T[1])

        if self.load_type_model_from is not None:
            classification_table = output_identifiers.copy()
            classification_table.add_column(prediction, name="prediction")
            # Save the prediction to the output file
            write_table(
                classification_table,
                self.output_path,
                f"/dl2/event/telescope/classification/{self.reco_algo}/tel_{self.tel_id:03d}",
            )
            self.log.info(
                "DL2 prediction data was stored in '%s' under '%s'",
                self.output_path,
                f"/dl2/event/telescope/classification/{self.reco_algo}/tel_{self.tel_id:03d}",
            )
        if self.load_energy_model_from is not None:
            energy_table = output_identifiers.copy()
            # Convert the reconstructed energy from log10(TeV) to TeV
            reco_energy = u.Quantity(
                np.power(10, np.squeeze(energy)), unit=u.TeV
            )
            # Add the reconstructed energy to the prediction table
            energy_table.add_column(reco_energy, name="energy")
            # Save the prediction to the output file
            write_table(
                energy_table,
                self.output_path,
                f"/dl2/event/telescope/energy/{self.reco_algo}/tel_{self.tel_id:03d}",
            )
            self.log.info(
                "DL2 prediction data was stored in '%s' under '%s'",
                self.output_path,
                f"/dl2/event/telescope/energy/{self.reco_algo}/tel_{self.tel_id:03d}",
            )

        if self.load_direction_model_from is not None:
            direction_table = output_identifiers.copy()
            # Convert reconstructed spherical offset (az, alt) to SkyCoord
            reco_spherical_offset_az = u.Quantity(az, unit=u.deg)
            reco_spherical_offset_alt = u.Quantity(alt, unit=u.deg)
            # Set the telescope pointing of the SkyOffsetSeparation tranformation
            pointing = SkyCoord(tel_az, tel_alt, frame="altaz")
            # Calculate the reconstructed direction (az, alt) based on the telescope pointing
            reco_direction = pointing.spherical_offsets_by(
                reco_spherical_offset_az, reco_spherical_offset_alt
            ).to_table()
            # Add the reconstructed direction (az, alt) to the prediction table
            direction_table.add_column(reco_direction["az"], name="az")
            direction_table.add_column(reco_direction["alt"], name="alt")

            if self.store_event_wise_pointing:
                # Add the event-wise telescope pointing to the prediction table
                direction_table.add_column(tel_az, name="telescope_pointing_azimuth")
                direction_table.add_column(tel_alt, name="telescope_pointing_altitude")
                # Add the timestamp of the event to the prediction table
                direction_table.add_column(time, name="time")
            
            # Save the prediction to the output file
            write_table(
                direction_table,
                self.output_path,
                f"/dl2/event/telescope/geometry/{self.reco_algo}/tel_{self.tel_id:03d}",
            )
            self.log.info(
                "DL2 prediction data was stored in '%s' under '%s'",
                self.output_path,
                f"/dl2/event/telescope/geometry/{self.reco_algo}/tel_{self.tel_id:03d}",
            )

    def finish(self):
        self.log.info("Tool is shutting down")



def main():
    # Run the tool
    tool = LST1PredictionTool()
    tool.run()


if __name__ == "main":
    main()




