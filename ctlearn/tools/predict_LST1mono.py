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

class LST1MonoPredictionTool(Tool):
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

    table_name = Unicode(
        #default_value="/dl1/event/telescope/images/tel_001",
        default_value="/dl1/event/telescope/image/LST_LSTCam",
        allow_none=False,
        help="Table name of the pixel-wise image data to be used",
    ).tag(config=True)

    
    load_model_from = Path(
        default_value=None,
        help="Path to a Keras model file (Keras3) or directory Keras2)",
        allow_none=True,
        exists=True,
        directory_ok=True,
        file_ok=True,
    ).tag(config=True) 

    reco_tasks = List(
        trait=CaselessStrEnum(["type", "energy", "direction"]),
        allow_none=False, 
        help=(
            "List of reconstruction tasks to perform. "
            "'type': classification of the primary particle type "
            "'energy': regression of the primary particle energy "
            "'direction': regression of the primary particle arrival direction "
        )
    ).tag(config=True)

    batch_size = Int(
        default_value=64,
        allow_none=False,
        help="Size of the batch to perform inference of the neural network.",
    ).tag(config=True)

    image_mapper_type = Unicode(
        default_value="BilinearMapper",
        allow_none=False,
        help=(
            "Instances of ``ImageMapper`` transforming a raw 1D vector into a 2D image. "
            "Different mapping methods can be selected for the telescope type."
        ),
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

    output_path = Path(
        default_value="./dl2_prediction.h5",
        allow_none=False,
        help="Output path to save the dl2 prediction results",
    ).tag(config=True)

    output_table_name = Unicode(
        default_value="/dl2/event/telescope/LST_LSTCam",
        #default_value="/dl2/event/telescope/LST_LSTCam",
        allow_none=False,
        help="Table name of the dl2 prediction data to be saved",
    ).tag(config=True)

    overwrite = Bool(help="Overwrite output file if it exists").tag(config=True)

    aliases = {
        ("i", "input_url"): "LST1MonoPredictionTool.input_url",
        "reco": "LST1MonoPredictionTool.reco_tasks",
        ("o", "output"): "LST1MonoPredictionTool.output_path",
    }

    flags = {
        "overwrite": (
            {"LST1MonoPredictionTool": {"overwrite": True}},
            "Overwrite existing files",
        ),
    }

    classes = classes_with_traits(ImageMapper)

    def setup(self):

        # Load the subarray description from the input file
        #self.subarray = SubarrayDescription.from_hdf(self.input_url)

        # Get the number of rows in the table
        with tables.open_file(self.input_url) as input_file:
            self.table_length = len(input_file.get_node(self.table_name))

        # Load the model from the specified path
        self.log.info("Loading the model from %s.", self.load_model_from)
        self.model = keras.saving.load_model(self.load_model_from)
        self.input_shape = self.model.input_shape[1:]

        # Create the image mappers
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
            # find out why this is not working
            #cam_geom_table = read_table(self.input_url, "/configuration/instrument/telescope/camera/geometry_0")
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
            geometry=cam_geom[self.camera_name], interpolation_image_shape=self.input_shape[0], subarray=self.subarray
        )
        print(self.image_mappers[self.camera_name].image_shape)
        #self.image_mappers[self.camera_name] = ImageMapper.from_name(
        #    self.image_mapper_type, geometry=cam_geom[self.camera_name], interpolation_image_shape=114, subarray=self.subarray, parent=self
        #)
        print(self.image_mappers)

        # Get offset and scaling of images
        self.transforms = {}
        self.transforms["image_scale"] = 0.0
        self.transforms["image_offset"] = 0
        self.transforms["peak_time_scale"] = 0.0
        self.transforms["peak_time_offset"] = 0
        # Get the number of rows in the table
        with tables.open_file(self.input_url) as input_file:
            img_table_v_attrs = input_file.get_node(self.table_name)._v_attrs
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

        # Iterate over the data in chunks based on the batch size
        for start in range(0, self.table_length, self.batch_size):
            stop = min(start + self.batch_size, self.table_length)
            print(f"Processing chunk from {start} to {stop - 1}")
            # Read the data
            dl1_table = read_table(self.input_url, self.table_name, start=start, stop=stop)
            print(dl1_table[0])
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

            predict_data = self.model.predict_on_batch(input_data)
            self.dl2_prediction = {}
            if "type" in self.reco_tasks:
                dl2_prediction = predict_data
            if "direction" in self.reco_tasks:
                dl2_prediction = Table
            if "energy" in self.reco_tasks:
                dl2_prediction = predict_data["direction"]
            print(dl2_prediction)



    def finish(self):
        # Save the prediction to the output file
        write_table(
            Table(self.dl2_prediction),
            self.output_path,
            self.output_table_name,
            append=True,
        )
        self.log.info("Tool is shutting down")


def main():
    # Run the tool
    tool = LST1MonoPredictionTool()
    tool.run()


if __name__ == "main":
    main()




