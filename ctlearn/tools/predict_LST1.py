"""
Predict the gammaness, energy and arrival direction from lstchain DL1 data.
"""

import pathlib
import numpy as np
import tables
import keras
from astropy import units as u
from astropy.coordinates.earth import EarthLocation
from astropy.coordinates import AltAz, Angle, SkyCoord
from astropy.table import Table, join, setdiff, vstack
from astropy.time import Time

from ctapipe.containers import (
    TelEventIndexContainer,
    EventIndexContainer,
    TelescopePointingContainer,
    TelescopeTriggerContainer,
    TriggerContainer,
    ParticleClassificationContainer,
    ReconstructedGeometryContainer,
    ReconstructedEnergyContainer,
)
from ctapipe.coordinates import CameraFrame, EngineeringCameraFrame
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
    ComponentName,
    Unicode,
    classes_with_traits,
)
from ctapipe.instrument import (
    SubarrayDescription,
    CameraDescription,
    CameraGeometry,
    CameraReadout,
    TelescopeDescription,
)
from ctapipe.instrument.optics import OpticsDescription, ReflectorShape, SizeType
from ctapipe.io import read_table, write_table
from ctapipe.reco.utils import add_defaults_and_meta

from ctlearn.core.model import LoadedModel
from dl1_data_handler.image_mapper import ImageMapper
from dl1_data_handler.reader import (
    get_unmapped_image,
    get_unmapped_waveform,
    TableQualityQuery,
    LST_EPOCH,
)

POINTING_GROUP = "/dl1/monitoring/telescope/pointing"
DL1_TELESCOPE_GROUP = "/dl1/event/telescope"
DL2_TELESCOPE_GROUP = "/dl2/event/telescope"
DL2_SUBARRAY_GROUP = "/dl2/event/subarray"
SUBARRAY_EVENT_KEYS = ["obs_id", "event_id"]
TELESCOPE_EVENT_KEYS = ["obs_id", "event_id", "tel_id"]


class LST1PredictionTool(Tool):
    """
    Tool to predict the gammaness, energy and arrival direction from lstchain DL1 data.

    This tool is used to predict the gammaness, energy and arrival direction
    from pixel-wise image data in lstchain format. The tool loads the trained models
    from the specified paths and performs inference on the input data. The
    input data is expected to be in the DL1 format of lstchain and the output data is
    stored in the DL2 format of ctapipe. Besides the DL2 predictions, the tool creates
    the SubarrayDescription of the LST-1 telescope and stores it in the output file.
    In addition, the tool also creates the trigger, pointing and DL1 parameters tables
    and stores them in the output file.

    CAUTION: The tool is designed to work with the DL1 data format of lstchain only.
    """

    name = "LST1PredictionTool"
    description = __doc__

    examples = """
    To predict from DL1 lstchain data using trained CTLearn models:
    > ctlearn-predict-model \\
        --input_url input.subrun.lstchain.dl1.h5 \\
        --LST1PredictionTool.batch_size=64 \\
        --LST1PredictionTool.channels=cleaned_image \\
        --LST1PredictionTool.channels=cleaned_relative_peak_time \\
        --LST1PredictionTool.image_mapper_type=BilinearMapper \\
        --type_model="/path/to/your/type/ctlearn_model.cpk" \\
        --energy_model="/path/to/your/energy/ctlearn_model.cpk" \\
        --cameradirection_model="/path/to/your/direction/ctlearn_model.cpk" \\
        --output output.dl2.h5 \\
        --overwrite \\
    """

    input_url = Path(
        help="Input LST-1 HDF5 files including pixel-wise image data",
        allow_none=True,
        exists=True,
        directory_ok=False,
        file_ok=True,
    ).tag(config=True)

    prefix = Unicode(
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

    load_cameradirection_model_from = Path(
        default_value=None,
        help=(
            "Path to a Keras model file (Keras3) or directory (Keras2) "
            "for the regression of the primary particle arrival direction "
            "based on the camera coordinate offsets."
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
                "cleaned_relative_peak_time",
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
            "cleaned_peak_time: extracted peak arrival times cleaned with the DL1 cleaning mask, "
            "cleaned_relative_peak_time: extracted relative peak arrival times cleaned with the DL1 cleaning mask."
        ),
    ).tag(config=True)

    image_mapper_type = ComponentName(ImageMapper, default_value="BilinearMapper").tag(
        config=True
    )

    override_obs_id = Int(
        default_value=None,
        allow_none=True,
        help=(
            "Use the given obs_id instead of the default one. "
            "Needed to merge subruns later with ctapipe-merge."
        ),
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
        ("d", "cameradirection_model"): "LST1PredictionTool.load_cameradirection_model_from",
        ("o", "output"): "LST1PredictionTool.output_path",
    }

    flags = {
        "overwrite": (
            {"LST1PredictionTool": {"overwrite": True}},
            "Overwrite existing files",
        ),
    }

    classes = classes_with_traits(ImageMapper)

    def setup(self):
        # Save dl1 image and parameters tree schemas and tel id for easy access
        self.image_table_path = "/dl1/event/telescope/image/LST_LSTCam"
        self.parameter_table_name = "/dl1/event/telescope/parameters/LST_LSTCam"
        self.tel_id = 1

        # Get the number of rows in the table
        with tables.open_file(self.input_url) as input_file:
            self.table_length = len(input_file.get_node(self.image_table_path))

        # Load the models from the specified paths
        if self.load_type_model_from is not None:
            self.log.info("Loading the type model from %s.", self.load_type_model_from)
            model_type = keras.saving.load_model(self.load_type_model_from)
            input_shape = model_type.input_shape[1:]
            self.backbone_type, self.head_type = self._split_model(model_type)
        if self.load_energy_model_from is not None:
            self.log.info(
                "Loading the energy model from %s.", self.load_energy_model_from
            )
            model_energy = keras.saving.load_model(
                self.load_energy_model_from
            )
            input_shape = model_energy.input_shape[1:]
            self.backbone_energy, self.head_energy = self._split_model(model_energy)
        if self.load_cameradirection_model_from is not None:
            self.log.info(
                "Loading the cameradirection model from %s.", self.load_cameradirection_model_from
            )
            model_direction = keras.saving.load_model(
                self.load_cameradirection_model_from
            )
            input_shape = model_direction.input_shape[1:]
            self.backbone_direction, self.head_direction = self._split_model(model_direction)

        # Create the SubarrayDescription of the LST-1 telescope
        self.subarray = self._create_subarray()
        # Write the SubarrayDescription to the output file
        self.subarray.to_hdf(self.output_path, overwrite=self.overwrite)
        self.log.info("SubarrayDescription was stored in '%s'", self.output_path)
        # Initialize the Table data quality query
        self.quality_query = TableQualityQuery(parent=self)
        # Copy the pixel rotation of the camera geometry of the subarray in a variable
        # since the ImageMapper will be derotated the pixels. The pixel rotation
        # is needed to create a rotated camera frame in order to transform the
        # predicted camera coordinate offsets back to the correct Alt/Az coordinates.
        self.pix_rotation = self.subarray.tel[self.tel_id].camera.geometry.pix_rotation
        # Create the ImageMapper
        self.image_mapper = ImageMapper.from_name(
            name=self.image_mapper_type,
            geometry=self.subarray.tel[self.tel_id].camera.geometry,
            subarray=self.subarray,
            parent=self,
        )
        # Check if the input shape of the model matches the image shape of the ImageMapper
        if input_shape[0] != self.image_mapper.image_shape:
            raise ToolConfigurationError(
                f"The input shape of the model ('{input_shape[0]}') does not match "
                f"the image shape of the ImageMapper ('{self.image_mapper.image_shape}'). "
                f"Use e.g. '--BilinearMapper.interpolation_image_shape={input_shape[0]}' ."
            )

        # Get offset and scaling of images
        self.transforms = {}
        self.transforms["image_scale"] = 0.0
        self.transforms["image_offset"] = 0
        self.transforms["peak_time_scale"] = 0.0
        self.transforms["peak_time_offset"] = 0
        # Get the number of rows in the table
        with tables.open_file(self.input_url) as input_file:
            img_table_v_attrs = input_file.get_node(self.image_table_path)._v_attrs

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

        all_identifiers = read_table(self.input_url, self.parameter_table_name)
        all_identifiers.meta = {}
        if self.override_obs_id is not None:
            all_identifiers["obs_id"] = self.override_obs_id
        self.obs_id = all_identifiers["obs_id"][0]
        parameter_table = all_identifiers.copy()
        tel_az = u.Quantity(parameter_table["az_tel"], unit=u.rad)
        tel_alt = u.Quantity(parameter_table["alt_tel"], unit=u.rad)
        event_type = parameter_table["event_type"]
        time = Time(parameter_table["dragon_time"] * u.s, format="unix")
        # Create the pointing table
        # This table is used to store the telescope pointing per event
        pointing_table = Table(
            {
                "time": time,
                "azimuth": tel_az,
                "altitude": tel_alt,
            }
        )
        write_table(
            pointing_table,
            self.output_path,
            f"{POINTING_GROUP}/tel_{self.tel_id:03d}",
            overwrite=self.overwrite,
        )
        self.log.info(
            "DL1 telescope pointing table was stored in '%s' under '%s'",
            self.output_path,
            f"{POINTING_GROUP}/tel_{self.tel_id:03d}",
        )
        # Set the time format to MJD since in the other table we store the time in MJD
        time.format = "mjd"
        # Keep only the necessary columns for the creation of tables
        all_identifiers.keep_columns(TELESCOPE_EVENT_KEYS)
        # Create the dl1 telescope trigger table
        trigger_table = all_identifiers.copy()
        trigger_table.add_column(time, name="time")
        trigger_table.add_column(-1, name="n_trigger_pixels")
        write_table(
            trigger_table,
            self.output_path,
            "/dl1/event/telescope/trigger",
            overwrite=self.overwrite,
        )
        self.log.info(
            "DL1 telescope trigger table was stored in '%s' under '%s'",
            self.output_path,
            "/dl1/event/telescope/trigger",
        )
        trigger_table.keep_columns(["obs_id", "event_id", "time"])
        trigger_table.add_column(
            np.ones((len(trigger_table), 1), dtype=bool), name="tel_with_trigger"
        )
        trigger_table.add_column(event_type, name="event_type")
        # Save the dl1 subrray trigger table to the output file
        write_table(
            trigger_table,
            self.output_path,
            "/dl1/event/subarray/trigger",
            overwrite=self.overwrite,
        )
        self.log.info(
            "DL1 subarray trigger table was stored in '%s' under '%s'",
            self.output_path,
            "/dl1/event/subarray/trigger",
        )
        # Create the dl1 parameters table
        parameter_table.rename_column("intensity", "hillas_intensity")
        parameter_table.rename_column("x", "hillas_x")
        parameter_table.rename_column("y", "hillas_y")
        parameter_table.rename_column("phi", "hillas_phi")
        parameter_table.rename_column("psi", "hillas_psi")
        parameter_table.rename_column("length", "hillas_length")
        parameter_table.rename_column("length_uncertainty", "hillas_length_uncertainty")
        parameter_table.rename_column("width", "hillas_width")
        parameter_table.rename_column("width_uncertainty", "hillas_width_uncertainty")
        parameter_table.rename_column("skewness", "hillas_skewness")
        parameter_table.rename_column("kurtosis", "hillas_kurtosis")
        parameter_table.rename_column("time_gradient", "timing_deviation")
        parameter_table.rename_column("intercept", "timing_intercept")
        parameter_table.rename_column("n_pixels", "morphology_n_pixels")
        parameter_table.rename_column("n_islands", "morphology_n_islands")
        parameter_table.keep_columns(
            [
                "obs_id",
                "event_id",
                "hillas_intensity",
                "hillas_x",
                "hillas_y",
                "hillas_phi",
                "hillas_psi",
                "hillas_length",
                "hillas_length_uncertainty",
                "hillas_width",
                "hillas_width_uncertainty",
                "hillas_skewness",
                "hillas_kurtosis",
                "timing_deviation",
                "timing_intercept",
                "morphology_n_pixels",
                "morphology_n_islands",
            ]
        )
        parameter_table.add_column(self.tel_id, name="tel_id", index=2)
        # Save the dl1 parameters table to the output file
        write_table(
            parameter_table,
            self.output_path,
            f"/dl1/event/telescope/parameters/tel_{self.tel_id:03d}",
            overwrite=self.overwrite,
        )
        self.log.info(
            "DL1 parameters table was stored in '%s' under '%s'",
            self.output_path,
            f"/dl1/event/telescope/parameters/tel_{self.tel_id:03d}",
        )

        # Add additional columns to the parameter table
        # which are not present in the originl DL1 parameter table.
        # They are needed for applying the quality selection.
        parameter_table.add_column(event_type, name="event_type")
        parameter_table.add_column(tel_az, name="tel_az")
        parameter_table.add_column(tel_alt, name="tel_alt")
        # Only select cosmic events for the prediction
        parameter_table = parameter_table[parameter_table["event_type"]==32]

        self.log.info("Starting the prediction...")
        event_id, tel_azimuth, tel_altitude, trigger_time = [], [], [], []
        prediction, energy, cam_coord_offset_x, cam_coord_offset_y = [], [], [], []
        classification_fvs, energy_fvs, direction_fvs = [], [], []
        # Iterate over the data in chunks based on the batch size
        for start in range(0, self.table_length, self.batch_size):
            stop = min(start + self.batch_size, self.table_length)
            self.log.debug("Processing chunk from '%d' to '%d'.", start, stop - 1)
            # Read the data
            dl1_table = read_table(
                self.input_url, self.image_table_path, start=start, stop=stop
            )
            # Join the dl1 table with the parameter table to perform quality selection
            dl1_table = join(
                left=dl1_table,
                right=parameter_table,
                keys=["event_id"],
            )
            dl1_table = join(
                left=dl1_table,
                right=trigger_table,
                keys=["event_id"],
            )
            # Initialize a boolean mask to True for all events in the sliced dl1 table
            passes_quality_checks = np.ones(len(dl1_table), dtype=bool)
            # Quality selection based on the dl1b parameter
            if self.quality_query:
                passes_quality_checks = self.quality_query.get_table_mask(dl1_table)
            # Apply the mask to filter events that are not fufilling the quality criteria
            dl1_table = dl1_table[passes_quality_checks]
            if len(dl1_table) == 0:
                self.log.debug("No events passed the quality selection.")
                continue
            data = []
            for event in dl1_table:
                # Get the unmapped image
                image = get_unmapped_image(event, self.channels, self.transforms)
                data.append(self.image_mapper.map_image(image))
            input_data = {"input": np.array(data)}
            # Temp fix for supporting keras2 & keras3
            if int(keras.__version__.split(".")[0]) >= 3:
                input_data = input_data["input"]

            event_id.extend(dl1_table["event_id"].data)
            tel_azimuth.extend(dl1_table["tel_az"].data)
            tel_altitude.extend(dl1_table["tel_alt"].data)
            trigger_time.extend(dl1_table["time"].mjd)
            if self.load_type_model_from is not None:
                classification_feature_vectors = self.backbone_type.predict_on_batch(input_data)
                classification_fvs.extend(classification_feature_vectors)
                predict_data = self.head_type.predict_on_batch(classification_feature_vectors)
                prediction.extend(predict_data[:, 1])
            if self.load_energy_model_from is not None:
                energy_feature_vectors = self.backbone_energy.predict_on_batch(input_data)
                energy_fvs.extend(energy_feature_vectors)
                predict_data = self.head_energy.predict_on_batch(energy_feature_vectors)
                energy.extend(predict_data.T[0])
            if self.load_cameradirection_model_from is not None:
                direction_feature_vectors = self.backbone_direction.predict_on_batch(input_data)
                direction_fvs.extend(direction_feature_vectors)
                predict_data = self.head_direction.predict_on_batch(direction_feature_vectors)
                cam_coord_offset_x.extend(predict_data.T[0])
                cam_coord_offset_y.extend(predict_data.T[1])

        # Create the prediction tables
        example_identifiers = Table(
            {
                "obs_id": np.full(len(event_id), self.obs_id, dtype=int),
                "event_id": event_id,
                "tel_id": np.full(len(event_id), self.tel_id, dtype=int),
            }
        )
        nonexample_identifiers = setdiff(
            all_identifiers, example_identifiers, keys=TELESCOPE_EVENT_KEYS
        )
        if len(nonexample_identifiers) > 0:
            nonexample_identifiers.sort(TELESCOPE_EVENT_KEYS)
        # Create the feature vector table
        feature_vector_table = example_identifiers.copy()
        fvs_columns_list, fvs_shapes_list = [], []
        if self.load_type_model_from is not None:
            classification_table = example_identifiers.copy()
            classification_table.add_column(
                prediction, name=f"{self.prefix}_tel_prediction"
            )
            # Produce output table with NaNs for missing predictions
            if len(nonexample_identifiers) > 0:
                nan_table = self._create_nan_table(
                    nonexample_identifiers,
                    columns=[f"{self.prefix}_tel_prediction"],
                    shapes=[(len(nonexample_identifiers),)],
                )
                classification_table = vstack([classification_table, nan_table])
            classification_table.sort(TELESCOPE_EVENT_KEYS)
            classification_is_valid = ~np.isnan(classification_table[f"{self.prefix}_tel_prediction"].data, dtype=bool)
            classification_table.add_column(
                classification_is_valid,
                name=f"{self.prefix}_tel_is_valid",
            )
            # Add the default values and meta data to the table
            add_defaults_and_meta(
                classification_table,
                ParticleClassificationContainer,
                prefix=self.prefix,
                add_tel_prefix=True,
            )
            # Save the prediction to the output file
            write_table(
                classification_table,
                self.output_path,
                f"{DL2_TELESCOPE_GROUP}/classification/{self.prefix}/tel_{self.tel_id:03d}",
                overwrite=self.overwrite,
            )
            self.log.info(
                "DL2 prediction data was stored in '%s' under '%s'",
                self.output_path,
                f"{DL2_TELESCOPE_GROUP}/classification/{self.prefix}/tel_{self.tel_id:03d}",
            )
            # Write the mono telescope prediction to the subarray prediction table
            subarray_classification_table = classification_table.copy()
            subarray_classification_table.remove_column("tel_id")
            for colname in subarray_classification_table.colnames:
                if "_tel_" in colname:
                    subarray_classification_table.rename_column(
                        colname, colname.replace("_tel", "")
                    )
            subarray_classification_table.add_column(
                classification_is_valid[np.newaxis], name=f"{self.prefix}_telescopes"
            )
            # Save the prediction to the output file
            write_table(
                subarray_classification_table,
                self.output_path,
                f"{DL2_SUBARRAY_GROUP}/classification/{self.prefix}",
                overwrite=self.overwrite,
            )
            self.log.info(
                "DL2 prediction data was stored in '%s' under '%s'",
                self.output_path,
                f"{DL2_SUBARRAY_GROUP}/classification/{self.prefix}",
            )
            # Adding the feature vectors for the classification
            is_valid_col = ~np.isnan(
                np.min(classification_fvs, axis=1), dtype=bool
            )
            feature_vector_table.add_column(
                classification_fvs,
                name=f"{self.prefix}_tel_classification_feature_vectors",
            )
            if nonexample_identifiers is not None:
                fvs_columns_list.append(f"{self.prefix}_tel_classification_feature_vectors")
                fvs_shapes_list.append(
                    (
                        len(nonexample_identifiers),
                        classification_fvs[0].shape[0],
                    )
                )
        if self.load_energy_model_from is not None:
            energy_table = example_identifiers.copy()
            # Convert the reconstructed energy from log10(TeV) to TeV
            reco_energy = u.Quantity(np.power(10, np.squeeze(energy)), unit=u.TeV)
            # Add the reconstructed energy to the prediction table
            energy_table.add_column(reco_energy, name=f"{self.prefix}_tel_energy")
            # Produce output table with NaNs for missing predictions
            if len(nonexample_identifiers) > 0:
                nan_table = self._create_nan_table(
                    nonexample_identifiers,
                    columns=[f"{self.prefix}_tel_energy"],
                    shapes=[(len(nonexample_identifiers),)],
                )
                energy_table = vstack([energy_table, nan_table])
            energy_table.sort(TELESCOPE_EVENT_KEYS)
            energy_is_valid = ~np.isnan(energy_table[f"{self.prefix}_tel_energy"].data, dtype=bool)
            energy_table.add_column(
                energy_is_valid,
                name=f"{self.prefix}_tel_is_valid",
            )
            # Add the default values and meta data to the table
            add_defaults_and_meta(
                energy_table,
                ReconstructedEnergyContainer,
                prefix=self.prefix,
                add_tel_prefix=True,
            )
            # Save the prediction to the output file
            write_table(
                energy_table,
                self.output_path,
                f"{DL2_TELESCOPE_GROUP}/energy/{self.prefix}/tel_{self.tel_id:03d}",
                overwrite=self.overwrite,
            )
            self.log.info(
                "DL2 prediction data was stored in '%s' under '%s'",
                self.output_path,
                f"{DL2_TELESCOPE_GROUP}/energy/{self.prefix}/tel_{self.tel_id:03d}",
            )
            # Write the mono telescope prediction to the subarray prediction table
            subarray_energy_table = energy_table.copy()
            subarray_energy_table.remove_column("tel_id")
            for colname in subarray_energy_table.colnames:
                if "_tel_" in colname:
                    subarray_energy_table.rename_column(
                        colname, colname.replace("_tel", "")
                    )
            subarray_energy_table.add_column(
                energy_is_valid[np.newaxis], name=f"{self.prefix}_telescopes"
            )
            # Save the prediction to the output file
            write_table(
                subarray_energy_table,
                self.output_path,
                f"{DL2_SUBARRAY_GROUP}/energy/{self.prefix}",
                overwrite=self.overwrite,
            )
            self.log.info(
                "DL2 prediction data was stored in '%s' under '%s'",
                self.output_path,
                f"{DL2_SUBARRAY_GROUP}/energy/{self.prefix}",
            )
            # Adding the feature vectors for the energy regression
            is_valid_col = ~np.isnan(
                np.min(energy_fvs, axis=1), dtype=bool
            )
            feature_vector_table.add_column(
                energy_fvs,
                name=f"{self.prefix}_tel_energy_feature_vectors",
            )
            if nonexample_identifiers is not None:
                fvs_columns_list.append(f"{self.prefix}_tel_energy_feature_vectors")
                fvs_shapes_list.append(
                    (
                        len(nonexample_identifiers),
                        energy_fvs[0].shape[0],
                    )
                )
        if self.load_cameradirection_model_from is not None:
            direction_table = example_identifiers.copy()
            # Set the telescope position
            tel_ground_frame = self.subarray.tel_coords[
                self.subarray.tel_ids_to_indices(self.tel_id)
            ]
            # Set the telescope pointing with the trigger timestamp and the telescope position
            trigger_time = Time(trigger_time, format="mjd")
            altaz = AltAz(
                location=tel_ground_frame.to_earth_location(),
                obstime=trigger_time,
            )
            # Set the telescope pointing
            tel_pointing = SkyCoord(
                az=u.Quantity(tel_azimuth, unit=u.rad),
                alt=u.Quantity(tel_altitude, unit=u.rad),
                frame=altaz,
            )
            # Set the camera frame with the focal length and rotation of the camera
            camera_frame = CameraFrame(
                focal_length=self.subarray.tel[self.tel_id].optics.equivalent_focal_length,
                rotation=self.pix_rotation,
                telescope_pointing=tel_pointing,
            )
            # Set the camera coordinate offset
            cam_coord_offset = SkyCoord(
                x=u.Quantity(cam_coord_offset_x, unit=u.m),
                y=u.Quantity(cam_coord_offset_y, unit=u.m),
                frame=camera_frame
            )
            # Transform the true Alt/Az coordinates to camera coordinates
            reco_direction = cam_coord_offset.transform_to(altaz)
            # Add the reconstructed direction (az, alt) to the prediction table
            direction_table.add_column(
                reco_direction.az.to(u.deg), name=f"{self.prefix}_tel_az"
            )
            direction_table.add_column(
                reco_direction.alt.to(u.deg), name=f"{self.prefix}_tel_alt"
            )
            # Produce output table with NaNs for missing predictions
            if len(nonexample_identifiers) > 0:
                nan_table = self._create_nan_table(
                    nonexample_identifiers,
                    columns=[f"{self.prefix}_tel_az", f"{self.prefix}_tel_alt"],
                    shapes=[(len(nonexample_identifiers),), (len(nonexample_identifiers),)],
                )
                direction_table = vstack([direction_table, nan_table])
            direction_table.keep_columns(
                TELESCOPE_EVENT_KEYS
                + [f"{self.prefix}_tel_az", f"{self.prefix}_tel_alt"]
            )
            direction_table.sort(TELESCOPE_EVENT_KEYS)
            direction_is_valid = ~np.isnan(direction_table[f"{self.prefix}_tel_az"].data, dtype=bool)
            direction_table.add_column(
                direction_is_valid,
                name=f"{self.prefix}_tel_is_valid",
            )
            # Add the default values and meta data to the table
            add_defaults_and_meta(
                direction_table,
                ReconstructedGeometryContainer,
                prefix=self.prefix,
                add_tel_prefix=True,
            )
            # Save the prediction to the output file
            write_table(
                direction_table,
                self.output_path,
                f"{DL2_TELESCOPE_GROUP}/geometry/{self.prefix}/tel_{self.tel_id:03d}",
                overwrite=self.overwrite,
            )
            self.log.info(
                "DL2 prediction data was stored in '%s' under '%s'",
                self.output_path,
                f"{DL2_TELESCOPE_GROUP}/geometry/{self.prefix}/tel_{self.tel_id:03d}",
            )
            # Write the mono telescope prediction to the subarray prediction table
            subarray_direction_table = direction_table.copy()
            subarray_direction_table.remove_column("tel_id")
            for colname in subarray_direction_table.colnames:
                if "_tel_" in colname:
                    subarray_direction_table.rename_column(
                        colname, colname.replace("_tel", "")
                    )
            subarray_direction_table.add_column(
                direction_is_valid[np.newaxis], name=f"{self.prefix}_telescopes"
            )
            # Save the prediction to the output file
            write_table(
                subarray_direction_table,
                self.output_path,
                f"{DL2_SUBARRAY_GROUP}/geometry/{self.prefix}",
                overwrite=self.overwrite,
            )
            self.log.info(
                "DL2 prediction data was stored in '%s' under '%s'",
                self.output_path,
                f"{DL2_SUBARRAY_GROUP}/geometry/{self.prefix}",
            )
            # Adding the feature vectors for the arrival direction regression
            is_valid_col = ~np.isnan(
                np.min(direction_fvs, axis=1), dtype=bool
            )
            feature_vector_table.add_column(
                direction_fvs,
                name=f"{self.prefix}_tel_direction_feature_vectors",
            )
            if nonexample_identifiers is not None:
                fvs_columns_list.append(f"{self.prefix}_tel_direction_feature_vectors")
                fvs_shapes_list.append(
                    (
                        len(nonexample_identifiers),
                        direction_fvs[0].shape[0],
                    )
                )
        # Produce output table with NaNs for missing predictions
        if nonexample_identifiers is not None:
            if len(nonexample_identifiers) > 0:
                nan_table = self._create_nan_table(
                    nonexample_identifiers,
                    columns=fvs_columns_list,
                    shapes=fvs_shapes_list,
                )
                feature_vector_table = vstack([feature_vector_table, nan_table])
                is_valid_col = np.concatenate(
                    (is_valid_col, np.zeros(len(nonexample_identifiers), dtype=bool))
                )
        # Add is_valid column to the feature vector table
        feature_vector_table.add_column(
            is_valid_col,
            name=f"{self.prefix}_tel_is_valid",
        )
        # Save the prediction to the output file
        write_table(
            feature_vector_table,
            self.output_path,
            f"{DL1_TELESCOPE_GROUP}/features/{self.prefix}/tel_{self.tel_id:03d}",
            overwrite=self.overwrite,
        )
        self.log.info(
            "DL1 feature vectors was stored in '%s' under '%s'",
            self.output_path,
            f"{DL1_TELESCOPE_GROUP}/features/{self.prefix}/tel_{self.tel_id:03d}",
        )

    def finish(self):
        self.log.info("Tool is shutting down")

    def _create_subarray(self, tel_id=1, reference_location=None):
        """
        Obtain a single-lst subarray description (Code taken from ctapipe_io_lst)

        Returns
        -------
        ctapipe.instrument.SubarrayDescription
        """

        self.log.info("Make sure 'ctapipe_io_lst' is installed in your enviroment!")
        try:
            import ctapipe_io_lst
        except ImportError:
            raise ImportError("'ctapipe_io_lst' is not installed in your environment!")

        if reference_location is None:
            reference_location = ctapipe_io_lst.constants.LST1_LOCATION

        # Load the camera geometry from 'ctapipe_io_lst'
        camera_geom = ctapipe_io_lst.load_camera_geometry()
        # Needs to be renamed because the ImageMapper smooths the pixel positions
        camera_geom.name = "RealLSTCam"
        # get info on the camera readout:
        (
            daq_time_per_sample,
            pulse_shape_time_step,
            pulse_shapes,
        ) = ctapipe_io_lst.read_pulse_shapes()

        camera_readout = CameraReadout(
            name="LSTCam",
            n_pixels=ctapipe_io_lst.constants.N_PIXELS,
            n_channels=ctapipe_io_lst.constants.N_GAINS,
            n_samples=ctapipe_io_lst.constants.N_SAMPLES,
            sampling_rate=(1 / daq_time_per_sample).to(u.GHz),
            reference_pulse_shape=pulse_shapes,
            reference_pulse_sample_width=pulse_shape_time_step,
        )

        camera = CameraDescription(
            name="LSTCam", geometry=camera_geom, readout=camera_readout
        )

        lst_tel_descr = TelescopeDescription(
            name="LST", optics=ctapipe_io_lst.OPTICS, camera=camera
        )

        tel_descriptions = {tel_id: lst_tel_descr}

        try:
            location = ctapipe_io_lst.constants.LST_LOCATIONS[tel_id]
        except KeyError:
            known = list(ctapipe_io_lst.constants.LST_LOCATIONS.keys())
            msg = f"Location missing for tel_id={tel_id}. Known tel_ids: {known}. Is this LST data?"
            raise KeyError(msg) from None

        ground_frame = ctapipe_io_lst.ground_frame.ground_frame_from_earth_location(
            location, reference_location
        )
        tel_positions = {tel_id: ground_frame.cartesian.xyz}
        subarray = SubarrayDescription(
            name=f"LST-{tel_id} subarray",
            tel_descriptions=tel_descriptions,
            tel_positions=tel_positions,
            reference_location=reference_location,
        )

        return subarray

    def _split_model(self, model):
        """
        Split the model into backbone and head.

        This method splits the model into backbone and head. The backbone is summarized
        into a single layer which can be retrieved by the layer index 1. The model input
        has layer index 0 and the head is the rest of the model with layer index 2 and above.

        Parameters:
        -----------
        model : keras.Model
            Keras model to split into backbone and head.

        Returns:
        --------
        backbone : keras.Model
            Backbone model of the original model.
        head : keras.Model
            Head model of the original model.
        """
        # Get the backbone model which is the second layer of the model
        backbone = model.get_layer(index=1)
        # Create a new head model with the same layers as the original model.
        # The output of the backbone model is the input of the head model.
        backbone_output_shape = keras.Input(model.layers[2].input_shape[1:])
        x = backbone_output_shape
        for layer in model.layers[2:]:
            x = layer(x)
        head = keras.Model(inputs=backbone_output_shape, outputs=x)
        return backbone, head

    def _create_nan_table(self, nonexample_identifiers, columns, shapes):
        """
        Create a table with NaNs for missing predictions.

        This method creates a table with NaNs for missing predictions for the non-example identifiers.

        Parameters:
        -----------
        nonexample_identifiers : astropy.table.Table
            Table containing the non-example identifiers.
        columns : list of str
            List of column names to create in the table.
        shapes : list of shapes
            List of shapes for the columns to create in the table.

        Returns:
        --------
        nan_table : astropy.table.Table
            Table containing NaNs for missing predictions.
        """
        # Create a table with NaNs for missing predictions
        nan_table = nonexample_identifiers.copy()
        for column_name, shape in zip(columns, shapes):
            nan_table.add_column(np.full(shape, np.nan), name=column_name)
        return nan_table


def main():
    # Run the tool
    tool = LST1PredictionTool()
    tool.run()


if __name__ == "main":
    main()
