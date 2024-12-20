"""
Tool to predict the gammaness, energy and arrival direction in monoscopic mode using ``CTLearnModel`` on R1/DL1a data using the ``DLDataReader`` and ``DLDataLoader``.
"""

import atexit
import pathlib
import numpy as np
import tensorflow as tf
import keras

from astropy import units as u
from astropy.coordinates.earth import EarthLocation
from astropy.coordinates import SkyCoord
from astropy.table import (
    Table,
    hstack,
    vstack,
    join,
    setdiff,
)

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
from ctapipe.monitoring.interpolation import PointingInterpolator
from ctapipe.io import read_table, write_table, HDF5Merger
from dl1_data_handler.reader import DLDataReader, ProcessType
from dl1_data_handler.loader import DLDataLoader

FIXED_POINTING_GROUP = "/configuration/telescope/pointing"
POINTING_GROUP = "/dl0/monitoring/telescope/pointing"
DL2_SUBARRAY_GROUP = "/dl2/event/subarray"
DL2_TELESCOPE_GROUP = "/dl2/event/telescope"
TELESCOPE_EVENT_KEYS = ["obs_id", "event_id", "tel_id"]

class MonoPredictionTool(Tool):
    """
    Tool to predict the gammaness, energy and arrival direction from R1/DL1a data using CTLearn models.

    The tool predicts the gammaness, energy and arrival direction from pixel-wise image or waveform data.
    The input data is loaded from the input url using the ``~dl1_data_handler.reader.DLDataReader`` and
    ``~dl1_data_handler.loader.DLDataLoader`` and the prediction is performed using the CTLearn models.
    """

    name = "MonoPredictionTool"
    description = __doc__

    examples = """
    To predict from pixel-wise image data using trained CTLearn models:
    > ctlearn-predict-mono \\
        --input_url input.dl1.h5 \\
        --MonoPredictionTool.tel_id=1 \\
        --MonoPredictionTool.batch_size=64 \\
        --MonoPredictionTool.dl1dh_reader_type=DLImageReader \\
        --DLImageReader.channels=cleaned_image \\
        --DLImageReader.channels=cleaned_relative_peak_time \\
        --DLImageReader.image_mapper_type=BilinearMapper \\
        --type_model="/path/to/your/type/ctlearn_model.cpk" \\
        --energy_model="/path/to/your/energy/ctlearn_model.cpk" \\
        --direction_model="/path/to/your/direction/ctlearn_model.cpk" \\
        --no-dl1-images \\
        --no-true-images \\
        --output output.dl2.h5 \\
        --overwrite \\

    To predict from pixel-wise waveform data using trained CTLearn models:
    > ctlearn-predict-mono \\
        --input_url input.r1.h5 \\
        --MonoPredictionTool.tel_id=1 \\
        --MonoPredictionTool.dl1dh_reader_type=DLWaveformReader \\
        --DLWaveformReader.sequnce_length=20 \\
        --DLWaveformReader.image_mapper_type=BilinearMapper \\
        --type_model="/path/to/your/type/ctlearn_model.cpk" \\
        --energy_model="/path/to/your/energy/ctlearn_model.cpk" \\
        --direction_model="/path/to/your/direction/ctlearn_model.cpk" \\
        --no-r0-waveforms \\
        --no-r1-waveforms \\
        --no-dl1-images \\
        --no-true-images \\
        --output output.dl2.h5 \\
        --overwrite \\
    """

    input_url = Path(
        help="Input ctapipe HDF5 files including pixel-wise image or waveform data",
        allow_none=True,
        exists=True,
        directory_ok=False,
        file_ok=True,
    ).tag(config=True)

    #FIXME: Select the telescope ID from dl1dh reader
    tel_id = Int(
        default_value=1,
        allow_none=False,
        help="Telescope ID to process.",
    ).tag(config=True)

    dl1dh_reader_type = ComponentName(DLDataReader, default_value="DLImageReader").tag(
        config=True
    )

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

    output_path = Path(
        default_value="./output.dl2.h5",
        allow_none=False,
        help="Output path to save the dl2 prediction results",
    ).tag(config=True)

    overwrite = Bool(help="Overwrite the table in the output file if it exists").tag(
        config=True
    )

    aliases = {
        ("i", "input_url"): "MonoPredictionTool.input_url",
        ("t", "type_model"): "MonoPredictionTool.load_type_model_from",
        ("e", "energy_model"): "MonoPredictionTool.load_energy_model_from",
        ("d", "direction_model"): "MonoPredictionTool.load_direction_model_from",
        ("o", "output"): "MonoPredictionTool.output_path",
    }

    flags = {
        **flag(
            "r0-waveforms",
            "HDF5Merger.r0_waveforms",
            "Include r0 waveforms",
            "Exclude r0 waveforms",
        ),
        **flag(
            "r1-waveforms",
            "HDF5Merger.r1_waveforms",
            "Include r1 waveforms",
            "Exclude r1 waveforms",
        ),
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
        "overwrite": (
            {"HDF5Merger": {"overwrite": True}},
            "Overwrite existing files",
        ),
    }

    classes = classes_with_traits(DLDataReader)

    def setup(self):
        # Copy selected tables from the input file to the output file
        self.log.info("Copying to output destination.")
        with HDF5Merger(self.output_path, parent=self) as merger:
            merger(self.input_url)

        # Create a MirroredStrategy.
        self.strategy = tf.distribute.MirroredStrategy()
        atexit.register(self.strategy._extended._collective_ops._lock.locked)  # type: ignore
        self.log.info("Number of devices: %s", self.strategy.num_replicas_in_sync)

        # Set up the data reader
        self.log.info("Loading data reader:")
        self.log.info("  For a large dataset, this may take a while...")
        self.dl1dh_reader = DLDataReader.from_name(
            self.dl1dh_reader_type,
            input_url_signal=[self.input_url],
            parent=self,
        )
        self.log.info(
            "  Number of events loaded: %s", self.dl1dh_reader._get_n_events()
        )
        # Check if the number of events is enough to form a batch
        if self.dl1dh_reader._get_n_events() < self.batch_size:
            raise ToolConfigurationError(
                f"{self.dl1dh_reader._get_n_events()} events are not enough "
                f"to form a batch of size {self.batch_size}. Reduce the batch size."
            )
        # Set the indices for the data loaders
        self.indices = list(range(self.dl1dh_reader._get_n_events()))
        self.last_batch_size = len(self.indices) % self.batch_size

    def start(self):
        self.log.info("Starting the prediction...")
        # Retrieve the IDs from the example_identifiers of the dl1dh for the prediction table
        example_identifiers = self.dl1dh_reader.example_identifiers.copy()
        example_identifiers.keep_columns(TELESCOPE_EVENT_KEYS)
        # Retrieve the IDs from the tel_trigger_table of the dl1dh for the final output table
        tel_trigger_table = self.dl1dh_reader.tel_trigger_table[
            self.dl1dh_reader.tel_trigger_table["tel_id"] == self.tel_id
        ]
        all_identifiers = tel_trigger_table.copy()
        all_identifiers.keep_columns(TELESCOPE_EVENT_KEYS)
        nonexample_identifiers = setdiff(all_identifiers, example_identifiers, keys=TELESCOPE_EVENT_KEYS)
        if len(nonexample_identifiers) > 0:
            nonexample_identifiers.sort(TELESCOPE_EVENT_KEYS)
        # Perform the prediction and fill the prediction table with the prediction results
        # based on the different selected tasks
        if self.load_type_model_from is not None:
            self.log.info(
                "Predicting for the classification of the primary particle type."
            )
            # Predict the data using the loaded type_model
            predict_data = self._predict_with_model(self.load_type_model_from)
            classification_table = example_identifiers.copy()
            classification_table.add_column(predict_data["col1"], name=f"{self.prefix}_tel_prediction")
            # Produce output table with NaNs for missing predictions
            if len(nonexample_identifiers) > 0:
                nan_table = nonexample_identifiers.copy()
                nan_table.add_column(np.nan * np.ones(len(nan_table)), name=f"{self.prefix}_tel_prediction")
                classification_table = vstack([classification_table, nan_table])
            classification_table.sort(TELESCOPE_EVENT_KEYS)
            classification_table.add_column(
                ~np.isnan(prediction, dtype=bool), name=f"{self.prefix}_tel_is_valid"
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
            )
            self.log.info(
                "DL2 prediction data was stored in '%s' under '%s'",
                self.output_path,
                f"{DL2_TELESCOPE_GROUP}/classification/{self.prefix}/tel_{self.tel_id:03d}",
            )

        if self.load_energy_model_from is not None:
            self.log.info(
                "Predicting for the regression of the primary particle energy."
            )
            # Predict the data using the loaded energy_model
            predict_data = self._predict_with_model(self.load_energy_model_from)
            # Convert the reconstructed energy from log10(TeV) to TeV
            reco_energy = u.Quantity(
                np.power(10, np.squeeze(predict_data["energy"])), unit=u.TeV
            )
            # Add the reconstructed energy to the prediction table
            energy_table = example_identifiers.copy()
            energy_table.add_column(reco_energy, name=f"{self.prefix}_tel_energy")
            energy_table.add_column(
                ~np.isnan(reco_energy.data, dtype=bool),
                name=f"{self.prefix}_tel_is_valid",
            )
            # Produce output table with NaNs for missing predictions
            if len(nonexample_identifiers) > 0:
                nan_table = nonexample_identifiers.copy()
                nan_table.add_column(np.nan * np.ones(len(nan_table)), name=f"{self.prefix}_tel_energy")
                nan_table.add_column(np.nan * np.ones(len(nan_table)), name=f"{self.prefix}_tel_is_valid")
                energy_table = vstack([energy_table, nan_table])
            energy_table.sort(TELESCOPE_EVENT_KEYS)
            # Save the prediction to the output file
            write_table(
                energy_table,
                self.output_path,
                f"{DL2_TELESCOPE_GROUP}/energy/{self.prefix}/tel_{self.tel_id:03d}",
            )
            self.log.info(
                "DL2 prediction data was stored in '%s' under '%s'",
                self.output_path,
                f"{DL2_TELESCOPE_GROUP}/energy/{self.prefix}/tel_{self.tel_id:03d}",
            )

        if self.load_direction_model_from is not None:
            self.log.info(
                "Predicting for the regression of the primary particle arrival direction."
            )
            # Predict the data using the loaded direction_model
            predict_data = self._predict_with_model(self.load_direction_model_from)
            # For the direction task, the prediction is the spherical offset (az, alt)
            # from the telescope pointing. The telescope pointing is read from the
            # configuration tree for simulated data and interpolated from the monitoring
            # tree using the trigger timestamps for observational data.
            direction_table = example_identifiers.copy()
            if self.dl1dh_reader.process_type == ProcessType.Simulation:
                # Read the telescope pointing table
                tel_pointing = read_table(
                    self.input_url,
                    f"{FIXED_POINTING_GROUP}/tel_{self.tel_id:03d}",
                )
                # Join the prediction table with the telescope pointing table
                direction_table = join(
                    left=direction_table,
                    right=tel_pointing,
                    keys=["obs_id", "tel_id"],
                )
                # TODO: use keep_order for astropy v7.0.0
                direction_table.sort(TELESCOPE_EVENT_KEYS)
            elif self.dl1dh_reader.process_type == ProcessType.Observation:
                # Initialize the pointing interpolator from ctapipe
                pointing_interpolator = PointingInterpolator(
                    bounds_error=False, extrapolate=True
                )
                # Get the telescope pointing from the dl1dh reader
                tel_pointing = self.dl1dh_reader.telescope_pointings[
                    f"tel_{self.tel_id:03d}"
                ]
                # Add the telescope pointing table to the pointing interpolator
                pointing_interpolator.add_table(self.tel_id, tel_pointing)
                # Join the prediction table with the telescope trigger table from the dl1dh reader
                direction_table = join(
                    left=direction_table,
                    right=tel_trigger_table,
                    keys=TELESCOPE_EVENT_KEYS,
                )
                # TODO: use keep_order for astropy v7.0.0
                direction_table.sort(TELESCOPE_EVENT_KEYS)
                # Interpolate the telescope pointing
                tel_altitude, tel_azimuth = pointing_interpolator(
                    self.tel_id, direction_table["time"]
                )
                # Save the telescope pointing (az, alt) to the prediction table
                direction_table.add_column(
                    tel_azimuth, name="telescope_pointing_azimuth"
                )
                direction_table.add_column(
                    tel_altitude, name="telescope_pointing_altitude"
                )
            # Convert reconstructed spherical offset (az, alt) to SkyCoord
            reco_spherical_offset_az = u.Quantity(
                predict_data["direction"].T[0], unit=u.deg
            )
            reco_spherical_offset_alt = u.Quantity(
                predict_data["direction"].T[1], unit=u.deg
            )
            # Set the telescope pointing of the SkyOffsetSeparation tranformation
            pointing = SkyCoord(
                direction_table["telescope_pointing_azimuth"],
                direction_table["telescope_pointing_altitude"],
                frame="altaz",
            )
            # Calculate the reconstructed direction (az, alt) based on the telescope pointing
            reco_direction = pointing.spherical_offsets_by(
                reco_spherical_offset_az, reco_spherical_offset_alt
            ).to_table()
            # Add the reconstructed direction (az, alt) to the prediction table
            direction_table.add_column(reco_direction["az"], name=f"{self.prefix}_tel_az")
            direction_table.add_column(reco_direction["alt"], name=f"{self.prefix}_tel_alt")
            direction_table.add_column(
                ~np.isnan(reco_direction["az"].data, dtype=bool),
                name=f"{self.prefix}_tel_is_valid",
            )
            # Remove the telescope pointing columns and trigger time from the prediction table
            if not self.store_event_wise_pointing:
                direction_table.keep_columns(["obs_id", "event_id", "tel_id", "az", "alt"])
            # Produce output table with NaNs for missing predictions
            if len(nonexample_identifiers) > 0:
                nan_table = nonexample_identifiers.copy()
                nan_table.add_column(np.nan * np.ones(len(nan_table)), name=f"{self.prefix}_tel_az")
                nan_table.add_column(np.nan * np.ones(len(nan_table)), name=f"{self.prefix}_tel_alt")
                nan_table.add_column(np.nan * np.ones(len(nan_table)), name=f"{self.prefix}_tel_is_valid")
                if self.store_event_wise_pointing:
                    nan_table = join(
                        left=nan_table,
                        right=tel_trigger_table,
                        keys=TELESCOPE_EVENT_KEYS,
                    )
                    # TODO: use keep_order for astropy v7.0.0
                    nan_table.sort(TELESCOPE_EVENT_KEYS)
                    nan_table.remove_column("n_trigger_pixels")
                    nan_tel_altitude, nan_tel_azimuth = pointing_interpolator(
                        self.tel_id, nan_table["time"]
                    )
                    nan_table.add_column(nan_tel_azimuth, name="telescope_pointing_azimuth")
                    nan_table.add_column(nan_tel_altitude, name="telescope_pointing_altitude")
                direction_table = vstack([direction_table, nan_table])
            direction_table.sort(TELESCOPE_EVENT_KEYS)
            self.log.info("Saving the prediction to the output file.")
            write_table(
                direction_table,
                self.output_path,
                f"{DL2_TELESCOPE_GROUP}/geometry/{self.prefix}/tel_{self.tel_id:03d}",
            )
            self.log.info(
                "DL2 prediction data was stored in '%s' under '%s'",
                self.output_path,
                f"{DL2_TELESCOPE_GROUP}/geometry/{self.prefix}/tel_{self.tel_id:03d}",
            )

    def finish(self):
        self.log.info("Tool is shutting down")

    def _predict_with_model(self, model_path):
        """
        Load and predict with a CTLearn model.

        Load a model from the specified path and predict the data using the loaded model.
        If a last batch loader is provided, predict the last batch and stack the results.

        Parameters
        ----------
        model_path : str
            Path to a Keras model file (Keras3) or directory (Keras2).

        Returns
        -------
        predict_data : astropy.table.Table
            Table containing the prediction results.
        """
        # Create a new DLDataLoader for each task
        # It turned out to be more robust to initialize the DLDataLoader separately.
        dl1dh_loader = DLDataLoader(
            self.dl1dh_reader,
            self.indices,
            tasks=[],
            batch_size=self.batch_size * self.strategy.num_replicas_in_sync,
        )
        # Keras is only considering the last complete batch.
        # In prediction mode we don't want to loose the last
        # uncomplete batch, so we are creating an additional
        # batch generator for the remaining events.
        dl1dh_loader_last_batch = None
        if self.last_batch_size > 0:
            last_batch_indices = self.indices[-self.last_batch_size:]
            dl1dh_loader_last_batch = DLDataLoader(
                self.dl1dh_reader,
                last_batch_indices,
                tasks=[],
                batch_size=self.last_batch_size,
            )
        # Load the model from the specified path
        model = keras.saving.load_model(model_path)
        # Predict the data using the loaded model
        predict_data = Table(model.predict(dl1dh_loader))
        # Predict the last batch and stack the results to the prediction data
        if dl1dh_loader_last_batch is not None:
            predict_data = vstack(
                [predict_data, Table(model.predict(dl1dh_loader_last_batch))]
            )
        return predict_data


def main():
    # Run the tool
    tool = MonoPredictionTool()
    tool.run()


if __name__ == "main":
    main()
