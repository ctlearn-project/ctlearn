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

    overwrite = Bool(help="Overwrite the table in the output file if it exists").tag(config=True)

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
        self.dl1dh_reader = DLDataReader.from_name(
            self.dl1dh_reader_type,
            input_url_signal=[self.input_url],
            parent=self,
        )
        # Set up the data loaders for prediction
        indices = list(range(self.dl1dh_reader._get_n_events()))
        self.dl1dh_loader = DLDataLoader(
            self.dl1dh_reader,
            indices,
            tasks=self.reco_tasks,
            batch_size=self.batch_size * self.strategy.num_replicas_in_sync,
        )
        # Keras is only considering the last complete batch.
        # In prediction mode we don't want to loose the last
        # uncomplete batch, so we are creating an additional
        # batch generator for the remaining events.
        self.dl1dh_loader_last_batch = None
        last_batch_size = len(indices) % self.batch_size
        if last_batch_size > 0:
            last_batch_indices = indices[-last_batch_size:]
            self.dl1dh_loader_last_batch = DLDataLoader(
                self.dl1dh_reader,
                last_batch_indices,
                tasks=self.reco_tasks,
                batch_size=last_batch_size,
            )

    def start(self):
        self.log.info("Starting the prediction...")
        # Retrieve the IDs from the example_identifiers of the dl1dh for the prediction table
        prediction_table = self.dl1dh_reader.example_identifiers
        prediction_table.keep_columns(["obs_id", "event_id", "tel_id"])
        # Perform the prediction and fill the prediction table with the prediction results
        # based on the different selected tasks
        if self.load_type_model_from is not None:
            self.log.info("Predicting for the classification of the primary particle type.")
            # Predict the data using the loaded type_model
            predict_data = self._predict_with_model(
                self.load_type_model_from,
                self.dl1dh_loader,
                self.dl1dh_loader_last_batch
            )
            prediction_table.add_column(predict_data["col1"], name="prediction")
            # Save the prediction to the output file
            write_table(
                prediction_table,
                self.output_path,
                f"/dl2/event/telescope/classification/{self.reco_algo}/tel_{self.tel_id:03d}",
            )
            prediction_table.keep_columns(["obs_id", "event_id", "tel_id"])

        if self.load_energy_model_from is not None:
            self.log.info("Predicting for the regression of the primary particle energy.")
            # Predict the data using the loaded energy_model
            predict_data = self._predict_with_model(
                self.load_energy_model_from,
                self.dl1dh_loader,
                self.dl1dh_loader_last_batch
            )
            # Convert the reconstructed energy from log10(TeV) to TeV
            reco_energy = u.Quantity(
                np.power(10, np.squeeze(predict_data["energy"])), unit=u.TeV
            )
            # Add the reconstructed energy to the prediction table
            prediction_table.add_column(reco_energy, name="energy")
            # Save the prediction to the output file
            write_table(
                prediction_table,
                self.output_path,
                f"/dl2/event/telescope/energy/{self.reco_algo}/tel_{self.tel_id:03d}",
            )
            prediction_table.keep_columns(["obs_id", "event_id", "tel_id"])
        
        if self.load_direction_model_from is not None:
            self.log.info("Predicting for the regression of the primary particle arrival direction.")
            # Predict the data using the loaded direction_model
            predict_data = self._predict_with_model(
                self.load_direction_model_from,
                self.dl1dh_loader,
                self.dl1dh_loader_last_batch
            )
            # For the direction task, the prediction is the spherical offset (az, alt)
            # from the telescope pointing. The telescope pointing is read from the
            # configuration tree for simulated data and interpolated from the monitoring
            # tree using the trigger timestamps for observational data.
            if self.dl1dh_reader.process_type == ProcessType.Simulation:
                # Read the telescope pointing table
                tel_pointing = read_table(
                    self.input_url,
                    f"/configuration/telescope/pointing/tel_{self.tel_id:03d}",
                )
                # Join the prediction table with the telescope pointing table
                prediction_table = join(
                    left=prediction_table,
                    right=tel_pointing,
                    keys=["obs_id", "tel_id"],
                )
                prediction_table.sort(["obs_id", "event_id", "tel_id"])
            elif self.dl1dh_reader.process_type == ProcessType.Observation:
                # Initialize the pointing interpolator from ctapipe
                pointing_interpolator = PointingInterpolator(bounds_error=False, extrapolate=True)
                # Get the telescope pointing from the dl1dh reader
                tel_pointing = self.dl1dh_reader.telescope_pointings[f"tel_{self.tel_id:03d}"]
                # Join the prediction table with the telescope trigger table from the dl1dh reader
                prediction_table = join(
                    left=prediction_table,
                    right=self.dl1dh_reader.tel_trigger_table,
                    keys=["obs_id", "event_id", "tel_id"],
                )
                prediction_table.sort(["obs_id", "event_id", "tel_id"])
                # Interpolate the telescope pointing
                tel_altitude, tel_azimuth = pointing_interpolator(self.tel_id, prediction_table['time'])
                # Save the telescope pointing (az, alt) to the prediction table
                prediction_table.add_column(tel_azimuth, name="telescope_pointing_azimuth")
                prediction_table.add_column(tel_altitude, name="telescope_pointing_altitude")
            # Convert reconstructed spherical offset (az, alt) to SkyCoord
            reco_spherical_offset_az = u.Quantity(predict_data["direction"].T[0], unit=u.deg)
            reco_spherical_offset_alt = u.Quantity(predict_data["direction"].T[1], unit=u.deg)
            # Set the telescope pointing of the SkyOffsetSeparation tranformation
            pointing = SkyCoord(
                prediction_table["telescope_pointing_azimuth"],
                prediction_table["telescope_pointing_altitude"],
                frame="altaz",
            )
            # Calculate the reconstructed direction (az, alt) based on the telescope pointing
            reco_direction = pointing.spherical_offsets_by(reco_spherical_offset_az, reco_spherical_offset_alt).to_table()
            # Add the reconstructed direction (az, alt) to the prediction table
            prediction_table.add_column(reco_direction["az"], name="az")
            prediction_table.add_column(reco_direction["alt"], name="alt")
            # Remove the telescope pointing columns and trigger time from the prediction table
            prediction_table.keep_columns(["obs_id", "event_id", "tel_id", "az", "alt"])
            # Save the prediction to the output file
            write_table(
                prediction_table,
                self.output_path,
                f"/dl2/event/telescope/geometry/{self.reco_algo}/tel_{self.tel_id:03d}",
            )


    def finish(self):
        self.log.info("Tool is shutting down")


    def _predict_with_model(model_path, data_loader, last_batch_loader=None):
        """
        Load and predict with a CTLearn model.

        Load a model from the specified path and predict the data using the loaded model.
        If a last batch loader is provided, predict the last batch and stack the results.

        Parameters
        ----------
        model_path : str
            Path to a Keras model file (Keras3) or directory (Keras2).
        data_loader : DLDataLoader
            Data loader for the main batches.
        last_batch_loader : DLDataLoader, optional
            Data loader for the last batch.

        Returns
        -------
        predict_data : astropy.table.Table
            Table containing the prediction results.
        """
        # Load the model from the specified path
        model = keras.saving.load_model(model_path)
        # Predict the data using the loaded model
        predict_data = Table(model.predict(data_loader))
        # Predict the last batch and stack the results to the prediction data
        if last_batch_loader is not None:
            predict_data = vstack([predict_data, Table(model.predict(last_batch_loader))])
        return predict_data


def main():
    # Run the tool
    tool = MonoPredictionTool()
    tool.run()


if __name__ == "main":
    main()
