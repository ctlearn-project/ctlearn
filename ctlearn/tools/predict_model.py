"""
Tools to predict the gammaness, energy and arrival direction in monoscopic and stereoscopic mode using ``CTLearnModel`` on R1/DL1 data using the ``DLDataReader`` and ``DLDataLoader``.
"""

import atexit
import pathlib
import numpy as np
import os
import tensorflow as tf
import keras

from astropy import units as u
from astropy.coordinates.earth import EarthLocation
from astropy.coordinates import AltAz, SkyCoord
from astropy.table import (
    Table,
    hstack,
    vstack,
    join,
    setdiff,
)

from ctapipe.containers import (
    ParticleClassificationContainer,
    ReconstructedGeometryContainer,
    ReconstructedEnergyContainer,
)
from ctapipe.coordinates import CameraFrame, NominalFrame
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
from ctapipe.reco.reconstructor import ReconstructionProperty
from ctapipe.reco.stereo_combination import StereoCombiner
from ctapipe.reco.utils import add_defaults_and_meta
from dl1_data_handler.reader import (
    DLDataReader,
    ProcessType,
    LST_EPOCH,
)
from ctlearn.core.loader import DLDataLoader

SIMULATION_CONFIG_TABLE = "/configuration/simulation/run"
FIXED_POINTING_GROUP = "/configuration/telescope/pointing"
POINTING_GROUP = "/dl1/monitoring/telescope/pointing"
SUBARRAY_POINTING_GROUP = "/dl1/monitoring/subarray/pointing"
DL1_TELESCOPE_GROUP = "/dl1/event/telescope"
DL1_SUBARRAY_GROUP = "/dl1/event/subarray"
DL2_SUBARRAY_GROUP = "/dl2/event/subarray"
DL2_TELESCOPE_GROUP = "/dl2/event/telescope"
SUBARRAY_EVENT_KEYS = ["obs_id", "event_id"]
TELESCOPE_EVENT_KEYS = ["obs_id", "event_id", "tel_id"]

__all__ = [
    "PredictCTLearnModel",
    "MonoPredictCTLearnModel",
    "StereoPredictCTLearnModel",
]


class PredictCTLearnModel(Tool):
    """
    Base tool to predict the gammaness, energy and arrival direction from R1/DL1 data using CTLearn models.

    This class handles the prediction of the gammaness, energy and arrival direction from pixel-wise image
    or waveform data. It also supports the extraction of the feature vectors from the backbone submodel to
    store them in the output file. The input data is loaded from the input url using the
    ``~dl1_data_handler.reader.DLDataReader`` and ``~ctlearn.core.loader.DLDataLoader``.
    The prediction is performed using the CTLearn models. The data is stored in the output file
    following the ctapipe DL2 data format. The ``start`` method is implemented in the subclasses to
    handle the prediction for mono and stereo mode.

    Attributes
    ----------
    input_url : pathlib.Path
        Input ctapipe HDF5 files including pixel-wise image or waveform data.
    use_HDF5Merger : bool
        Set whether to use the HDF5Merger component to copy the selected tables from the input file to the output file.
    dl1_features : bool
        Set whether to include the dl1 feature vectors in the output file.
    dl2_telescope : bool
        Set whether to include dl2 telescope-event-wise data in the output file.
    dl2_subarray : bool
        Set whether to include dl2 subarray-event-wise data in the output file.
    dl1dh_reader : dl1_data_handler.reader.DLDataReader
        DLDataReader object to read the data.
    dl1dh_reader_type : str
        Type of the DLDataReader to use for the prediction.
    stack_telescope_images : bool
        Set whether to stack the telescope images in the data loader. Requires ``stereo``.
    sort_by_intensity : bool
        Set whether to sort the telescope images by intensity in the data loader. Requires ``stereo``.
    prefix : str
        Name of the reconstruction algorithm used to generate the dl2 data.
    load_type_model_from : pathlib.Path
        Path to a Keras model file (Keras3) or directory (Keras2) for the classification of the primary particle type.
    load_energy_model_from : pathlib.Path
        Path to a Keras model file (Keras3) or directory (Keras2) for the regression of the primary particle energy.
    load_cameradirection_model_from : pathlib.Path
        Path to a Keras model file (Keras3) or directory (Keras2) for the regression
        of the primary particle arrival direction based on camera coordinate offsets.
    load_cameradirection_model_from : pathlib.Path
        Path to a Keras model file (Keras3) or directory (Keras2) for the regression
        of the primary particle arrival direction based on spherical coordinate offsets.
    output_path : pathlib.Path
        Output path to save the dl2 prediction results.
    overwrite_tables : bool
        Overwrite the table in the output file if it exists.
    keras_verbose : int
        Verbosity mode of Keras during the prediction.
    strategy : tf.distribute.Strategy
        MirroredStrategy to distribute the prediction.
    data_loader : ctlearn.core.loader.DLDataLoader
        DLDataLoader object to load the data.
    indices : list of int
        List of indices for the data loaders.
    batch_size : int
        Size of the batch to perform inference of the neural network.
    last_batch_size : int
        Size of the last batch in the data loaders.

    Methods
    -------
    setup()
        Set up the tool.
    finish()
        Finish the tool.
    _predict_with_model(model_path)
        Load and predict with a CTLearn model.
    _predict_classification(example_identifiers)
        Predict the classification of the primary particle type.
    _predict_energy(example_identifiers)
        Predict the energy of the primary particle.
    _predict_cameradirection(example_identifiers)
        Predict the arrival direction of the primary particle based on camera coordinate offsets.
    _predict_skydirection(example_identifiers)
        Predict the arrival direction of the primary particle based on spherical coordinate offsets.
    _transform_cam_coord_offsets_to_sky(table)
        Transform to camera coordinate offsets w.r.t. the telescope pointing to Alt/Az coordinates.
    _transform_spher_coord_offsets_to_sky(table)
        Transform to spherical coordinate offsets w.r.t. the telescope pointing to Alt/Az coordinates.
    _create_nan_table(nonexample_identifiers, columns, shapes)
        Create a table with NaNs for missing predictions.
    _store_pointing(all_identifiers)
        Store the telescope pointing table from to the output file.
    _create_feature_vectors_table(example_identifiers, nonexample_identifiers, classification_feature_vectors, energy_feature_vectors, direction_feature_vectors)
        Create the table for the DL1 feature vectors.
    """

    input_url = Path(
        help="Input ctapipe HDF5 files including pixel-wise image or waveform data",
        allow_none=True,
        exists=True,
        directory_ok=False,
        file_ok=True,
    ).tag(config=True)

    use_HDF5Merger = Bool(
        default_value=True,
        allow_none=False,
        help=(
            "Set whether to use the HDF5Merger component to copy the selected tables "
            "from the input file to the output file. CAUTION: This can only be used "
            "if the output file not exists."
        ),
    ).tag(config=True)

    dl1_features = Bool(
        default_value=False,
        allow_none=False,
        help="Set whether to include the dl1 feature vectors in the output file.",
    ).tag(config=True)

    dl2_telescope = Bool(
        default_value=True,
        allow_none=False,
        help="Set whether to include dl2 telescope-event-wise data in the output file.",
    ).tag(config=True)

    dl2_subarray = Bool(
        default_value=True,
        allow_none=False,
        help="Set whether to include dl2 subarray-event-wise data in the output file.",
    ).tag(config=True)

    dl1dh_reader_type = ComponentName(DLDataReader, default_value="DLImageReader").tag(
        config=True
    )

    stack_telescope_images = Bool(
        default_value=False,
        allow_none=False,
        help=(
            "Set whether to stack the telescope images in the data loader. "
            "Requires DLDataReader mode to be ``stereo``."
        ),
    ).tag(config=True)

    sort_by_intensity = Bool(
        default_value=False,
        allow_none=False,
        help=(
            "Set whether to sort the telescope images by intensity in the data loader. "
            "Requires DLDataReader mode to be ``stereo``."
        ),
    ).tag(config=True)

    prefix = Unicode(
        default_value="CTLearn",
        allow_none=False,
        help="Name of the reconstruction algorithm used to generate the dl2 data.",
    ).tag(config=True)

    load_type_model_from = Path(
        default_value=None,
        help=(
            "Path to a Keras model file (Keras3) or directory (Keras2) for the classification "
            "of the primary particle type."
        ),
        allow_none=True,
        exists=True,
        directory_ok=True,
        file_ok=True,
    ).tag(config=True)

    load_energy_model_from = Path(
        default_value=None,
        help=(
            "Path to a Keras model file (Keras3) or directory (Keras2) for the regression "
            "of the primary particle energy."
        ),
        allow_none=True,
        exists=True,
        directory_ok=True,
        file_ok=True,
    ).tag(config=True)

    load_cameradirection_model_from = Path(
        default_value=None,
        help=(
            "Path to a Keras model file (Keras3) or directory (Keras2) for the regression "
            "of the primary particle arrival direction based on camera coordinate offsets."
        ),
        allow_none=True,
        exists=True,
        directory_ok=True,
        file_ok=True,
    ).tag(config=True)

    load_skydirection_model_from = Path(
        default_value=None,
        help=(
            "Path to a Keras model file (Keras3) or directory (Keras2) for the regression "
            "of the primary particle arrival direction based on spherical coordinate offsets."
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

    overwrite_tables = Bool(
        default_value=True,
        allow_none=False,
        help="Overwrite the table in the output file if it exists",
    ).tag(config=True)

    keras_verbose = Int(
        default_value=1,
        min=0,
        max=2,
        allow_none=False,
        help=(
            "Verbosity mode of Keras during the prediction: "
            "0 = silent, 1 = progress bar, 2 = one line per call."
        ),
    ).tag(config=True)

    aliases = {
        ("i", "input_url"): "PredictCTLearnModel.input_url",
        ("t", "type_model"): "PredictCTLearnModel.load_type_model_from",
        ("e", "energy_model"): "PredictCTLearnModel.load_energy_model_from",
        (
            "d",
            "cameradirection_model",
        ): "PredictCTLearnModel.load_cameradirection_model_from",
        ("s", "skydirection_model"): "PredictCTLearnModel.load_skydirection_model_from",
        ("o", "output"): "PredictCTLearnModel.output_path",
    }

    flags = {
        **flag(
            "dl1-features",
            "PredictCTLearnModel.dl1_features",
            "Include dl1 features",
            "Exclude dl1 features",
        ),
        **flag(
            "dl2-telescope",
            "PredictCTLearnModel.dl2_telescope",
            "Include dl2 telescope-event-wise data in the output file",
            "Exclude dl2 telescope-event-wise data in the output file",
        ),
        **flag(
            "dl2-subarray",
            "PredictCTLearnModel.dl2_subarray",
            "Include dl2 telescope-event-wise data in the output file",
            "Exclude dl2 telescope-event-wise data in the output file",
        ),
        **flag(
            "use-HDF5Merger",
            "PredictCTLearnModel.use_HDF5Merger",
            "Copy data using the HDF5Merger component (CAUTION: This can not be used if the output file already exists)",
            "Do not copy data using the HDF5Merger component",
        ),
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
    }

    classes = classes_with_traits(DLDataReader)

    def setup(self):
        # Check if the ctapipe HDF5Merger component is enabled
        if self.use_HDF5Merger:
            if os.path.exists(self.output_path):
                raise ToolConfigurationError(
                    f"The output file '{self.output_path}' already exists. Please use "
                    "'--no-use-HDF5Merger' to disable the usage of the HDF5Merger component."
                )
            # Copy selected tables from the input file to the output file
            self.log.info("Copying to output destination.")
            with HDF5Merger(self.output_path, parent=self) as merger:
                merger(self.input_url)
        else:
            self.log.info(
                "No copy to output destination, since the usage of the HDF5Merger component is disabled."
            )

        # Create a MirroredStrategy.
        self.strategy = tf.distribute.MirroredStrategy()
        atexit.register(self.strategy._extended._collective_ops._lock.locked)  # type: ignore
        self.log.info("Number of devices: %s", self.strategy.num_replicas_in_sync)

        # Set up the data reader
        self.log.info("Loading data reader:")
        self.log.info("For a large dataset, this may take a while...")
        self.dl1dh_reader = DLDataReader.from_name(
            self.dl1dh_reader_type,
            input_url_signal=[self.input_url],
            parent=self,
        )
        self.log.info("Number of events loaded: %s", self.dl1dh_reader._get_n_events())
        # Check if the number of events is enough to form a batch
        if self.dl1dh_reader._get_n_events() < self.batch_size:
            raise ToolConfigurationError(
                f"{self.dl1dh_reader._get_n_events()} events are not enough "
                f"to form a batch of size {self.batch_size}. Reduce the batch size."
            )
        # Set the indices for the data loaders
        self.indices = list(range(self.dl1dh_reader._get_n_events()))
        self.last_batch_size = len(self.indices) % (
            self.batch_size * self.strategy.num_replicas_in_sync
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
        feature_vectors : np.ndarray
            Feature vectors extracted from the backbone model.
        """
        # Create a new DLDataLoader for each task
        # It turned out to be more robust to initialize the DLDataLoader separately.
        data_loader = DLDataLoader(
            self.dl1dh_reader,
            self.indices,
            tasks=[],
            batch_size=self.batch_size * self.strategy.num_replicas_in_sync,
            sort_by_intensity=self.sort_by_intensity,
            stack_telescope_images=self.stack_telescope_images,
        )
        # Keras is only considering the last complete batch.
        # In prediction mode we don't want to loose the last
        # uncomplete batch, so we are creating an additional
        # batch generator for the remaining events.
        data_loader_last_batch = None
        if self.last_batch_size > 0:
            last_batch_indices = self.indices[-self.last_batch_size :]
            data_loader_last_batch = DLDataLoader(
                self.dl1dh_reader,
                last_batch_indices,
                tasks=[],
                batch_size=self.last_batch_size,
                sort_by_intensity=self.sort_by_intensity,
                stack_telescope_images=self.stack_telescope_images,
            )
        # Load the model from the specified path
        model = keras.saving.load_model(model_path)
        prediction_colname = (
            model.layers[-1].name if model.layers[-1].name != "softmax" else "type"
        )
        backbone_model, feature_vectors = None, None
        if self.dl1_features:
            # Get the backbone model which is the second layer of the model
            backbone_model = model.get_layer(index=1)
            # Create a new head model with the same layers as the original model.
            # The output of the backbone model is the input of the head model.
            backbone_output_shape = keras.Input(model.layers[2].input_shape[1:])
            x = backbone_output_shape
            for layer in model.layers[2:]:
                x = layer(x)
            head = keras.Model(inputs=backbone_output_shape, outputs=x)
            # Apply the backbone model with the data loader to retrieve the feature vectors
            feature_vectors = backbone_model.predict(
                data_loader, verbose=self.keras_verbose
            )
            # Apply the head model with the feature vectors to retrieve the prediction
            predict_data = Table(
                {
                    prediction_colname: head.predict(
                        feature_vectors, verbose=self.keras_verbose
                    )
                }
            )
            # Predict the last batch and stack the results to the prediction data
            if data_loader_last_batch is not None:
                feature_vectors_last_batch = backbone_model.predict(
                    data_loader_last_batch, verbose=self.keras_verbose
                )
                feature_vectors = np.concatenate(
                    (feature_vectors, feature_vectors_last_batch)
                )
                predict_data = vstack(
                    [
                        predict_data,
                        Table(
                            {
                                prediction_colname: head.predict(
                                    feature_vectors_last_batch,
                                    verbose=self.keras_verbose,
                                )
                            }
                        ),
                    ]
                )
        else:
            # Predict the data using the loaded model
            predict_data = model.predict(data_loader, verbose=self.keras_verbose)
            # Create a astropy table with the prediction results
            # The classification task has a softmax layer as the last layer
            # which returns the probabilities for each class in an array, while
            # the regression tasks have output neurons which returns the
            # predicted value for the task in a dictionary.
            if prediction_colname == "type":
                predict_data = Table({prediction_colname: predict_data})
            else:
                predict_data = Table(predict_data)
            # Predict the last batch and stack the results to the prediction data
            if data_loader_last_batch is not None:
                predict_data_last_batch = model.predict(
                    data_loader_last_batch, verbose=self.keras_verbose
                )
                if model.layers[-1].name == "type":
                    predict_data_last_batch = Table(
                        {prediction_colname: predict_data_last_batch}
                    )
                else:
                    predict_data_last_batch = Table(predict_data_last_batch)
                predict_data = vstack([predict_data, predict_data_last_batch])
        return predict_data, feature_vectors

    def _predict_classification(self, example_identifiers):
        """
        Predict the classification of the primary particle type.

        This method uses a pre-trained type model to predict the type of the primary particle
        for a given set of example identifiers. The predicted classification score ('gammaness')
        is added to the example identifiers table.

        Parameters:
        -----------
        classification_table : astropy.table.Table
            Table containing the example identifiers with an additional column for the
            predicted classification score ('gammaness').
        feature_vectors : np.ndarray
            Feature vectors extracted from the backbone model.
        """
        self.log.info(
            "Predicting for the classification of the primary particle type..."
        )
        # Predict the data using the loaded type_model
        predict_data, feature_vectors = self._predict_with_model(
            self.load_type_model_from
        )
        # Create prediction table and add the predicted classification score ('gammaness')
        classification_table = example_identifiers.copy()
        classification_table.add_column(
            predict_data["type"].T[0], name=f"{self.prefix}_tel_prediction"
        )
        return classification_table, feature_vectors

    def _predict_energy(self, example_identifiers):
        """
        Predict the energy of the primary particle.

        This method uses a pre-trained energy model to predict the energy of the primary particle
        for a given set of example identifiers. The predicted energy is then converted from
        log10(TeV) to TeV and added to the example identifiers table.

        Parameters:
        -----------
        energy_table : astropy.table.Table
            Table containing the example identifiers with an additional column for the
            reconstructed energy in TeV.
        feature_vectors : np.ndarray
            Feature vectors extracted from the backbone model.
        """
        self.log.info("Predicting for the regression of the primary particle energy...")
        # Predict the data using the loaded energy_model
        predict_data, feature_vectors = self._predict_with_model(
            self.load_energy_model_from
        )
        # Convert the reconstructed energy from log10(TeV) to TeV
        reco_energy = u.Quantity(
            np.power(10, np.squeeze(predict_data["energy"])),
            unit=u.TeV,
        )
        # Create prediction table and add the reconstructed energy in TeV
        energy_table = example_identifiers.copy()
        energy_table.add_column(reco_energy, name=f"{self.prefix}_tel_energy")
        return energy_table, feature_vectors

    def _predict_cameradirection(self, example_identifiers):
        """
        Predict the arrival direction of the primary particle based on camera coordinate offsets.

        This method uses a pre-trained direction model to predict the arrival direction of the
        primary particle for a given set of example identifiers. The predicted camera coordinate offsets
        is added to the example identifiers table.

        Parameters:
        -----------
        example_identifiers : astropy.table.Table
            Table containing the example identifiers.

        Returns:
        --------
        cameradirection_table : astropy.table.Table
            Table containing the example identifiers with an additional column for the
            reconstructed camera coordinate offsets in x and y.
        feature_vectors : np.ndarray
            Feature vectors extracted from the backbone model.
        """
        self.log.info(
            "Predicting for the regression of the primary particle arrival direction based on camera coordinate offsets..."
        )
        # Predict the data using the loaded direction_model
        predict_data, feature_vectors = self._predict_with_model(
            self.load_cameradirection_model_from
        )
        # For the direction task, the prediction is the camera coordinate offset in x and y
        # from the telescope pointing.
        cam_coord_offset_x = u.Quantity(predict_data["cameradirection"].T[0], unit=u.m)
        cam_coord_offset_y = u.Quantity(predict_data["cameradirection"].T[1], unit=u.m)
        # Create prediction table and add the reconstructed energy in TeV
        cameradirection_table = example_identifiers.copy()
        cameradirection_table.add_column(cam_coord_offset_x, name="cam_coord_offset_x")
        cameradirection_table.add_column(cam_coord_offset_y, name="cam_coord_offset_y")
        return cameradirection_table, feature_vectors

    def _predict_skydirection(self, example_identifiers):
        """
        Predict the arrival direction of the primary particle based on spherical coordinate offsets.

        This method uses a pre-trained direction model to predict the arrival direction of the primary
        particle for a given set of example identifiers. The predicted spherical coordinate offsets is
        added to the example identifiers table.

        Parameters:
        -----------
        example_identifiers : astropy.table.Table
            Table containing the example identifiers.

        Returns:
        --------
        skydirection_table : astropy.table.Table
            Table containing the example identifiers with an additional column for the
            reconstructed spherical coordinate offsets in fov_lon and fov_lat.
        feature_vectors : np.ndarray
            Feature vectors extracted from the backbone model.
        """
        self.log.info(
            "Predicting for the regression of the primary particle arrival direction based on spherical coordinate offsets..."
        )
        # Predict the data using the loaded direction_model
        predict_data, feature_vectors = self._predict_with_model(
            self.load_skydirection_model_from
        )
        # For the direction task, the prediction is the spherical offset in fov_lon and fov_lat
        # from the telescope pointing.
        fov_lon = u.Quantity(predict_data["skydirection"].T[0], unit=u.deg)
        fov_lat = u.Quantity(predict_data["skydirection"].T[1], unit=u.deg)
        # Create prediction table and add the reconstructed energy in TeV
        skydirection_table = example_identifiers.copy()
        skydirection_table.add_column(fov_lon, name="fov_lon")
        skydirection_table.add_column(fov_lat, name="fov_lat")
        return skydirection_table, feature_vectors

    def _transform_cam_coord_offsets_to_sky(self, table) -> Table:
        """
        Transform the predicted camera coordinate offsets w.r.t. the telescope pointing to Alt/Az coordinates.

        This method converts the predicted camera coordinate offsets w.r.t. the telescope pointing
        in the provided table to Alt/Az coordinates. It also removes the unnecessary columns
        from the table that do not the ctapipe DL2 data format.

        Parameters:
        -----------
        table : astropy.table.Table
            A Table containing the trigger time, telescope pointing, and predicted camera coordinate offsets.

        Returns:
        --------
        table : astropy.table.Table
            A Table with the Alt/Az coordinates following the ctapipe DL2 data format.
        """
        # Get the telescope ID from the table
        tel_id = table["tel_id"][0]
        # Set the telescope position
        tel_ground_frame = self.dl1dh_reader.subarray.tel_coords[
            self.dl1dh_reader.subarray.tel_ids_to_indices(tel_id)
        ]
        # Set the trigger timestamp based on the process type
        if self.dl1dh_reader.process_type == ProcessType.Simulation:
            trigger_time = LST_EPOCH
        elif self.dl1dh_reader.process_type == ProcessType.Observation:
            trigger_time = table["time"]
        # Set the telescope pointing with the trigger timestamp and the telescope position
        altaz = AltAz(
            location=tel_ground_frame.to_earth_location(),
            obstime=trigger_time,
        )
        # Set the telescope pointing
        tel_pointing = SkyCoord(
            az=table["pointing_azimuth"],
            alt=table["pointing_altitude"],
            frame=altaz,
        )
        # Set the camera frame with the focal length and rotation of the camera
        camera_frame = CameraFrame(
            focal_length=self.dl1dh_reader.subarray.tel[
                tel_id
            ].optics.equivalent_focal_length,
            rotation=self.dl1dh_reader.pix_rotation[tel_id],
            telescope_pointing=tel_pointing,
        )
        # Set the camera coordinate offset
        cam_coord_offset = SkyCoord(
            x=table["cam_coord_offset_x"],
            y=table["cam_coord_offset_y"],
            frame=camera_frame,
        )
        # tel_identifiers = tel_identifiers[tel_identifiers["tel_id"] == tel_id]
        # Transform the true Alt/Az coordinates to camera coordinates
        reco_direction = cam_coord_offset.transform_to(altaz)
        # Add the reconstructed direction (az, alt) to the prediction table
        table.add_column(reco_direction.az.to(u.deg), name=f"{self.prefix}_tel_az")
        table.add_column(reco_direction.alt.to(u.deg), name=f"{self.prefix}_tel_alt")
        # Remove unnecessary columns from the table that do not the ctapipe DL2 data format
        table.remove_columns(
            [
                "time",
                "pointing_azimuth",
                "pointing_altitude",
                "cam_coord_offset_x",
                "cam_coord_offset_y",
            ]
        )
        return table

    def _transform_spher_coord_offsets_to_sky(self, table) -> Table:
        """
        Transform the predicted spherical offsets w.r.t. the telescope pointing to Alt/Az coordinates.

        This method converts the predicted spherical offsets w.r.t. the telescope pointing
        in the provided table to Alt/Az coordinates. It also removes the unnecessary columns
        from the table that do not the ctapipe DL2 data format.

        Parameters:
        -----------
        table : astropy.table.Table
            A Table containing the trigger time, telescope pointing, and predicted spherical offsets.

        Returns:
        --------
        table : astropy.table.Table
            A Table with the Alt/Az coordinates following the ctapipe DL2 data format.
        """

        # Set the trigger timestamp based on the process type
        if self.dl1dh_reader.process_type == ProcessType.Simulation:
            trigger_time = LST_EPOCH
        elif self.dl1dh_reader.process_type == ProcessType.Observation:
            trigger_time = table["time"]
        # Set the AltAz frame with the reference location and time
        altaz = AltAz(
            location=self.dl1dh_reader.subarray.reference_location,
            obstime=trigger_time,
        )
        # Set the array pointing
        array_pointing = SkyCoord(
            az=table["pointing_azimuth"],
            alt=table["pointing_altitude"],
            frame=altaz,
        )
        # Set the nominal frame with the array pointing
        nom_frame = NominalFrame(
            origin=array_pointing,
            location=self.dl1dh_reader.subarray.reference_location,
            obstime=trigger_time,
        )
        # Set the reco direction in (alt, az) coordinates
        reco_direction = SkyCoord(
            fov_lon=table["fov_lon"],
            fov_lat=table["fov_lat"],
            frame=nom_frame,
        )
        # Transform the reco direction from nominal frame to the AltAz frame
        sky_coord = reco_direction.transform_to(altaz)
        # Add the reconstructed direction (az, alt) to the prediction table
        table.add_column(reco_direction.az.to(u.deg), name=f"{self.prefix}_az")
        table.add_column(reco_direction.alt.to(u.deg), name=f"{self.prefix}_alt")
        # Remove unnecessary columns from the table that do not the ctapipe DL2 data format
        table.remove_columns(
            [
                "time",
                "pointing_azimuth",
                "pointing_altitude",
                "fov_lon",
                "fov_lat",
            ]
        )
        return table

    def _create_nan_table(self, nonexample_identifiers, columns, shapes):
        """
        Create a table with NaNs for missing predictions.

        This method creates a table with NaNs for missing predictions for the non-example identifiers.
        In stereo mode, the table also a column for the valid telescopes is added with all False values.

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
        # Add that no telescope is valid for the non-example identifiers in stereo mode
        if self.dl1dh_reader.mode == "stereo":
            nan_table.add_column(
                np.zeros(
                    (len(nonexample_identifiers), len(self.dl1dh_reader.tel_ids)),
                    dtype=bool,
                ),
                name=f"{self.prefix}_telescopes",
            )
        return nan_table

    def _store_pointing(self, all_identifiers):
        """
        Store the telescope pointing table from  to the output file.

        Parameters:
        -----------
        all_identifiers : astropy.table.Table
            Table containing the telescope pointing information.
        """

        # Initialize the pointing interpolator from ctapipe
        pointing_interpolator = PointingInterpolator(
            bounds_error=False, extrapolate=True
        )
        pointing_info = []
        for tel_id in self.dl1dh_reader.selected_telescopes[self.dl1dh_reader.tel_type]:
            # Get the telescope pointing from the dl1dh reader
            tel_pointing = self.dl1dh_reader.telescope_pointings[f"tel_{tel_id:03d}"]
            # Add the telescope pointing table to the pointing interpolator
            pointing_interpolator.add_table(tel_id, tel_pointing)
            tel_identifiers = all_identifiers.copy()
            if self.dl1dh_reader.mode == "mono":
                tel_identifiers = tel_identifiers[tel_identifiers["tel_id"] == tel_id]
            # Interpolate the telescope pointing
            tel_altitude, tel_azimuth = pointing_interpolator(
                tel_id, tel_identifiers["time"]
            )
            tel_identifiers.add_column(tel_azimuth, name="pointing_azimuth")
            tel_identifiers.add_column(tel_altitude, name="pointing_altitude")
            pointing_info.append(tel_identifiers)
            if self.dl1dh_reader.mode == "mono":
                tel_pointing_table = Table(
                    {
                        "time": tel_identifiers["time"],
                        "azimuth": tel_identifiers["pointing_azimuth"],
                        "altitude": tel_identifiers["pointing_altitude"],
                    }
                )
                write_table(
                    tel_pointing_table,
                    self.output_path,
                    f"{POINTING_GROUP}/tel_{tel_id:03d}",
                    overwrite=self.overwrite_tables,
                )
                self.log.info(
                    "DL1 telescope pointing table was stored in '%s' under '%s'",
                    self.output_path,
                    f"{POINTING_GROUP}/tel_{tel_id:03d}",
                )
        pointing_info = vstack(pointing_info)
        if self.dl1dh_reader.mode == "stereo":
            # Group the pointing information by subarray event keys
            # TODO: This needs to be debugged with SST1M data
            pointing_info_grouped = pointing_info.group_by(SUBARRAY_EVENT_KEYS)
            pointing_mean = pointing_info_grouped.groups.aggregate(np.mean)
            pointing_info = join(
                all_identifiers,
                pointing_mean,
                keys=SUBARRAY_EVENT_KEYS,
            )
            # TODO: use keep_order for astropy v7.0.0
            pointing_info.sort(SUBARRAY_EVENT_KEYS)
            # Create the pointing table
            pointing_table = Table(
                {
                    "time": pointing_info["time"],
                    "array_azimuth": pointing_info["pointing_azimuth"],
                    "array_altitude": pointing_info["pointing_altitude"],
                    "array_ra": np.nan * np.ones(len(pointing_info)),
                    "array_dec": np.nan * np.ones(len(pointing_info)),
                }
            )
            # Save the pointing table to the output file
            write_table(
                pointing_table,
                self.output_path,
                f"{SUBARRAY_POINTING_GROUP}",
                overwrite=self.overwrite_tables,
            )
            self.log.info(
                "DL1 subarray pointing table was stored in '%s' under '%s'",
                self.output_path,
                f"{SUBARRAY_POINTING_GROUP}",
            )
        return pointing_info

    def _create_feature_vectors_table(
        self,
        example_identifiers,
        nonexample_identifiers=None,
        classification_feature_vectors=None,
        energy_feature_vectors=None,
        direction_feature_vectors=None,
    ):
        """
        Create the table for the DL1 feature vectors.

        This method creates a table with the DL1 feature vectors for the example identifiers and fill NaNs for
        non-example identifiers. The feature vectors are stored in the columns of the table. The table also
        contains a column for the valid predictions.

        Parameters:
        -----------
        example_identifiers : astropy.table.Table
            Table containing the example identifiers.
        nonexample_identifiers : astropy.table.Table or None
            Table containing the non-example identifiers to fill the NaNs.
        classification_feature_vectors : np.ndarray or None
            Array containing the classification feature vectors.
        energy_feature_vectors : np.ndarray or None
            Array containing the energy feature vectors.
        direction_feature_vectors : np.ndarray or None
            Array containing the direction feature vectors.

        Returns:
        --------
        feature_vector_table : astropy.table.Table
            Table containing the DL1 feature vectors for the example and non-example identifiers.
        """
        # Create the feature vector table
        feature_vector_table = example_identifiers.copy()
        feature_vector_table.remove_columns(
            ["pointing_azimuth", "pointing_altitude", "time"]
        )
        columns_list, shapes_list = [], []
        if classification_feature_vectors is not None:
            is_valid_col = ~np.isnan(
                np.min(classification_feature_vectors, axis=1), dtype=bool
            )
            feature_vector_table.add_column(
                classification_feature_vectors,
                name=f"{self.prefix}_tel_classification_feature_vectors",
            )
            if nonexample_identifiers is not None:
                columns_list.append(f"{self.prefix}_tel_classification_feature_vectors")
                shapes_list.append(
                    (
                        len(nonexample_identifiers),
                        classification_feature_vectors.shape[1],
                    )
                )
        if energy_feature_vectors is not None:
            is_valid_col = ~np.isnan(np.min(energy_feature_vectors, axis=1), dtype=bool)
            feature_vector_table.add_column(
                energy_feature_vectors, name=f"{self.prefix}_tel_energy_feature_vectors"
            )
            if nonexample_identifiers is not None:
                columns_list.append(f"{self.prefix}_tel_energy_feature_vectors")
                shapes_list.append(
                    (
                        len(nonexample_identifiers),
                        energy_feature_vectors.shape[1],
                    )
                )
        if direction_feature_vectors is not None:
            is_valid_col = ~np.isnan(
                np.min(direction_feature_vectors, axis=1), dtype=bool
            )
            feature_vector_table.add_column(
                direction_feature_vectors,
                name=f"{self.prefix}_tel_geometry_feature_vectors",
            )
            if nonexample_identifiers is not None:
                columns_list.append(f"{self.prefix}_tel_geometry_feature_vectors")
                shapes_list.append(
                    (
                        len(nonexample_identifiers),
                        direction_feature_vectors.shape[1],
                    )
                )
        # Produce output table with NaNs for missing predictions
        if nonexample_identifiers is not None:
            if len(nonexample_identifiers) > 0:
                nan_table = self._create_nan_table(
                    nonexample_identifiers,
                    columns=columns_list,
                    shapes=shapes_list,
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
        return feature_vector_table


class MonoPredictCTLearnModel(PredictCTLearnModel):
    """
    Tool to predict the gammaness, energy and arrival direction from monoscopic R1/DL1 data using CTLearn models.

    This tool extends the ``PredictCTLearnModel`` to specifically handle monoscopic R1/DL1 data. The prediction
    is performed using the CTLearn models. The data is stored in the output file following the ctapipe DL2 data format.
    It also stores the telescope pointing monitoring and DL1 feature vectors (if selected) in the output file.

    Attributes
    ----------
    name : str
        Name of the tool.
    description : str
        Description of the tool.
    examples : str
        Examples of how to use the tool.

    Methods
    -------
    start()
        Start the tool.
    _store_mc_telescope_pointing(all_identifiers)
        Store the telescope pointing table for the mono mode for MC simulation.
    """

    name = "ctlearn-predict-mono-model"
    description = __doc__

    examples = """
    To predict from pixel-wise image data in mono mode using trained CTLearn models:
    > ctlearn-predict-mono-model \\
        --input_url input.dl1.h5 \\
        --PredictCTLearnModel.batch_size=64 \\
        --PredictCTLearnModel.dl1dh_reader_type=DLImageReader \\
        --DLImageReader.channels=cleaned_image \\
        --DLImageReader.channels=cleaned_relative_peak_time \\
        --DLImageReader.image_mapper_type=BilinearMapper \\
        --type_model="/path/to/your/mono/type/ctlearn_model.cpk" \\
        --energy_model="/path/to/your/mono/energy/ctlearn_model.cpk" \\
        --cameradirection_model="/path/to/your/mono/cameradirection/ctlearn_model.cpk" \\
        --dl1-features \\
        --use-HDF5Merger \\
        --no-dl1-images \\
        --no-true-images \\
        --output output.dl2.h5 \\
        --PredictCTLearnModel.overwrite_tables=True \\

    To predict from pixel-wise waveform data in mono mode using trained CTLearn models:
    > ctlearn-predict-mono-model \\
        --input_url input.r1.h5 \\
        --PredictCTLearnModel.dl1dh_reader_type=DLWaveformReader \\
        --DLWaveformReader.sequnce_length=20 \\
        --DLWaveformReader.image_mapper_type=BilinearMapper \\
        --type_model="/path/to/your/mono_waveform/type/ctlearn_model.cpk" \\
        --energy_model="/path/to/your/mono_waveform/energy/ctlearn_model.cpk" \\
        --cameradirection_model="/path/to/your/mono_waveform/cameradirection/ctlearn_model.cpk" \\
        --use-HDF5Merger \\
        --no-r0-waveforms \\
        --no-r1-waveforms \\
        --no-dl1-images \\
        --no-true-images \\
        --output output.dl2.h5 \\
        --PredictCTLearnModel.overwrite_tables=True \\
    """

    stereo_combiner_cls = ComponentName(
        StereoCombiner,
        default_value="StereoMeanCombiner",
        help="Which stereo combination method to use after the monoscopic reconstruction.",
    ).tag(config=True)

    def start(self):
        self.log.info("Processing the telescope pointings...")
        # Retrieve the IDs from the dl1dh for the prediction tables
        example_identifiers = self.dl1dh_reader.example_identifiers.copy()
        example_identifiers.keep_columns(TELESCOPE_EVENT_KEYS)
        all_identifiers = self.dl1dh_reader.tel_trigger_table.copy()
        all_identifiers.keep_columns(TELESCOPE_EVENT_KEYS + ["time"])
        nonexample_identifiers = setdiff(
            all_identifiers, example_identifiers, keys=TELESCOPE_EVENT_KEYS
        )
        nonexample_identifiers.remove_column("time")
        # Pointing table for the mono mode for MC simulation
        if self.dl1dh_reader.process_type == ProcessType.Simulation:
            pointing_info = self._store_mc_telescope_pointing(all_identifiers)

        # Pointing table for the observation mode
        if self.dl1dh_reader.process_type == ProcessType.Observation:
            pointing_info = super()._store_pointing(all_identifiers)

        self.log.info("Starting the prediction...")
        classification_feature_vectors = None
        if self.load_type_model_from is not None:
            self.type_stereo_combiner = StereoCombiner.from_name(
                self.stereo_combiner_cls,
                prefix=self.prefix,
                property=ReconstructionProperty.PARTICLE_TYPE,
                parent=self,
            )
            # Predict the energy of the primary particle
            classification_table, classification_feature_vectors = (
                super()._predict_classification(example_identifiers)
            )
            if self.dl2_telescope:
                # Produce output table with NaNs for missing predictions
                if len(nonexample_identifiers) > 0:
                    nan_table = super()._create_nan_table(
                        nonexample_identifiers,
                        columns=[f"{self.prefix}_tel_prediction"],
                        shapes=[(len(nonexample_identifiers),)],
                    )
                    classification_table = vstack([classification_table, nan_table])
                # Add is_valid column to the energy table
                classification_table.add_column(
                    ~np.isnan(
                        classification_table[f"{self.prefix}_tel_prediction"].data,
                        dtype=bool,
                    ),
                    name=f"{self.prefix}_tel_is_valid",
                )
                # Add the default values and meta data to the table
                add_defaults_and_meta(
                    classification_table,
                    ParticleClassificationContainer,
                    prefix=self.prefix,
                    add_tel_prefix=True,
                )
                for tel_id in self.dl1dh_reader.selected_telescopes[
                    self.dl1dh_reader.tel_type
                ]:
                    # Retrieve the example identifiers for the selected telescope
                    telescope_mask = classification_table["tel_id"] == tel_id
                    classification_tel_table = classification_table[telescope_mask]
                    classification_tel_table.sort(TELESCOPE_EVENT_KEYS)
                    # Save the prediction to the output file for the selected telescope
                    write_table(
                        classification_tel_table,
                        self.output_path,
                        f"{DL2_TELESCOPE_GROUP}/classification/{self.prefix}/tel_{tel_id:03d}",
                        overwrite=self.overwrite_tables,
                    )
                    self.log.info(
                        "DL2 prediction data was stored in '%s' under '%s'",
                        self.output_path,
                        f"{DL2_TELESCOPE_GROUP}/classification/{self.prefix}/tel_{tel_id:03d}",
                    )
            if self.dl2_subarray:
                self.log.info("Processing and storing the subarray type prediction...")
                # Combine the telescope predictions to the subarray prediction using the stereo combiner
                subarray_classification_table = self.type_stereo_combiner.predict_table(
                    classification_table
                )
                # TODO: Remove temporary fix once the stereo combiner returns correct table
                # Check if the table has to be converted to a boolean mask
                if (
                    subarray_classification_table[f"{self.prefix}_telescopes"].dtype
                    != np.bool_
                ):
                    # Create boolean mask for telescopes that participate in the stereo reconstruction combination
                    reco_telescopes = np.zeros(
                        (
                            len(subarray_classification_table),
                            len(self.dl1dh_reader.tel_ids),
                        ),
                        dtype=bool,
                    )
                    # Loop over the table and set the boolean mask for the telescopes
                    for index, tel_id_mask in enumerate(
                        subarray_classification_table[f"{self.prefix}_telescopes"]
                    ):
                        if not tel_id_mask:
                            continue
                        for tel_id in tel_id_mask:
                            reco_telescopes[index][
                                self.dl1dh_reader.subarray.tel_ids_to_indices(tel_id)
                            ] = True
                    # Overwrite the column with the boolean mask with fix length
                    subarray_classification_table[f"{self.prefix}_telescopes"] = (
                        reco_telescopes
                    )
                # Save the prediction to the output file
                write_table(
                    subarray_classification_table,
                    self.output_path,
                    f"{DL2_SUBARRAY_GROUP}/classification/{self.prefix}",
                    overwrite=self.overwrite_tables,
                )
                self.log.info(
                    "DL2 prediction data was stored in '%s' under '%s'",
                    self.output_path,
                    f"{DL2_SUBARRAY_GROUP}/classification/{self.prefix}",
                )
        energy_feature_vectors = None
        if self.load_energy_model_from is not None:
            self.energy_stereo_combiner = StereoCombiner.from_name(
                self.stereo_combiner_cls,
                prefix=self.prefix,
                property=ReconstructionProperty.ENERGY,
                parent=self,
            )
            # Predict the energy of the primary particle
            energy_table, energy_feature_vectors = super()._predict_energy(
                example_identifiers
            )
            if self.dl2_telescope:
                # Produce output table with NaNs for missing predictions
                if len(nonexample_identifiers) > 0:
                    nan_table = super()._create_nan_table(
                        nonexample_identifiers,
                        columns=[f"{self.prefix}_tel_energy"],
                        shapes=[(len(nonexample_identifiers),)],
                    )
                    energy_table = vstack([energy_table, nan_table])
                # Add is_valid column to the energy table
                energy_table.add_column(
                    ~np.isnan(
                        energy_table[f"{self.prefix}_tel_energy"].data, dtype=bool
                    ),
                    name=f"{self.prefix}_tel_is_valid",
                )
                # Add the default values and meta data to the table
                add_defaults_and_meta(
                    energy_table,
                    ReconstructedEnergyContainer,
                    prefix=self.prefix,
                    add_tel_prefix=True,
                )
                for tel_id in self.dl1dh_reader.selected_telescopes[
                    self.dl1dh_reader.tel_type
                ]:
                    # Retrieve the example identifiers for the selected telescope
                    telescope_mask = energy_table["tel_id"] == tel_id
                    energy_tel_table = energy_table[telescope_mask]
                    energy_tel_table.sort(TELESCOPE_EVENT_KEYS)
                    # Save the prediction to the output file
                    write_table(
                        energy_tel_table,
                        self.output_path,
                        f"{DL2_TELESCOPE_GROUP}/energy/{self.prefix}/tel_{tel_id:03d}",
                        overwrite=self.overwrite_tables,
                    )
                    self.log.info(
                        "DL2 prediction data was stored in '%s' under '%s'",
                        self.output_path,
                        f"{DL2_TELESCOPE_GROUP}/energy/{self.prefix}/tel_{tel_id:03d}",
                    )
            if self.dl2_subarray:
                self.log.info(
                    "Processing and storing the subarray energy prediction..."
                )
                # Combine the telescope predictions to the subarray prediction using the stereo combiner
                subarray_energy_table = self.energy_stereo_combiner.predict_table(
                    energy_table
                )
                # TODO: Remove temporary fix once the stereo combiner returns correct table
                # Check if the table has to be converted to a boolean mask
                if subarray_energy_table[f"{self.prefix}_telescopes"].dtype != np.bool_:
                    # Create boolean mask for telescopes that participate in the stereo reconstruction combination
                    reco_telescopes = np.zeros(
                        (len(subarray_energy_table), len(self.dl1dh_reader.tel_ids)),
                        dtype=bool,
                    )
                    # Loop over the table and set the boolean mask for the telescopes
                    for index, tel_id_mask in enumerate(
                        subarray_energy_table[f"{self.prefix}_telescopes"]
                    ):
                        if not tel_id_mask:
                            continue
                        for tel_id in tel_id_mask:
                            reco_telescopes[index][
                                self.dl1dh_reader.subarray.tel_ids_to_indices(tel_id)
                            ] = True
                    # Overwrite the column with the boolean mask with fix length
                    subarray_energy_table[f"{self.prefix}_telescopes"] = reco_telescopes
                # Save the prediction to the output file
                write_table(
                    subarray_energy_table,
                    self.output_path,
                    f"{DL2_SUBARRAY_GROUP}/energy/{self.prefix}",
                    overwrite=self.overwrite_tables,
                )
                self.log.info(
                    "DL2 prediction data was stored in '%s' under '%s'",
                    self.output_path,
                    f"{DL2_SUBARRAY_GROUP}/energy/{self.prefix}",
                )
        direction_feature_vectors = None
        if self.load_cameradirection_model_from is not None:
            self.geometry_stereo_combiner = StereoCombiner.from_name(
                self.stereo_combiner_cls,
                prefix=self.prefix,
                property=ReconstructionProperty.GEOMETRY,
                parent=self,
            )
            # Join the prediction table with the telescope pointing table
            example_identifiers = join(
                left=example_identifiers,
                right=pointing_info,
                keys=TELESCOPE_EVENT_KEYS,
            )
            # Predict the arrival direction of the primary particle
            direction_table, direction_feature_vectors = (
                super()._predict_cameradirection(example_identifiers)
            )
            direction_tel_tables = []
            if self.dl2_telescope:
                for tel_id in self.dl1dh_reader.selected_telescopes[
                    self.dl1dh_reader.tel_type
                ]:
                    # Retrieve the example identifiers for the selected telescope
                    telescope_mask = direction_table["tel_id"] == tel_id
                    direction_tel_table = direction_table[telescope_mask]
                    direction_tel_table = super()._transform_cam_coord_offsets_to_sky(
                        direction_tel_table
                    )
                    # Produce output table with NaNs for missing predictions
                    nan_telescope_mask = nonexample_identifiers["tel_id"] == tel_id
                    nonexample_identifiers_tel = nonexample_identifiers[
                        nan_telescope_mask
                    ]
                    if len(nonexample_identifiers_tel) > 0:
                        nan_table = super()._create_nan_table(
                            nonexample_identifiers_tel,
                            columns=[f"{self.prefix}_tel_alt", f"{self.prefix}_tel_az"],
                            shapes=[
                                (len(nonexample_identifiers_tel),),
                                (len(nonexample_identifiers_tel),),
                            ],
                        )
                        direction_tel_table = vstack([direction_tel_table, nan_table])
                    direction_tel_table.sort(TELESCOPE_EVENT_KEYS)
                    # Add is_valid column to the direction table
                    direction_tel_table.add_column(
                        ~np.isnan(
                            direction_tel_table[f"{self.prefix}_tel_alt"].data,
                            dtype=bool,
                        ),
                        name=f"{self.prefix}_tel_is_valid",
                    )
                    # Add the default values and meta data to the table
                    add_defaults_and_meta(
                        direction_tel_table,
                        ReconstructedGeometryContainer,
                        prefix=self.prefix,
                        add_tel_prefix=True,
                    )
                    direction_tel_tables.append(direction_tel_table)
                    # Save the prediction to the output file
                    write_table(
                        direction_tel_table,
                        self.output_path,
                        f"{DL2_TELESCOPE_GROUP}/geometry/{self.prefix}/tel_{tel_id:03d}",
                        overwrite=self.overwrite_tables,
                    )
                    self.log.info(
                        "DL2 prediction data was stored in '%s' under '%s'",
                        self.output_path,
                        f"{DL2_TELESCOPE_GROUP}/geometry/{self.prefix}/tel_{tel_id:03d}",
                    )
            if self.dl2_subarray:
                self.log.info(
                    "Processing and storing the subarray geometry prediction..."
                )
                # Stack the telescope tables to the subarray table
                direction_tel_tables = vstack(direction_tel_tables)
                # Sort the table by the telescope event keys
                direction_tel_tables.sort(TELESCOPE_EVENT_KEYS)
                # Combine the telescope predictions to the subarray prediction using the stereo combiner
                subarray_direction_table = self.geometry_stereo_combiner.predict_table(
                    direction_tel_tables
                )
                # TODO: Remove temporary fix once the stereo combiner returns correct table
                # Check if the table has to be converted to a boolean mask
                if (
                    subarray_direction_table[f"{self.prefix}_telescopes"].dtype
                    != np.bool_
                ):
                    # Create boolean mask for telescopes that participate in the stereo reconstruction combination
                    reco_telescopes = np.zeros(
                        (len(subarray_direction_table), len(self.dl1dh_reader.tel_ids)),
                        dtype=bool,
                    )
                    # Loop over the table and set the boolean mask for the telescopes
                    for index, tel_id_mask in enumerate(
                        subarray_direction_table[f"{self.prefix}_telescopes"]
                    ):
                        if not tel_id_mask:
                            continue
                        for tel_id in tel_id_mask:
                            reco_telescopes[index][
                                self.dl1dh_reader.subarray.tel_ids_to_indices(tel_id)
                            ] = True
                    # Overwrite the column with the boolean mask with fix length
                    subarray_direction_table[f"{self.prefix}_telescopes"] = (
                        reco_telescopes
                    )
                # Save the prediction to the output file
                write_table(
                    subarray_direction_table,
                    self.output_path,
                    f"{DL2_SUBARRAY_GROUP}/geometry/{self.prefix}",
                    overwrite=self.overwrite_tables,
                )
                self.log.info(
                    "DL2 prediction data was stored in '%s' under '%s'",
                    self.output_path,
                    f"{DL2_SUBARRAY_GROUP}/geometry/{self.prefix}",
                )
        # Create the feature vector table if the DL1 features are enabled
        if self.dl1_features:
            self.log.info("Processing and storing dl1 feature vectors...")
            feature_vector_table = super()._create_feature_vectors_table(
                example_identifiers,
                nonexample_identifiers,
                classification_feature_vectors,
                energy_feature_vectors,
                direction_feature_vectors,
            )
            # Loop over the selected telescopes and store the feature vectors
            # for each telescope in the output file. The feature vectors are stored
            # in the DL1_TELESCOPE_GROUP/features/{prefix}/tel_{tel_id:03d} table.
            for tel_id in self.dl1dh_reader.selected_telescopes[
                self.dl1dh_reader.tel_type
            ]:
                # Retrieve the example identifiers for the selected telescope
                telescope_mask = feature_vector_table["tel_id"] == tel_id
                feature_vectors_tel_table = feature_vector_table[telescope_mask]
                feature_vectors_tel_table.sort(TELESCOPE_EVENT_KEYS)
                # Save the prediction to the output file
                write_table(
                    feature_vectors_tel_table,
                    self.output_path,
                    f"{DL1_TELESCOPE_GROUP}/features/{self.prefix}/tel_{tel_id:03d}",
                    overwrite=self.overwrite_tables,
                )
                self.log.info(
                    "DL1 feature vectors was stored in '%s' under '%s'",
                    self.output_path,
                    f"{DL1_TELESCOPE_GROUP}/features/{self.prefix}/tel_{tel_id:03d}",
                )

    def _store_mc_telescope_pointing(self, all_identifiers):
        """
        Store the telescope pointing table from MC simulation to the output file.

        Parameters:
        -----------
        all_identifiers : astropy.table.Table
            Table containing the telescope pointing information.
        """
        # Create the pointing table for each telescope
        pointing_info = []
        for tel_id in self.dl1dh_reader.selected_telescopes[self.dl1dh_reader.tel_type]:
            # Pointing table for the mono mode
            tel_pointing = self.dl1dh_reader.get_tel_pointing(self.input_url, tel_id)
            tel_pointing.rename_column("telescope_pointing_azimuth", "pointing_azimuth")
            tel_pointing.rename_column(
                "telescope_pointing_altitude", "pointing_altitude"
            )
            # Join the prediction table with the telescope pointing table
            tel_pointing = join(
                left=tel_pointing,
                right=all_identifiers,
                keys=["obs_id", "tel_id"],
            )
            # TODO: use keep_order for astropy v7.0.0
            tel_pointing.sort(TELESCOPE_EVENT_KEYS)
            # Retrieve the example identifiers for the selected telescope
            tel_pointing_table = Table(
                {
                    "time": tel_pointing["time"],
                    "azimuth": tel_pointing["pointing_azimuth"],
                    "altitude": tel_pointing["pointing_altitude"],
                }
            )
            write_table(
                tel_pointing_table,
                self.output_path,
                f"{POINTING_GROUP}/tel_{tel_id:03d}",
                overwrite=self.overwrite_tables,
            )
            self.log.info(
                "DL1 telescope pointing table was stored in '%s' under '%s'",
                self.output_path,
                f"{POINTING_GROUP}/tel_{tel_id:03d}",
            )
            pointing_info.append(tel_pointing)
        pointing_info = vstack(pointing_info)
        return pointing_info


class StereoPredictCTLearnModel(PredictCTLearnModel):
    """
    Tool to predict the gammaness, energy and arrival direction from R1/DL1 stereoscopic data using CTLearn models.

    This tool extends the ``PredictCTLearnModel`` to specifically handle stereoscopic R1/DL1 data. The prediction
    is performed using the CTLearn models. The data is stored in the output file following the ctapipe DL2 data format.
    It also stores the telescope/subarray pointing monitoring and DL1 feature vectors (if selected) in the output file.

    Attributes
    ----------
    name : str
        Name of the tool.
    description : str
        Description of the tool.
    examples : str
        Examples of how to use the tool.

    Methods
    -------
    start()
        Start the tool.
    _store_mc_subarray_pointing(all_identifiers)
        Store the subarray pointing table for the stereo mode for MC simulation.
    """

    name = "ctlearn-predict-stereo-model"
    description = __doc__

    examples = """
    To predict from pixel-wise image data in stereo mode using trained CTLearn models:
    > ctlearn-predict-stereo-model \\
        --input_url input.dl1.h5 \\
        --PredictCTLearnModel.batch_size=16 \\
        --PredictCTLearnModel.dl1dh_reader_type=DLImageReader \\
        --DLImageReader.channels=cleaned_image \\
        --DLImageReader.channels=cleaned_relative_peak_time \\
        --DLImageReader.image_mapper_type=BilinearMapper \\
        --DLImageReader.mode=stereo \\
        --DLImageReader.min_telescopes=2 \\
        --PredictCTLearnModel.stack_telescope_images=True \\
        --type_model="/path/to/your/stereo/type/ctlearn_model.cpk" \\
        --energy_model="/path/to/your/stereo/energy/ctlearn_model.cpk" \\
        --skydirection_model="/path/to/your/stereo/skydirection/ctlearn_model.cpk" \\
        --output output.dl2.h5 \\
        --PredictCTLearnModel.overwrite_tables=True \\
    """

    def start(self):
        self.log.info("Processing the telescope pointings...")
        # Retrieve the IDs from the dl1dh for the prediction tables
        example_identifiers = self.dl1dh_reader.unique_example_identifiers.copy()
        example_identifiers.keep_columns(SUBARRAY_EVENT_KEYS)
        all_identifiers = self.dl1dh_reader.subarray_trigger_table.copy()
        all_identifiers.keep_columns(SUBARRAY_EVENT_KEYS + ["time"])
        nonexample_identifiers = setdiff(
            all_identifiers, example_identifiers, keys=SUBARRAY_EVENT_KEYS
        )
        nonexample_identifiers.remove_column("time")
        # Construct the survival telescopes for each event of the example_identifiers
        survival_telescopes = []
        for subarray_event in self.dl1dh_reader.example_identifiers_grouped.groups:
            survival_mask = np.zeros(len(self.dl1dh_reader.tel_ids), dtype=bool)
            survival_tels = [
                self.dl1dh_reader.subarray.tel_indices[tel_id]
                for tel_id in subarray_event["tel_id"].data
            ]
            survival_mask[survival_tels] = True
            survival_telescopes.append(survival_mask)
        # Add the survival telescopes to the example_identifiers
        example_identifiers.add_column(
            survival_telescopes, name=f"{self.prefix}_telescopes"
        )
        # Pointing table for the stereo mode for MC simulation
        if self.dl1dh_reader.process_type == ProcessType.Simulation:
            pointing_info = self._store_mc_subarray_pointing(all_identifiers)

        # Pointing table for the observation mode
        if self.dl1dh_reader.process_type == ProcessType.Observation:
            pointing_info = super()._store_pointing(all_identifiers)

        self.log.info("Starting the prediction...")
        classification_feature_vectors = None
        if self.load_type_model_from is not None:
            # Predict the energy of the primary particle
            classification_table, classification_feature_vectors = (
                super()._predict_classification(example_identifiers)
            )
            if self.dl2_subarray:
                # Produce output table with NaNs for missing predictions
                if len(nonexample_identifiers) > 0:
                    nan_table = super()._create_nan_table(
                        nonexample_identifiers,
                        columns=[f"{self.prefix}_tel_prediction"],
                        shapes=[(len(nonexample_identifiers),)],
                    )
                    classification_table = vstack([classification_table, nan_table])
                # Add is_valid column to the energy table
                classification_table.add_column(
                    ~np.isnan(
                        classification_table[f"{self.prefix}_tel_prediction"].data,
                        dtype=bool,
                    ),
                    name=f"{self.prefix}_tel_is_valid",
                )
                # Rename the columns for the stereo mode
                classification_table.rename_column(
                    f"{self.prefix}_tel_prediction", f"{self.prefix}_prediction"
                )
                classification_table.rename_column(
                    f"{self.prefix}_tel_is_valid", f"{self.prefix}_is_valid"
                )
                classification_table.sort(SUBARRAY_EVENT_KEYS)
                # Add the default values and meta data to the table
                add_defaults_and_meta(
                    classification_table,
                    ParticleClassificationContainer,
                    prefix=self.prefix,
                )
                # Save the prediction to the output file
                write_table(
                    classification_table,
                    self.output_path,
                    f"{DL2_SUBARRAY_GROUP}/classification/{self.prefix}",
                    overwrite=self.overwrite_tables,
                )
                self.log.info(
                    "DL2 prediction data was stored in '%s' under '%s'",
                    self.output_path,
                    f"{DL2_SUBARRAY_GROUP}/classification/{self.prefix}",
                )
        energy_feature_vectors = None
        if self.load_energy_model_from is not None:
            # Predict the energy of the primary particle
            energy_table, energy_feature_vectors = super()._predict_energy(
                example_identifiers
            )
            if self.dl2_subarray:
                # Produce output table with NaNs for missing predictions
                if len(nonexample_identifiers) > 0:
                    nan_table = super()._create_nan_table(
                        nonexample_identifiers,
                        columns=[f"{self.prefix}_tel_energy"],
                        shapes=[(len(nonexample_identifiers),)],
                    )
                    energy_table = vstack([energy_table, nan_table])
                # Add is_valid column to the energy table
                energy_table.add_column(
                    ~np.isnan(
                        energy_table[f"{self.prefix}_tel_energy"].data, dtype=bool
                    ),
                    name=f"{self.prefix}_tel_is_valid",
                )
                # Rename the columns for the stereo mode
                energy_table.rename_column(
                    f"{self.prefix}_tel_energy", f"{self.prefix}_energy"
                )
                energy_table.rename_column(
                    f"{self.prefix}_tel_is_valid", f"{self.prefix}_is_valid"
                )
                energy_table.sort(SUBARRAY_EVENT_KEYS)
                # Add the default values and meta data to the table
                add_defaults_and_meta(
                    energy_table,
                    ReconstructedEnergyContainer,
                    prefix=self.prefix,
                )
                # Save the prediction to the output file
                write_table(
                    energy_table,
                    self.output_path,
                    f"{DL2_SUBARRAY_GROUP}/energy/{self.prefix}",
                    overwrite=self.overwrite_tables,
                )
                self.log.info(
                    "DL2 prediction data was stored in '%s' under '%s'",
                    self.output_path,
                    f"{DL2_SUBARRAY_GROUP}/energy/{self.prefix}",
                )
        direction_feature_vectors = None
        if self.load_skydirection_model_from is not None:
            # Join the prediction table with the telescope pointing table
            example_identifiers = join(
                left=example_identifiers,
                right=pointing_info,
                keys=SUBARRAY_EVENT_KEYS,
            )
            # Predict the arrival direction of the primary particle
            direction_table, direction_feature_vectors = super()._predict_skydirection(
                example_identifiers
            )
            if self.dl2_subarray:
                # Transform the spherical coordinate offsets to sky coordinates
                direction_table = super()._transform_spher_coord_offsets_to_sky(
                    direction_table
                )
                # Produce output table with NaNs for missing predictions
                if len(nonexample_identifiers) > 0:
                    nan_table = super()._create_nan_table(
                        nonexample_identifiers,
                        columns=[f"{self.prefix}_alt", f"{self.prefix}_az"],
                        shapes=[
                            (len(nonexample_identifiers),),
                            (len(nonexample_identifiers),),
                        ],
                    )
                    direction_table = vstack([direction_table, nan_table])
                # Add is_valid column to the direction table
                direction_table.add_column(
                    ~np.isnan(direction_table[f"{self.prefix}_alt"].data, dtype=bool),
                    name=f"{self.prefix}_is_valid",
                )
                direction_table.sort(SUBARRAY_EVENT_KEYS)
                # Add the default values and meta data to the table
                add_defaults_and_meta(
                    direction_table,
                    ReconstructedGeometryContainer,
                    prefix=self.prefix,
                )
                # Save the prediction to the output file
                write_table(
                    direction_table,
                    self.output_path,
                    f"{DL2_SUBARRAY_GROUP}/geometry/{self.prefix}",
                    overwrite=self.overwrite_tables,
                )
                self.log.info(
                    "DL2 prediction data was stored in '%s' under '%s'",
                    self.output_path,
                    f"{DL2_SUBARRAY_GROUP}/geometry/{self.prefix}",
                )

        # Create the feature vector table if the DL1 features are enabled
        if self.dl1_features:
            self.log.info("Processing and storing dl1 feature vectors...")
            feature_vector_table = super()._create_feature_vectors_table(
                example_identifiers,
                nonexample_identifiers,
                classification_feature_vectors,
                energy_feature_vectors,
                direction_feature_vectors,
            )
            # Loop over the selected telescopes and store the feature vectors
            # for each telescope in the output file. The feature vectors are stored
            # in the DL1_TELESCOPE_GROUP/features/{prefix}/tel_{tel_id:03d} table.
            # Rename the columns for the stereo mode
            feature_vector_table.rename_column(
                f"{self.prefix}_tel_classification_feature_vectors",
                f"{self.prefix}_classification_feature_vectors",
            )
            feature_vector_table.rename_column(
                f"{self.prefix}_tel_energy_feature_vectors",
                f"{self.prefix}_energy_feature_vectors",
            )
            feature_vector_table.rename_column(
                f"{self.prefix}_tel_geometry_feature_vectors",
                f"{self.prefix}_geometry_feature_vectors",
            )
            feature_vector_table.rename_column(
                f"{self.prefix}_tel_is_valid", f"{self.prefix}_is_valid"
            )
            feature_vector_table.sort(SUBARRAY_EVENT_KEYS)
            # Save the prediction to the output file
            write_table(
                feature_vector_table,
                self.output_path,
                f"{DL1_SUBARRAY_GROUP}/features/{self.prefix}",
                overwrite=self.overwrite_tables,
            )
            self.log.info(
                "DL1 feature vectors was stored in '%s' under '%s'",
                self.output_path,
                f"{DL1_SUBARRAY_GROUP}/features/{self.prefix}",
            )

    def _store_mc_subarray_pointing(self, all_identifiers):
        """
        Store the subarray pointing table from MC simulation to the output file.

        Parameters:
        -----------
        all_identifiers : astropy.table.Table
            Table containing the subarray pointing information.
        """
        # Read the subarray pointing table
        pointing_info = read_table(
            self.input_url,
            f"{SIMULATION_CONFIG_TABLE}",
        )
        # Assuming min_az = max_az and min_alt = max_alt
        pointing_info.keep_columns(["obs_id", "min_az", "min_alt"])
        pointing_info.rename_column("min_az", "pointing_azimuth")
        pointing_info.rename_column("min_alt", "pointing_altitude")
        # Join the prediction table with the telescope pointing table
        pointing_info = join(
            left=pointing_info,
            right=all_identifiers,
            keys=["obs_id"],
        )
        # TODO: use keep_order for astropy v7.0.0
        pointing_info.sort(SUBARRAY_EVENT_KEYS)
        # Create the pointing table
        pointing_table = Table(
            {
                "time": pointing_info["time"],
                "array_azimuth": pointing_info["pointing_azimuth"],
                "array_altitude": pointing_info["pointing_altitude"],
                "array_ra": np.nan * np.ones(len(pointing_info)),
                "array_dec": np.nan * np.ones(len(pointing_info)),
            }
        )
        # Save the pointing table to the output file
        write_table(
            pointing_table,
            self.output_path,
            f"{SUBARRAY_POINTING_GROUP}",
            overwrite=self.overwrite_tables,
        )
        self.log.info(
            "DL1 subarray pointing table was stored in '%s' under '%s'",
            self.output_path,
            f"{SUBARRAY_POINTING_GROUP}",
        )
        return pointing_info


def mono_tool():
    # Run the tool
    mono_tool = MonoPredictCTLearnModel()
    mono_tool.run()


def stereo_tool():
    # Run the tool
    stereo_tool = StereoPredictCTLearnModel()
    stereo_tool.run()


if __name__ == "mono_tool":
    mono_tool()

if __name__ == "stereo_tool":
    stereo_tool()
