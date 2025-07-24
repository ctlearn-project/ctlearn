"""
Tools to predict the gammaness, energy and arrival direction in monoscopic and stereoscopic mode using ``CTLearnModel`` on R1/DL1 data using the ``DLDataReader`` and ``DLDataLoader``.
"""

import atexit
import pathlib
import numpy as np
import os
import tensorflow as tf
import keras
import threading
from ctlearn.core.ctlearn_enum import Task

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
from ctlearn.tools.train.pytorch.utils import (
    sanity_check,
    read_configuration,
    expected_structure,
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
from ctlearn.core.data_loader.loader import DLDataLoader
from ctlearn.utils import monitor_progress

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

__all__ = ["PredictCTLearnModel"]

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
    
    pytorch_config_file = Path(
        default_value="./ctlearn/tools/train/pytorch/config/training_config_iaa_neutron_training.yml",
        help="Pytorch config file",
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

    framework_type = CaselessStrEnum(
        ["pytorch", "keras"],
        default_value="keras",
        help="Framework to use: pytorch or keras",
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
        ("f", "framework"): "PredictCTLearnModel.framework_type",
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
        if self.framework_type == "pytorch":
            import torch
            self.log.info(f"Using {self.pytorch_config_file} config file for pytorch framework")
            self.parameters = read_configuration(self.pytorch_config_file)
            sanity_check(self.parameters, expected_structure)
            self.device_str = self.parameters["arch"]["device"]
            self.device = torch.device(self.device_str)
            self.tasks = []
            self.type_mu = self.parameters["normalization"]["type_mu"]
            self.type_sigma = self.parameters["normalization"]["type_sigma"]
            self.dir_mu = self.parameters["normalization"]["dir_mu"]
            self.dir_sigma = self.parameters["normalization"]["dir_sigma"]
            self.energy_mu = self.parameters["normalization"]["energy_mu"]
            self.energy_sigma = self.parameters["normalization"]["energy_sigma"]
            
            if self.load_type_model_from is not None:
                self.tasks.append(Task.type)
            if self.load_energy_model_from is not None:
                self.tasks.append(Task.energy)
            if self.load_cameradirection_model_from is not None:
                self.tasks.append(Task.direction)

        # Check if the ctapipe HDF5Merger component is enabled
        if self.use_HDF5Merger:
            if os.path.exists(self.output_path):
                raise ToolConfigurationError(
                    f"The output file '{self.output_path}' already exists. Please use "
                    "'--no-use-HDF5Merger' to disable the usage of the HDF5Merger component."
                )
            # Copy selected tables from the input file to the output file
            self.log.info("Copying to output destination.")
            stop_event = threading.Event()
            monitor_thread = threading.Thread(target=monitor_progress, args=(self.input_url, self.output_path, stop_event, self.log))
            monitor_thread.start()
            
            try:
                with HDF5Merger(self.output_path, parent=self) as merger:
                    merger(self.input_url)
            finally:
                stop_event.set()
                monitor_thread.join()
                
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

    def _predict_with_model(self, model_path, task):
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
        predict_data = None
        feature_vectors = None
        
        if self.framework_type == "keras":
             from ctlearn.tools.predict.keras.predic_model_keras import predict_with_model
             predict_data, feature_vectors = predict_with_model(self,model_path)
             return predict_data, feature_vectors
         
        if self.framework_type == "pytorch":
            from ctlearn.tools.predict.pytorch.predic_model_pytorch import predict_with_model_pytorch
            predict_data, feature_vectors = predict_with_model_pytorch(self, task)
            return predict_data, feature_vectors
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
            self.load_type_model_from, Task.type
        )
        # Create prediction table and add the predicted classification score ('gammaness')
        classification_table = example_identifiers.copy()
        classification_table.add_column(
            predict_data["type"].T[1], name=f"{self.prefix}_tel_prediction"
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
            self.load_energy_model_from, Task.energy
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
            self.load_cameradirection_model_from, Task.direction
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
            ].camera.geometry.frame.focal_length,
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
        # Set the reco direction in (fov_lon, fov_lat) coordinates
        reco_direction = SkyCoord(
            fov_lon=table["fov_lon"],
            fov_lat=table["fov_lat"],
            frame=nom_frame,
        )
        # Transform the reco direction from nominal frame to the AltAz frame
        sky_coord = reco_direction.transform_to(altaz)
        # Add the reconstructed direction (az, alt) to the prediction table
        table.add_column(sky_coord.az.to(u.deg), name=f"{self.prefix}_az")
        table.add_column(sky_coord.alt.to(u.deg), name=f"{self.prefix}_alt")
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


if __name__ == "__main__":
    pass