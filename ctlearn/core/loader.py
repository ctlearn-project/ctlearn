import numpy as np
import astropy.units as u
import keras
from keras.utils import Sequence, to_categorical

from dl1_data_handler.reader import ProcessType


class DLDataLoader(Sequence):
    """
    Generates batches for Keras application.

    DLDataLoader is a data loader class that inherits from ``~keras.utils.Sequence``.
    It is designed to handle and load data for deep learning models in a batch-wise manner.

    Attributes:
    -----------
    data_reader : DLDataReader
        An instance of DLDataReader to read the input data.
    indices : list
        List of indices to specify the data to be loaded.
    tasks : list
        List of tasks to be performed on the data to properly set up the labels.
    batch_size : int
        Size of the batch to load the data.
    random_seed : int, optional
        Whether to shuffle the data after each epoch with a provided random seed.

    Methods:
    --------
    __len__():
        Returns the number of batches per epoch.
    on_epoch_end():
        Updates indices after each epoch if random seed is provided.
    __getitem__(index):
        Generates one batch of data using _get_mono_item(index) or _get_stereo_item(index).
    _get_mono_item(index):
        Generates one batch of monoscopic data.
    _get_stereo_item(index):
        Generates one batch of stereoscopic data.
    """

    def __init__(
        self,
        DLDataReader,
        indices,
        tasks,
        batch_size=64,
        random_seed=None,
        sort_by_intensity=False,
        stack_telescope_images=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        "Initialization"
        self.DLDataReader = DLDataReader
        self.indices = indices
        self.tasks = tasks
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.on_epoch_end()
        self.stack_telescope_images = stack_telescope_images
        self.sort_by_intensity = sort_by_intensity

        # Set the input shape based on the mode of the DLDataReader
        if self.DLDataReader.__class__.__name__ != "DLFeatureVectorReader":
            if self.DLDataReader.mode == "mono":
                self.input_shape = self.DLDataReader.input_shape
            elif self.DLDataReader.mode == "stereo":
                self.input_shape = self.DLDataReader.input_shape[
                    list(self.DLDataReader.selected_telescopes)[0]
                ]
                # Reshape inputs into proper dimensions
                # for the stereo analysis with stacked images
                if self.stack_telescope_images:
                    self.input_shape = (
                        self.input_shape[1],
                        self.input_shape[2],
                        self.input_shape[0] * self.input_shape[3],
                    )

    def __len__(self):
        """
        Returns the number of batches per epoch.

        This method calculates the number of batches required to cover the entire dataset
        based on the batch size.

        Returns:
        --------
        int
            Number of batches per epoch.
        """
        return int(np.floor(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        """
        Updates indices after each epoch. If a random seed is provided, the indices are shuffled.

        This method is called at the end of each epoch to ensure that the data is shuffled
        if the shuffle attribute is set to True. This helps in improving the training process
        by providing the model with a different order of data in each epoch.
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        """
        Generate one batch of data and retrieve the features and labels.

        This method is called to generate one batch of monoscopic and stereoscopic data based on
        the index provided. It calls either _get_mono_item(batch) or _get_stereo_item(batch)
        based on the mode of the DLDataReader.

        Parameters:
        -----------
        index : int
            Index of the batch to generate.

        Returns:
        --------
        tuple
            A tuple containing the input data as features and the corresponding labels.
        """
        # Generate indices of the batch
        batch_indices = self.indices[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        features, labels = None, None
        if self.DLDataReader.mode == "mono":
            batch = self.DLDataReader.generate_mono_batch(batch_indices)
            features, labels = self._get_mono_item(batch)
        elif self.DLDataReader.mode == "stereo":
            batch = self.DLDataReader.generate_stereo_batch(batch_indices)
            features, labels = self._get_stereo_item(batch)
        return features, labels

    def _get_mono_item(self, batch):
        """
        Retrieve the features and labels for one batch of monoscopic data.

        This method is called to retrieve the features and labels for one batch of
        monoscopic data. The labels are set up based on the tasks specified.

        Parameters:
        -----------
        batch : astropy.table.Table
            A table containing the data for the batch.

        Returns:
        --------
        tuple
            A tuple containing the input data as features and the corresponding labels.
        """
        # Retrieve the telescope images and store in the features dictionary
        labels = {}
        features = {"input": batch["features"].data}
        if "type" in self.tasks:
            labels["type"] = to_categorical(
                batch["true_shower_primary_class"].data,
                num_classes=2,
            )
            # Temp fix till keras support class weights for multiple outputs or I wrote custom loss
            # https://github.com/keras-team/keras/issues/11735
            if len(self.tasks) == 1:
                labels = to_categorical(
                    batch["true_shower_primary_class"].data,
                    num_classes=2,
                )
        if "energy" in self.tasks:
            labels["energy"] = batch["log_true_energy"].data
        if "skydirection" in self.tasks:
            labels["skydirection"] = np.stack(
                (
                    batch["fov_lon"].data,
                    batch["fov_lat"].data,
                ),
                axis=1,
            )
        if "cameradirection" in self.tasks:
            labels["cameradirection"] = np.stack(
                (
                    batch["cam_coord_offset_x"].data,
                    batch["cam_coord_offset_y"].data,
                ),
                axis=1,
            )
        # Temp fix for supporting keras2 & keras3
        if int(keras.__version__.split(".")[0]) >= 3:
            features = features["input"]
        return features, labels

    def _get_stereo_item(self, batch):
        """
        Retrieve the features and labels for one batch of stereoscopic data.

        This method is called to retrieve the features and labels for one batch of
        stereoscopic data. The original batch is grouped to retrieve the telescope
        data for each event and then the telescope images or waveforms are stored
        by the hillas intensity or stacked if required. Feature vectors can also
        be retrieved if available for ``telescope``- and ``subarray``level. The
        labels are set up based on the tasks specified.

        Parameters:
        -----------
        batch : astropy.table.Table
            A table containing the data for the batch.

        Returns:
        --------
        tuple
            A tuple containing the input data as features and the corresponding labels.
        """
        labels = {}
        if self.DLDataReader.process_type == ProcessType.Simulation:
            batch_grouped = batch.group_by(
                ["obs_id", "event_id", "tel_type_id", "true_shower_primary_class"]
            )
        elif self.DLDataReader.process_type == ProcessType.Observation:
            batch_grouped = batch.group_by(["obs_id", "event_id", "tel_type_id"])
        features, mono_feature_vectors, stereo_feature_vectors = [], [], []
        true_shower_primary_class = []
        log_true_energy = []
        fov_lon, fov_lat, angular_separation = [], [], []
        cam_coord_offset_x, cam_coord_offset_y, cam_coord_distance = [], [], []
        for group_element in batch_grouped.groups:
            if "features" in batch.colnames:
                if self.sort_by_intensity:
                    # Sort images by the hillas intensity in a given batch if requested
                    group_element.sort(["hillas_intensity"], reverse=True)
                # Stack the telescope images for stereo analysis
                if self.stack_telescope_images:
                    # Retrieve the telescope images
                    plain_features = group_element["features"].data
                    # Stack the telescope images along the last axis
                    stacked_features = np.concatenate(
                        [plain_features[i] for i in range(plain_features.shape[0])],
                        axis=-1,
                    )
                    # Append the stacked images to the features list
                    # shape: (batch_size, image_shape, image_shape, n_channels * n_tel)
                    features.append(stacked_features)
                else:
                    # Append the plain images to the features list
                    # shape: (batch_size, n_tel, image_shape, image_shape, n_channels)
                    features.append(group_element["features"].data)
            # Retrieve the feature vectors
            if "mono_feature_vectors" in batch.colnames:
                mono_feature_vectors.append(group_element["mono_feature_vectors"].data)
            if "stereo_feature_vectors" in batch.colnames:
                stereo_feature_vectors.append(
                    group_element["stereo_feature_vectors"].data
                )
            # Retrieve the labels for the tasks
            # FIXME: This won't work for divergent pointing directions
            if "type" in self.tasks:
                true_shower_primary_class.append(
                    group_element["true_shower_primary_class"].data[0]
                )
            if "energy" in self.tasks:
                log_true_energy.append(group_element["log_true_energy"].data[0])
            if "skydirection" in self.tasks:
                fov_lon.append(group_element["fov_lon"].data[0])
                fov_lat.append(
                    group_element["fov_lat"].data[0]
                )
            if "cameradirection" in self.tasks:
                cam_coord_offset_x.append(group_element["cam_coord_offset_x"].data)
                cam_coord_offset_y.append(
                    group_element["cam_coord_offset_y"].data
                )
        # Store the labels in the labels dictionary
        if "type" in self.tasks:
            labels["type"] = to_categorical(
                np.array(true_shower_primary_class),
                num_classes=2,
            )
            # Temp fix till keras support class weights for multiple outputs or I wrote custom loss
            # https://github.com/keras-team/keras/issues/11735
            if len(self.tasks) == 1:
                labels = to_categorical(
                    np.array(true_shower_primary_class),
                    num_classes=2,
                )
        if "energy" in self.tasks:
            labels["energy"] = np.array(log_true_energy)
        if "skydirection" in self.tasks:
            labels["skydirection"] = np.stack(
                (
                    np.array(fov_lon),
                    np.array(fov_lat),
                ),
                axis=1,
            )
        if "cameradirection" in self.tasks:
            labels["cameradirection"] = np.stack(
                (
                    np.array(cam_coord_offset_x),
                    np.array(cam_coord_offset_y),
                ),
                axis=1,
            )
        # Store the fatures in the features dictionary
        if "features" in batch.colnames:
            features = {"input": np.array(features)}
        # TDOO: Add support for both feature vectors
        if "mono_feature_vectors" in batch.colnames:
            features = {"input": np.array(mono_feature_vectors)}
        if "stereo_feature_vectors" in batch.colnames:
            features = {"input": np.array(stereo_feature_vectors)}
        # Temp fix for supporting keras2 & keras3
        if int(keras.__version__.split(".")[0]) >= 3:
            features = features["input"]
        return features, labels
