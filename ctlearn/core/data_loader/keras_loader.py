"""
Keras Data Loader Module

This module provides a Keras-specific data loader implementation for CTLearn.
It extends both Keras Sequence and the base data loader to provide efficient
batch generation for training with Keras/TensorFlow models.

Classes:
    KerasDLDataLoader: Keras Sequence for loading and preprocessing telescope data
"""

import numpy as np
import keras
from keras.utils import Sequence, to_categorical
from .base_loader import BaseDLDataLoader

from dl1_data_handler.reader import ProcessType

class KerasDLDataLoader(Sequence, BaseDLDataLoader):
    """
    Keras data loader for Cherenkov telescope data.
    
    This class implements the Keras Sequence interface to provide efficient
    data loading for training. It supports both monoscopic (single telescope)
    and stereoscopic (multiple telescopes) observation modes, with options
    for image stacking and intensity-based sorting.
    
    Inherits from:
        Sequence: Keras sequence for thread-safe batch generation
        BaseDLDataLoader: Base class providing common data loading functionality
    
    Attributes:
        Inherited from BaseDLDataLoader including:
        - DLDataReader: Data reader for accessing telescope events
        - indices: Array of event indices to process
        - tasks: List of tasks (type, energy, direction)
        - batch_size: Number of samples per batch
        - random_seed: Seed for shuffling
        - sort_by_intensity: Whether to sort by Hillas intensity
        - stack_telescope_images: Whether to stack images from multiple telescopes
    """
    
    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialize the Keras data loader.
        
        This constructor initializes the data loader by calling the parent
        class constructors and performing initial index shuffling if a
        random seed is provided.
        
        Args:
            **kwargs: Keyword arguments passed to BaseDLDataLoader, including:
                - DLDataReader: Data reader instance
                - indices: Event indices to load
                - tasks: List of tasks to perform
                - batch_size: Batch size
                - random_seed: Random seed for shuffling
                - sort_by_intensity: Whether to sort by intensity
                - stack_telescope_images: Whether to stack images
        """
        super().__init__(**kwargs)
        # Perform initial shuffling of indices if random seed is set
        self.on_epoch_end()

    def __len__(self):
        """
        Get the number of batches per epoch.

        This method calculates the number of batches required to cover the entire dataset
        based on the batch size. Uses floor division to ensure complete batches only.

        Returns:
            int: Number of batches per epoch (total_samples // batch_size)
            
        Note:
            Samples that don't fit into a complete batch are not included in the epoch.
            For example, with 100 samples and batch_size=32, this returns 3 (96 samples used).
        """
        return int(np.floor(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        """
        Update indices after each epoch.
        
        This method is called automatically by Keras at the end of each epoch.
        If a random seed is provided, it shuffles the indices to provide different
        batch compositions in each epoch, which can improve training convergence.
        
        The shuffling is deterministic (uses the same seed each time) to ensure
        reproducibility while still providing epoch-to-epoch variation.
        
        Side Effects:
            - If random_seed is not None: Shuffles self.indices in-place using
              the configured random seed
            - If random_seed is None: No operation performed
        """
        if self.random_seed is not None:
            # Set random seed for reproducibility
            np.random.seed(self.random_seed)
            # Shuffle indices in-place to randomize batch composition
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        """
        Generate one batch of data.

        This method is called by Keras to generate one batch of data based on
        the provided index. It delegates to mode-specific methods (_get_mono_item
        or _get_stereo_item) depending on the observation mode.

        Args:
            index (int): Index of the batch to generate (0 to len(self) - 1)

        Returns:
            tuple: (features, labels) where:
                - features: Input data formatted for the model
                    * Keras 2: dict with 'input' key
                    * Keras 3: numpy array directly
                    * Shape varies by mode and configuration
                - labels: Ground truth labels as dict or array
                    * Dict keys depend on tasks (type, energy, skydirection, cameradirection)
                    * For single task classification: categorical array instead of dict
                    
        Note:
            The batch indices are computed as:
            batch_indices = self.indices[index * batch_size : (index + 1) * batch_size]
        """
        # Generate indices of the batch
        batch_indices = self.indices[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        features, labels = None, None
        
        # Generate batch based on observation mode
        if self.DLDataReader.mode == "mono":
            # Monoscopic mode: single telescope per event
            batch = self.DLDataReader.generate_mono_batch(batch_indices)
            features, labels = self._get_mono_item(batch)
        elif self.DLDataReader.mode == "stereo":
            # Stereoscopic mode: multiple telescopes per event
            batch = self.DLDataReader.generate_stereo_batch(batch_indices)
            features, labels = self._get_stereo_item(batch)
        return features, labels

    def _get_mono_item(self, batch):
        """
        Retrieve features and labels for one batch of monoscopic data.

        This method extracts and formats data from a single-telescope batch,
        preparing features and task-specific labels for training or inference.

        Args:
            batch (astropy.table.Table): Table containing monoscopic event data
                Expected columns include:
                - features: Telescope images or feature vectors
                - true_shower_primary_class: Particle type (0=gamma, 1=proton)
                - log_true_energy: Logarithm of true energy
                - fov_lon, fov_lat: Sky direction in field-of-view coordinates
                - cam_coord_offset_x, cam_coord_offset_y: Camera coordinate offsets

        Returns:
            tuple: (features, labels) where:
                - features: Input features formatted for Keras
                    * Keras 2: dict with 'input' key containing numpy array
                    * Keras 3: numpy array directly
                    * Shape: (batch_size, height, width, channels)
                - labels: Task-specific labels
                    * If only 'type' task: categorical array (batch_size, 2)
                    * Otherwise: dict with keys matching self.tasks:
                        - 'type': one-hot encoded particle type
                        - 'energy': log energy values
                        - 'skydirection': (lon, lat) coordinates
                        - 'cameradirection': (offset_x, offset_y) coordinates
        """
        # Initialize labels dictionary
        labels = {}
        
        # Retrieve telescope images and store in features dictionary
        features = {"input": batch["features"].data}
        
        image = features["input"][..., 0:1]
        peak_time = features["input"][..., 1:2]
        
        active_task = self.tasks[0] if self.tasks else None
        image, peak_time = self.clean_and_normalize(image, peak_time, active_task)
        
        if self.use_augmentation:
            image, peak_time = self.apply_augmentation(image, peak_time, active_task)
            
        image, peak_time = self.apply_log_scaling_to_channels(image, peak_time)
        features["input"] = np.concatenate([image, peak_time], axis=-1)
        
        # Extract particle type classification labels
        if "type" in self.tasks:
            # Convert to one-hot encoding (0=gamma, 1=proton)
            labels["type"] = to_categorical(
                batch["true_shower_primary_class"].data,
                num_classes=2,
            )
            # Temporary fix: Use array instead of dict for single-task classification
            # Required until Keras fully supports class weights for multiple outputs
            # See: https://github.com/keras-team/keras/issues/11735
            if len(self.tasks) == 1:
                labels = to_categorical(
                    batch["true_shower_primary_class"].data,
                    num_classes=2,
                )
        
        # Extract energy regression labels
        if "energy" in self.tasks:
            # Energy is already in log scale
            labels["energy"] = batch["log_true_energy"].data
        
        # Extract sky direction reconstruction labels
        if "skydirection" in self.tasks:
            # Stack longitude and latitude into single array
            labels["skydirection"] = np.stack(
                (
                    batch["fov_lon"].data,
                    batch["fov_lat"].data,
                ),
                axis=1,
            )
        
        # Extract camera direction reconstruction labels
        if "cameradirection" in self.tasks:
            # Stack camera x and y offsets into single array
            labels["cameradirection"] = np.stack(
                (
                    batch["cam_coord_offset_x"].data,
                    batch["cam_coord_offset_y"].data,
                ),
                axis=1,
            )
        
        # Temporary fix for Keras 2/3 compatibility
        # Keras 3 expects arrays directly, not wrapped in dict
        if int(keras.__version__.split(".")[0]) >= 3:
            features = features["input"]
        
        return features, labels

    def _get_stereo_item(self, batch):
        """
        Retrieve features and labels for one batch of stereoscopic data.

        This method processes multi-telescope events, grouping telescope data
        by event, optionally sorting by intensity, and stacking images if requested.
        It also handles both telescope-level and subarray-level feature vectors.

        Args:
            batch (astropy.table.Table): Table containing stereoscopic event data
                Expected columns include all mono columns plus:
                - obs_id: Observation run identifier
                - event_id: Event identifier within observation
                - tel_type_id: Telescope type identifier
                - hillas_intensity: For sorting (if sort_by_intensity=True)
                - mono_feature_vectors: Telescope-level features (optional)
                - stereo_feature_vectors: Subarray-level features (optional)

        Returns:
            tuple: (features, labels) where:
                - features: Input features formatted for Keras
                    * If stacked images: (batch_size, height, width, n_channels * n_tel)
                    * If unstacked: (batch_size, n_tel, height, width, n_channels)
                    * Feature vectors have different shapes
                - labels: Task-specific labels (same structure as _get_mono_item)
                
        Note:
            Events are grouped by (obs_id, event_id, tel_type_id) for simulations,
            or by (obs_id, event_id, tel_type_id) for observations.
            Labels are extracted from the first telescope in each group since they
            are event-level quantities (same for all telescopes in an event).
        """
        # Initialize labels dictionary
        labels = {}
        
        # Group batch by event to collect all telescopes for each event
        if self.DLDataReader.process_type == ProcessType.Simulation:
            # For simulations, group by observation, event, telescope type, and particle class
            batch_grouped = batch.group_by(
                ["obs_id", "event_id", "tel_type_id", "true_shower_primary_class"]
            )
        elif self.DLDataReader.process_type == ProcessType.Observation:
            # For real observations, particle class is unknown
            batch_grouped = batch.group_by(["obs_id", "event_id", "tel_type_id"])
        
        # Initialize lists for collecting event-level data
        features, mono_feature_vectors, stereo_feature_vectors = [], [], []
        true_shower_primary_class = []
        log_true_energy = []
        fov_lon, fov_lat, angular_separation = [], [], []
        cam_coord_offset_x, cam_coord_offset_y, cam_coord_distance = [], [], []
        
        # Process each event group
        for group_element in batch_grouped.groups:
            # Process telescope images if available
            if "features" in batch.colnames:
                # Sort telescopes by Hillas intensity if requested
                if self.sort_by_intensity:
                    # Sort in descending order (brightest first)
                    group_element.sort(["hillas_intensity"], reverse=True)
                
                # Stack telescope images for stereo analysis if requested
                if self.stack_telescope_images:
                    # Retrieve telescope images for this event
                    plain_features = group_element["features"].data
                    # Concatenate along channel axis: (h, w, c*n_tel)
                    stacked_features = np.concatenate(
                        [plain_features[i] for i in range(plain_features.shape[0])],
                        axis=-1,
                    )
                    # Append stacked images
                    # Shape: (height, width, n_channels * n_telescopes)
                    features.append(stacked_features)
                else:
                    # Keep telescopes as separate dimension
                    # Shape: (n_telescopes, height, width, n_channels)
                    features.append(group_element["features"].data)
            
            # Retrieve telescope-level feature vectors if available
            if "mono_feature_vectors" in batch.colnames:
                mono_feature_vectors.append(group_element["mono_feature_vectors"].data)
            
            # Retrieve subarray-level feature vectors if available
            if "stereo_feature_vectors" in batch.colnames:
                stereo_feature_vectors.append(
                    group_element["stereo_feature_vectors"].data
                )
            
            # Extract event-level labels (same for all telescopes in event)
            # FIXME: This won't work correctly for divergent pointing directions
            # where different telescopes point in different directions
            
            # Particle type classification
            if "type" in self.tasks:
                # Use first telescope's value (same for all telescopes in event)
                true_shower_primary_class.append(
                    group_element["true_shower_primary_class"].data[0]
                )
            
            # Energy regression
            if "energy" in self.tasks:
                log_true_energy.append(group_element["log_true_energy"].data[0])
            
            # Sky direction reconstruction
            if "skydirection" in self.tasks:
                fov_lon.append(group_element["fov_lon"].data[0])
                fov_lat.append(group_element["fov_lat"].data[0])
            
            # Camera direction reconstruction
            if "cameradirection" in self.tasks:
                cam_coord_offset_x.append(group_element["cam_coord_offset_x"].data)
                cam_coord_offset_y.append(group_element["cam_coord_offset_y"].data)
        
        # Format labels for each task
        if "type" in self.tasks:
            # Convert to one-hot encoding
            labels["type"] = to_categorical(
                np.array(true_shower_primary_class),
                num_classes=2,
            )
            # Temporary fix for single-task classification
            if len(self.tasks) == 1:
                labels = to_categorical(
                    np.array(true_shower_primary_class),
                    num_classes=2,
                )
        
        if "energy" in self.tasks:
            labels["energy"] = np.array(log_true_energy)
        
        if "skydirection" in self.tasks:
            # Stack longitude and latitude
            labels["skydirection"] = np.stack(
                (
                    np.array(fov_lon),
                    np.array(fov_lat),
                ),
                axis=1,
            )
        
        if "cameradirection" in self.tasks:
            # Stack camera coordinate offsets
            labels["cameradirection"] = np.stack(
                (
                    np.array(cam_coord_offset_x),
                    np.array(cam_coord_offset_y),
                ),
                axis=1,
            )
        
        # Format features based on available data type
        if "features" in batch.colnames:
            # Telescope images
            features = {"input": np.array(features)}
            
            features_arr = features["input"]
            active_task = self.tasks[0] if self.tasks else None

            # Slicing image and peak_time based on dimensionality
            if len(features_arr.shape) == 5: # Unstacked mode: (batch, tel, height, width, channels)
                image = features_arr[..., 0]
                peak_time = features_arr[..., 1]
                
                image, peak_time = self.clean_and_normalize(image, peak_time, active_task)
                image, peak_time = self.apply_log_scaling_to_channels(image, peak_time)
                
                features["input"] = np.stack([image, peak_time], axis=-1)
            else: # Stacked mode: (batch, height, width, channels)
                image = features_arr[..., ::2]
                peak_time = features_arr[..., 1::2]
                
                image, peak_time = self.clean_and_normalize(image, peak_time, active_task)
                image, peak_time = self.apply_log_scaling_to_channels(image, peak_time)
                
                # Re-stack alternating channels
                stacked = []
                for i in range(image.shape[-1]):
                    stacked.append(image[..., i:i+1])
                    stacked.append(peak_time[..., i:i+1])
                features["input"] = np.concatenate(stacked, axis=-1)
                
        # TODO: Add support for using both mono and stereo feature vectors simultaneously
        if "mono_feature_vectors" in batch.colnames:
            # Telescope-level feature vectors
            features = {"input": np.array(mono_feature_vectors)}
        if "stereo_feature_vectors" in batch.colnames:
            # Subarray-level feature vectors
            features = {"input": np.array(stereo_feature_vectors)}
        
        # Temporary fix for Keras 2/3 compatibility
        if int(keras.__version__.split(".")[0]) >= 3:
            features = features["input"]
        
        return features, labels
