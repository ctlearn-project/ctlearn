"""
Base Data Loader Module

This module provides an abstract base class for data loaders in CTLearn.
It defines the common interface and initialization logic for framework-specific
data loaders (Keras, PyTorch) while handling telescope data in both mono and
stereo observation modes.

Classes:
    BaseDLDataLoader: Abstract base class for deep learning data loaders
"""

from abc import ABC, abstractmethod
import numpy as np
import random
import cv2
from ctlearn.core.ctlearn_enum import Task

class BaseDLDataLoader(ABC):
    """
    Abstract base class for deep learning data loaders.
    
    This class provides the common interface and initialization logic for loading
    and processing Cherenkov telescope data. It handles both mono (single telescope)
    and stereo (multiple telescopes) observation modes and supports various data
    processing options like sorting by intensity and stacking telescope images.
    
    Attributes:
        DLDataReader: The data reader instance for accessing telescope event data
        indices (list): List of event indices to load from the dataset
        tasks (list): List of tasks to perform (e.g., classification, energy, direction)
        batch_size (int): Number of batches per epoch
        random_seed (int or None): Random seed for reproducibility
        stack_telescope_images (bool): Whether to stack images from multiple telescopes
        sort_by_intensity (bool): Whether to sort telescope images by Hillas intensity
        input_shape (tuple): Shape of input data (height, width, channels)
    
    Methods:
        __len__: Abstract method to return the number of batches per epoch
        __getitem__: Abstract method to generate one batch of data
        on_epoch_end: Abstract method called at the end of each epoch
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
        """
        Initialize the base data loader.
        
        Sets up the data loader with a data reader, configures batch processing
        parameters, and determines the input shape based on observation mode
        (mono vs stereo) and image stacking options.
        
        Args:
            DLDataReader: Instance of a data reader class (e.g., DLHDFDataReader)
                that provides access to telescope event data
            indices (list): List of integer indices specifying which events to load
                from the dataset
            tasks (list): List of Task enum values specifying which tasks to perform
                (e.g., [Task.type, Task.energy, Task.cameradirection])
            batch_size (int, optional): Number of samples to include in each batch.
                Defaults to 64
            random_seed (int or None, optional): Seed for random number generator
                to ensure reproducibility. If None, random behavior is not deterministic.
                Defaults to None
            sort_by_intensity (bool, optional): If True, sort telescope images by
                their Hillas intensity in descending order. Useful for stereo analysis.
                Defaults to False
            stack_telescope_images (bool, optional): If True, stack images from multiple
                telescopes along the channel dimension. Only applicable in stereo mode.
                Defaults to False
            **kwargs: Additional keyword arguments passed to parent classes
        """
        super().__init__()
        
        # Store initialization parameters
        self.DLDataReader = DLDataReader
        self.indices = indices
        self.tasks = tasks
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.stack_telescope_images = stack_telescope_images
        self.sort_by_intensity = sort_by_intensity

        # Get parent configuration if available
        parent = getattr(self.DLDataReader, "parent", None)
        
        # Helper to get attributes with fallbacks
        def get_val(name, default):
            if parent is not None and hasattr(parent, name):
                return getattr(parent, name)
            if name in kwargs:
                return kwargs[name]
            if "parameters" in kwargs and isinstance(kwargs["parameters"], dict):
                params = kwargs["parameters"]
                if name == "use_augmentation":
                    return params.get("augmentation", {}).get("use_augmentation", default)
                elif name == "aug_prob":
                    return params.get("augmentation", {}).get("aug_prob", default)
                elif name == "rot_prob":
                    return params.get("augmentation", {}).get("rot_prob", default)
                elif name == "trans_prob":
                    return params.get("augmentation", {}).get("trans_prob", default)
                elif name == "flip_hor_prob":
                    return params.get("augmentation", {}).get("flip_hor_prob", default)
                elif name == "flip_ver_prob":
                    return params.get("augmentation", {}).get("flip_ver_prob", default)
                elif name == "mask_prob":
                    return params.get("augmentation", {}).get("mask_prob", default)
                elif name == "mask_dvr_prob":
                    return params.get("augmentation", {}).get("mask_dvr_prob", default)
                elif name == "noise_prob":
                    return params.get("augmentation", {}).get("noise_prob", default)
                elif name == "max_rot":
                    return params.get("augmentation", {}).get("max_rot", default)
                elif name == "max_trans":
                    return params.get("augmentation", {}).get("max_trans", default)
                elif name == "apply_log_scaling":
                    return params.get("normalization", {}).get("apply_log_scaling", default)
                elif name == "use_clean":
                    return params.get("normalization", {}).get("use_clean", default)
                elif name == "use_clean_dvr":
                    return params.get("normalization", {}).get("use_clean_dvr", default)
                elif name == "type_mu":
                    return params.get("normalization", {}).get("type_mu", default)
                elif name == "type_sigma":
                    return params.get("normalization", {}).get("type_sigma", default)
                elif name == "dir_mu":
                    return params.get("normalization", {}).get("dir_mu", default)
                elif name == "dir_sigma":
                    return params.get("normalization", {}).get("dir_sigma", default)
                elif name == "energy_mu":
                    return params.get("normalization", {}).get("energy_mu", default)
                elif name == "energy_sigma":
                    return params.get("normalization", {}).get("energy_sigma", default)
                elif name == "leakage_intensity_cutoff":
                    return params.get("cut-off", {}).get("leakage_intensity", default)
                elif name == "intensity_cutoff":
                    return params.get("cut-off", {}).get("intensity", default)
            return default

        # Initialize pre-processing & augmentation options
        self.use_augmentation = get_val("use_augmentation", False)
        self.aug_prob = get_val("aug_prob", 0.5)
        self.rot_prob = get_val("rot_prob", 0.5)
        self.trans_prob = get_val("trans_prob", 0.5)
        self.flip_hor_prob = get_val("flip_hor_prob", 0.5)
        self.flip_ver_prob = get_val("flip_ver_prob", 0.5)
        self.mask_prob = get_val("mask_prob", 0.5)
        self.mask_dvr_prob = get_val("mask_dvr_prob", 0.5)
        self.noise_prob = get_val("noise_prob", 0.5)
        self.max_rot = get_val("max_rot", 5.0)
        self.max_trans = get_val("max_trans", 10.0)

        self.apply_log_scaling = get_val("apply_log_scaling", [True, True])
        self.use_clean = get_val("use_clean", True)
        self.use_clean_dvr = get_val("use_clean_dvr", False)
        
        self.type_mu = get_val("type_mu", 0.0)
        self.type_sigma = get_val("type_sigma", 1.0)
        self.dir_mu = get_val("dir_mu", 0.0)
        self.dir_sigma = get_val("dir_sigma", 1.0)
        self.energy_mu = get_val("energy_mu", 0.0)
        self.energy_sigma = get_val("energy_sigma", 1.0)

        self.leakage_intensity_cutoff = get_val("leakage_intensity_cutoff", 0.2)
        self.intensity_cutoff = get_val("intensity_cutoff", 50.0)

        # Determine input shape based on reader type and observation mode
        # Feature vector readers don't have spatial dimensions
        if self.DLDataReader.__class__.__name__ != "DLFeatureVectorReader":
            
            # Mono mode: single telescope per event
            if self.DLDataReader.mode == "mono":
                # Use the input shape directly from the data reader
                self.input_shape = self.DLDataReader.input_shape
                
            # Stereo mode: multiple telescopes per event
            elif self.DLDataReader.mode == "stereo":
                # Get input shape from the first selected telescope
                # All telescopes are assumed to have the same image dimensions
                self.input_shape = self.DLDataReader.input_shape[
                    list(self.DLDataReader.selected_telescopes)[0]
                ]
                
                # Modify input shape if stacking telescope images
                # Original shape: (num_telescopes, height, width, channels)
                # Stacked shape: (height, width, num_telescopes * channels)
                if self.stack_telescope_images:
                    self.input_shape = (
                        self.input_shape[1],  # height
                        self.input_shape[2],  # width
                        self.input_shape[0] * self.input_shape[3],  # stacked channels
                    )

    def clean_data(self, image, peak_time):
        # Remove negative numbers and avoid inf or nans
        image[image < 0] = 0
        peak_time[peak_time < 0] = 0
        image[np.isnan(image)] = 0
        image[np.isinf(image)] = 0
        peak_time[np.isnan(peak_time)] = 0
        peak_time[np.isinf(peak_time)] = 0
        return image, peak_time

    def normalize_data(self, image, peak_time, task):
        # Normalization
        if task == Task.type or task == "type":
            image = (image - self.type_mu) / self.type_sigma
            peak_time = (peak_time - self.type_mu) / self.type_sigma
        elif task == Task.energy or task == "energy":
            image = (image - self.energy_mu) / self.energy_sigma
            peak_time = (peak_time - self.energy_mu) / self.energy_sigma
        elif task in [Task.cameradirection, Task.skydirection, Task.direction, "cameradirection", "skydirection", "direction"]:
            image = (image - self.dir_mu) / self.dir_sigma
            peak_time = (peak_time - self.dir_mu) / self.dir_sigma

        return image, peak_time

    def apply_log_scaling_to_channels(self, image, peak_time):
        if self.apply_log_scaling[0]:
            image = image.astype(np.float32)
            image = np.log10(image + 1.0)
        if self.apply_log_scaling[1]:
            peak_time = peak_time.astype(np.float32)
            peak_time = np.log10(peak_time + 1.0)
        return image, peak_time

    def apply_augmentation(self, image, peak_time, task):
        for id_batch in range(image.shape[0]):
            random_aug = random.random()

            if random_aug > self.aug_prob:
                if task not in [Task.cameradirection, Task.skydirection, Task.direction, "cameradirection", "skydirection", "direction"]:
                    random_aug_flip_ver = random.random()
                    if random_aug_flip_ver > self.flip_ver_prob:
                        image[id_batch] = np.expand_dims(cv2.flip(image[id_batch].astype(np.float32), 0), axis=-1)
                        peak_time[id_batch] = np.expand_dims(cv2.flip(peak_time[id_batch].astype(np.float32), 0), axis=-1)
                        continue
                    random_aug_flip_hor = random.random()
                    if random_aug_flip_hor > self.flip_hor_prob:
                        image[id_batch] = np.expand_dims(cv2.flip(image[id_batch].astype(np.float32), 1), axis=-1)
                        peak_time[id_batch] = np.expand_dims(cv2.flip(peak_time[id_batch].astype(np.float32), 1), axis=-1)
                        continue
                    random_aug_rot = random.random()
                    if random_aug_rot > self.rot_prob:
                        (h, w) = image[id_batch].shape[:2]
                        angle = random.uniform(-self.max_rot, self.max_rot)
                        scale = 1.0
                        center = (w // 2, h // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
                        image[id_batch] = np.expand_dims(cv2.warpAffine(
                            image[id_batch].astype(np.float32), rotation_matrix, (w, h)
                        ), axis=-1)
                        peak_time[id_batch] = np.expand_dims(cv2.warpAffine(
                            peak_time[id_batch].astype(np.float32), rotation_matrix, (w, h)
                        ), axis=-1)
                        continue
                    random_aug_trans = random.random()
                    if random_aug_trans > self.trans_prob:
                        (h, w) = image[id_batch].shape[:2]
                        tx = random.uniform(-self.max_trans, self.max_trans)
                        ty = random.uniform(-self.max_trans, self.max_trans)
                        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
                        image[id_batch] = np.expand_dims(cv2.warpAffine(
                            image[id_batch].astype(np.float32), translation_matrix, (w, h)
                        ), axis=-1)
                        peak_time[id_batch] = np.expand_dims(cv2.warpAffine(
                            peak_time[id_batch].astype(np.float32), translation_matrix, (w, h)
                        ), axis=-1)
                        continue
        return image, peak_time

    @abstractmethod
    def __len__(self):
        """
        Get the number of batches per epoch.
        
        This method must be implemented by subclasses to return the total number
        of batches that will be generated in one epoch, typically calculated as
        ceil(total_samples / batch_size).
        
        Returns:
            int: Number of batches per epoch
        """
        pass

    @abstractmethod
    def __getitem__(self, index):
        """
        Generate one batch of data.
        
        This method must be implemented by subclasses to return a single batch
        of data at the specified index. The batch should contain input features
        and corresponding labels formatted appropriately for the framework.
        
        Args:
            index (int): Index of the batch to generate (0 to len(self) - 1)
            
        Returns:
            tuple: A tuple containing:
                - features (dict or array): Input features for the batch
                - labels (dict or array): Ground truth labels for the batch
                - metadata (optional): Additional information about the batch
        """
        pass

    @abstractmethod
    def on_epoch_end(self):
        """
        Perform operations at the end of each epoch.
        
        This method must be implemented by subclasses to perform any necessary
        cleanup or updates at the end of an epoch. Common operations include
        shuffling indices for the next epoch or updating internal state.
        
        This method is typically called automatically by the training framework
        after processing all batches in an epoch.
        """
        pass
