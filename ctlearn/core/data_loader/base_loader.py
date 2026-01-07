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
        batch_size (int): Number of samples per batch
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
        super().__init__(**kwargs)
        
        # Store initialization parameters
        self.DLDataReader = DLDataReader
        self.indices = indices
        self.tasks = tasks
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.stack_telescope_images = stack_telescope_images
        self.sort_by_intensity = sort_by_intensity

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
