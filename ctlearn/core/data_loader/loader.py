"""
Data Loader Factory Module

This module provides a factory class for creating framework-specific data loaders
for CTLearn. It supports both Keras and PyTorch frameworks and handles dynamic
imports to avoid unnecessary dependencies.

Classes:
    DLDataLoader: Factory class for creating data loaders based on the specified framework
"""

# from .keras_loader import KerasDLDataLoader
# from .pytorch_loader import PyTorchDLDataLoader

class DLDataLoader:
    """
    Factory class for creating deep learning data loaders.
    
    This class provides a static factory method to instantiate the appropriate
    data loader based on the specified framework (Keras or PyTorch). It uses
    dynamic imports to load only the required framework dependencies.
    
    Methods:
        create: Static factory method to create framework-specific data loaders
    """
    
    @staticmethod
    def create(framework, **kwargs):
        """
        Create a data loader for the specified framework.
        
        This factory method instantiates the appropriate data loader class based
        on the framework parameter. It dynamically imports the required loader
        class to avoid loading unnecessary dependencies for unused frameworks.
        
        Args:
            framework (str): The framework to use ('keras' or 'pytorch')
            **kwargs: Additional keyword arguments passed to the data loader constructor
                These may include:
                - config: Configuration dictionary
                - mode: Operation mode (train, validation, test, prediction)
                - data_files: List of input data files
                - batch_size: Number of samples per batch
                - num_workers: Number of worker processes for data loading
                
        Returns:
            KerasDLDataLoader or PyTorchDLDataLoader: The instantiated data loader
                for the specified framework
                
        Raises:
            ValueError: If the framework is not 'keras' or 'pytorch'
            ImportError: If the required framework-specific loader cannot be imported
            
        Examples:
            >>> # Create a PyTorch data loader
            >>> loader = DLDataLoader.create('pytorch', config=config, mode='train')
            
            >>> # Create a Keras data loader
            >>> loader = DLDataLoader.create('keras', config=config, mode='validation')
        """
        # Initialize dataloader variable
        dataloader = None 
        
        # Create Keras data loader
        if framework == "keras":
            try:
                # Dynamically import Keras loader to avoid unnecessary dependencies
                from .keras_loader import KerasDLDataLoader
                dataloader = KerasDLDataLoader(**kwargs)
            except ImportError as e:
                # Raise informative error if Keras dependencies are missing
                raise ImportError(f"Not possible to import KerasDLDataLoader: {e}") from e
             
        # Create PyTorch data loader
        elif framework == "pytorch":
            try:
                # Dynamically import PyTorch loader to avoid unnecessary dependencies
                from .pytorch_loader import PyTorchDLDataLoader
                dataloader = PyTorchDLDataLoader(**kwargs)
            except ImportError as e:
                # Raise informative error if PyTorch dependencies are missing
                raise ImportError(f"Not possible to import PyTorchDLDataLoader: {e}") from e
 
        # Handle unsupported framework
        else:
            raise ValueError(f"Unsupported framework: {framework}")
        
        return dataloader
