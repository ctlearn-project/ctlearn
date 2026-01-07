"""
PyTorch Neural Network Utilities Module

This module provides utility functions and helper classes for creating, managing,
and manipulating PyTorch neural network models in CTLearn. It includes functionality
for model creation, checkpoint management, visualization, and model export.

Functions:
    create_model: Factory function for instantiating models from configuration
    
Classes:
    ModelHelper: Collection of static utility methods for model operations
"""

import importlib
import torch
import numpy as np
import os.path
import pickle
from matplotlib import pyplot as plt
from skimage.filters import gabor_kernel
from skimage.transform import resize
import onnx
from onnxsim import simplify
import warnings
from ctlearn.core.ctlearn_enum import Task, Mode 

#-------------------------------------------------------------------------------------------------------------------
def create_model(model_parameters):
    """
    Factory function to dynamically create and instantiate a model.
    
    This function uses Python's importlib to dynamically load a model class
    based on the configuration parameters and instantiate it with the provided
    parameters. This allows for flexible model selection without hardcoded imports.
    
    Args:
        model_parameters (dict): Dictionary containing model configuration
            Required keys:
            - 'model_name' (str): Name of the model class to instantiate
              Example: 'ResNet', 'EfficientNet', 'ViT'
            - 'parameters' (dict): Dictionary of parameters to pass to model constructor
              Example: {'num_outputs': 2, 'input_channels': 1, 'depth': 50}
    
    Returns:
        torch.nn.Module: Instantiated model ready for training or inference
    
    Raises:
        ValueError: If the model class is not found in the models module
        ValueError: If there's an error instantiating the model (wrong parameters)
        RuntimeError: If an unexpected error occurs during model creation
        
    Example:
        >>> config = {
        ...     'model_name': 'ResNet',
        ...     'parameters': {
        ...         'num_outputs': 2,
        ...         'input_channels': 1,
        ...         'depth': 50
        ...     }
        ... }
        >>> model = create_model(config)
        >>> print(type(model))
        <class 'ctlearn.core.pytorch.nets.models.ResNet.ResNet'>
        
    Notes:
        - All models must be located in ctlearn.core.pytorch.nets.models
        - Model class name must match the module name
        - Models should inherit from torch.nn.Module
    """
    try:
        # Define the base module path for models
        module_name = "ctlearn.core.pytorch.nets.models"
        model_type = model_parameters["model_name"]
        model_params = model_parameters["parameters"]

        # Construct full class path (module.ModelClass)
        full_class_path = f"ctlearn.core.pytorch.nets.models.{model_type}"

        # Dynamically import the model module
        module = importlib.import_module(full_class_path)
        # Get the module (intermediate step)
        module = getattr(module, model_type)
        # Get the actual model class from the module
        model_class = getattr(module, model_type)
        
        # Instantiate the model with the provided parameters
        model_net = model_class(**model_params)
        return model_net

    except AttributeError:
        raise ValueError(f"Model class {model_type} not found in module {module_name}.")
    except TypeError as e:
        raise ValueError(f"Error instantiating model {model_type}: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {str(e)}")
#-------------------------------------------------------------------------------------------------------------------

class ModelHelper:
    """
    Collection of static utility methods for PyTorch model operations.
    
    This class provides a suite of utility functions for common model operations
    including parameter counting, serialization, visualization, kernel generation,
    and model export. All methods are static and can be called without instantiation.
    
    Methods:
        GetNumParamters: Count trainable parameters in a model
        savePickle: Serialize data to pickle file
        loadPickle: Deserialize data from pickle file
        plotImage: Visualize tensor as image
        GaborKernels: Generate Gabor filter kernels
        saveModel: Save model weights to checkpoint
        loadModel: Load model weights from checkpoint
        exportOnnx: Export model to ONNX format
    """
    
    # -------------------------------------------------------------------------------------------------------------
    def GetNumParamters(self):
        """
        Count the number of trainable parameters in the model.
        
        This method computes the total number of trainable parameters by summing
        the number of elements in each parameter tensor that has requires_grad=True.
        Useful for model complexity analysis and debugging.
        
        Returns:
            tuple: (total_params, param_list) where:
                - total_params (int): Total number of trainable parameters
                - param_list (list): List of parameter counts per tensor
                
        Example:
            >>> model = ResNet(num_outputs=2)
            >>> helper = ModelHelper()
            >>> total, per_layer = helper.GetNumParamters()
            >>> print(f"Total trainable parameters: {total:,}")
            Total trainable parameters: 23,512,130
            
        Notes:
            - Only counts parameters with requires_grad=True
            - Frozen layers are not counted
            - Includes biases if present
        """
        numel_list = [
            p.numel() for p in self.model.parameters() if p.requires_grad == True
        ]
        return sum(numel_list), numel_list
    
    # -------------------------------------------------------------------------------------------------------------
    def savePickle(path, fileName, data):
        """
        Save data to a pickle file.
        
        Serializes Python objects to a binary pickle file for later retrieval.
        Useful for saving training metrics, configurations, or intermediate results.
        
        Args:
            path (str): Directory path where the file will be saved
            fileName (str): Name of the pickle file (should include .pkl extension)
            data: Python object to serialize (can be dict, list, array, etc.)
            
        Example:
            >>> metrics = {'loss': [0.5, 0.3, 0.2], 'accuracy': [0.7, 0.8, 0.85]}
            >>> ModelHelper.savePickle('/path/to/results', 'metrics.pkl', metrics)
            
        Notes:
            - File is opened in append binary mode ('ab')
            - Multiple calls will append to the same file
            - Use loadPickle to retrieve the data
        """
        saveFile = os.path.join(path, fileName)
        File = open(saveFile, "ab")
        pickle.dump(data, File)
        
    # -------------------------------------------------------------------------------------------------------------
    def loadPickle(path, fileName):
        """
        Load data from a pickle file.
        
        Deserializes a Python object that was previously saved with savePickle.
        
        Args:
            path (str): Directory path where the file is located
            fileName (str): Name of the pickle file to load
            
        Returns:
            object: The deserialized Python object
            
        Raises:
            FileNotFoundError: If the pickle file doesn't exist
            pickle.UnpicklingError: If the file is corrupted or invalid
            
        Example:
            >>> metrics = ModelHelper.loadPickle('/path/to/results', 'metrics.pkl')
            >>> print(metrics['accuracy'])
            [0.7, 0.8, 0.85]
        """
        loadFile = os.path.join(path, fileName)
        File = open(loadFile, "rb")
        data = pickle.load(File)
        return data
    
    # -------------------------------------------------------------------------------------------------------------
    def plotImage(img, permute=True):
        """
        Display a tensor as an image using matplotlib.
        
        This utility function visualizes PyTorch tensors as grayscale images,
        automatically handling gradient tracking and dimension permutation.
        
        Args:
            img (torch.Tensor): Image tensor to display
                Expected shapes:
                - (H, W): Grayscale image
                - (C, H, W): Multi-channel image (will be permuted to H, W, C)
            permute (bool, optional): Whether to permute dimensions from (C, H, W)
                to (H, W, C) for display. Defaults to True
                
        Side Effects:
            - Displays image in matplotlib window
            - Detaches tensor from computation graph if needed
            
        Example:
            >>> # Display a model's first layer weights
            >>> weights = model.conv1.weight[0, 0]  # Single filter
            >>> ModelHelper.plotImage(weights, permute=False)
            
            >>> # Display a preprocessed image
            >>> img_tensor = batch['image'][0]  # Shape: (1, 120, 120)
            >>> ModelHelper.plotImage(img_tensor)
            
        Notes:
            - Uses 'gray' colormap for visualization
            - Automatically calls detach() for tensors in computation graph
            - Blocking call - window must be closed to continue execution
        """
        # Detach from computation graph if tensor is not a leaf
        if img.is_leaf == False:
            img = img.detach()

        # Permute from (C, H, W) to (H, W, C) for matplotlib
        if permute and len(img.shape) == 3:
            img = img.permute(1, 2, 0)

        plt.imshow(img, cmap="gray")
        plt.show()
        
    # -------------------------------------------------------------------------------------------------------------
    def GaborKernels(size=7, showPlots=False):
        """
        Generate a bank of Gabor filter kernels.
        
        Creates a set of Gabor filters with varying orientations and frequencies.
        Gabor filters are useful for edge detection and texture analysis in images,
        particularly for Cherenkov telescope images which have oriented features.
        
        Args:
            size (int, optional): Size of the kernel (size x size). Defaults to 7
            showPlots (bool, optional): Whether to display each kernel. Defaults to False
            
        Returns:
            list: List of numpy arrays, each representing a Gabor kernel
                Length: 4 (orientations) × 5 (frequencies) = 20 kernels
                Each kernel shape: (size, size)
                
        Kernel Parameters:
            - Orientations (theta): 0°, 45°, 90°, 135° (4 angles)
            - Frequencies: 0.15, 0.25, 0.35, 0.45, 0.55 (5 frequencies)
            - Sigma: Fixed at 3 (spatial extent of the kernel)
            
        Example:
            >>> # Generate 20 Gabor kernels of size 7x7
            >>> kernels = ModelHelper.GaborKernels(size=7, showPlots=False)
            >>> print(f"Generated {len(kernels)} kernels")
            Generated 20 kernels
            >>> print(kernels[0].shape)
            (7, 7)
            
            >>> # Visualize kernels during generation
            >>> kernels = ModelHelper.GaborKernels(size=11, showPlots=True)
            
        Notes:
            - Uses skimage.filters.gabor_kernel for generation
            - Only real part of Gabor kernel is used
            - Kernels are resized to specified size using bilinear interpolation
            - Useful for initializing convolutional layers with oriented filters
            
        Applications:
            - Initializing first convolutional layer weights
            - Feature extraction for shower image analysis
            - Edge and orientation detection in Cherenkov images
        """
        # prepare filter bank kernels
        kernels = []
        # Iterate over 4 orientations
        for theta in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
            sigma = 3  # Fixed spatial extent
            # Iterate over 5 frequencies
            for frequency in (0.15, 0.25, 0.35, 0.45, 0.55):
                # Generate Gabor kernel (complex-valued)
                kernel = np.real(
                    gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)
                )

                # Resize to specified size
                kernel = resize(kernel, [size, size])
                kernels.append(kernel)

                # Optionally display each kernel
                if showPlots:
                    print(
                        "Theta: ",
                        theta,
                        " Sigma: ",
                        sigma,
                        " Frequency: ",
                        frequency,
                        " Kernel size:",
                        kernel.shape,
                    )
                    plt.imshow(kernel)
                    plt.show()

        return kernels
    
    # -------------------------------------------------------------------------------------------------------------
    def saveModel(model, data_path, filename):
        """
        Save model weights to a checkpoint file.
        
        Saves the model's state dictionary (weights and biases) to a file
        for later loading and inference or continued training.
        
        Args:
            model (torch.nn.Module): The model to save
            data_path (str): Directory path where the checkpoint will be saved
            filename (str): Name of the checkpoint file (typically .pth extension)
            
        Side Effects:
            - Creates a .pth file in the specified directory
            - Prints confirmation message
            
        Example:
            >>> model = ResNet(num_outputs=2)
            >>> ModelHelper.saveModel(model, './checkpoints', 'best_model.pth')
            Saving model: best_model.pth
            
        Notes:
            - Only saves state_dict (weights), not the full model
            - Does not save optimizer state or training history
            - Use torch.save with full checkpoint dict for complete saving
        """
        print("Saving model: ", filename)
        torch.save(model.state_dict(), os.path.join(data_path, filename))
        
    # -------------------------------------------------------------------------------------------------------------
    def loadModel(model, data_path, filename, mode, device_str='cpu'):
        """
        Load model weights from a checkpoint file with robust error handling.
        
        This method loads pre-trained weights into a model, handling various
        checkpoint formats and partial weight loading. It includes automatic
        key name adjustment for different checkpoint structures and validates
        weight dimensions.
        
        Args:
            model (torch.nn.Module): The model to load weights into
            data_path (str): Directory path where the checkpoint is located
            filename (str): Name of the checkpoint file
            mode (Mode): Operation mode (train, results, validate, observation, tunning)
                Determines error handling strictness
            device_str (str, optional): Device to load model onto ('cpu', 'cuda', 'cuda:0').
                Defaults to 'cpu'
                
        Returns:
            torch.nn.Module: The model with loaded weights
            
        Checkpoint Format Compatibility:
            Handles multiple checkpoint formats:
            - Direct state_dict: {'layer.weight': tensor, ...}
            - Wrapped state_dict: {'state_dict': {...}}
            - Model state_dict: {'model_state_dict': {...}}
            - Prefixed keys: {'model.0.layer.weight': tensor, ...}
            
        Key Matching Strategy:
            1. Remove 'model.0.' prefix from checkpoint keys if present
            2. Filter out keys not in model's state_dict
            3. Filter out keys with dimension mismatches
            4. Load only matching keys (strict=False)
            
        Error Handling:
            - Training/Tunning mode: Issues warning for mismatches, continues
            - Other modes: Raises ValueError for mismatches
            - Missing checkpoint: Exits in non-training modes
            
        Example:
            >>> model = ResNet(num_outputs=2)
            >>> model = ModelHelper.loadModel(
            ...     model,
            ...     './checkpoints',
            ...     'best_model.pth',
            ...     Mode.results,
            ...     device_str='cuda:0'
            ... )
            Loading model: best_model.pth
            Model Loaded.
            
        Notes:
            - Uses weights_only=False for pickle compatibility (security warning)
            - Automatically moves model to specified device
            - Supports partial weight loading for transfer learning
            - Strict=False allows loading subset of weights
            
        Security Warning:
            Currently uses weights_only=False which can execute arbitrary code
            during unpickling. Future versions should use weights_only=True.
        """
        # Check if checkpoint file exists
        if os.path.isfile(os.path.join(data_path, filename)):
            print("Loading model: ", filename)
            
            # TODO: Test weights_only=True. Currently getting FutureWarning
            # about security implications of weights_only=False
            pretrained_dict = torch.load(
                os.path.join(data_path, filename),
                map_location=torch.device(device_str),
                weights_only=False
            )

            # Handle different checkpoint formats
            # Format 1: {'state_dict': {...}}
            if type(pretrained_dict) == dict and "state_dict" in pretrained_dict:
                pretrained_dict = pretrained_dict["state_dict"]

            # Format 2: {'model_state_dict': {...}}
            if type(pretrained_dict) == dict and "model_state_dict" in pretrained_dict:
                pretrained_dict = pretrained_dict["model_state_dict"]

            # Get current model's state dict
            model_dict = model.state_dict()

            # Remove 'model.0.' prefix if present in checkpoint keys
            modified_dict = {}
            prefix = "model.0."
            for key in pretrained_dict:
                # Check if the key starts with 'model.0.'
                if key.startswith(prefix):
                    # Remove the prefix and save with new key
                    new_key = key.replace(prefix, "")
                    modified_dict[new_key] = pretrained_dict[key]

            # Use modified dict if any keys were modified
            if len(modified_dict) > 0:
                pretrained_dict = modified_dict

            # Filter checkpoint to only include matching keys with same dimensions
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() 
                if k in model_dict and model_dict[k].size() == v.size()
            }            

            # Check for mismatches between checkpoint and model
            if (len(model_dict) != len(pretrained_dict) or 
                set(model_dict.keys()) != set(pretrained_dict.keys())):

                pretrain_len = len(pretrained_dict)
                model_len = len(model_dict)
                # Keys only in checkpoint
                unique_pretrained = set(pretrained_dict.keys()) - set(model_dict.keys())
                # Keys only in model
                unique_model = set(model_dict.keys()) - set(pretrained_dict.keys())

                # Strict error checking for non-training modes
                if (mode != Mode.train and mode != Mode.tunning):
                    raise ValueError(
                        f"Error Loading the model. Pretrained Dict length: {pretrain_len} "
                        f"Model Dict length: {model_len}. Differences -> "
                        f"Pretrained keys: {unique_pretrained}, Model keys: {unique_model}"
                    )
                else:
                    # Warning for training/tuning modes (allow partial loading)
                    warnings.warn(
                        f"Warning Loading the model. Pretrained Dict length: {pretrain_len} "
                        f"Model Dict length: {model_len}. "
                        f"Differences -> Pretrained keys: {unique_pretrained}, "
                        f"Model keys: {unique_model}",
                        UserWarning
                    )
                    
            # Update model dict with pretrained weights
            model_dict.update(pretrained_dict)
            # Load the new state dict (strict=False allows partial loading)
            model.load_state_dict(model_dict, strict=False)

            # Move model to specified device
            device = torch.device(device_str)
            model.to(device)

            print("Model Loaded.")
        else:
            # Checkpoint doesn't exist
            model.to(torch.device(device_str))
            
            print(f"CheckPoint file does not exist: {filename}")
            # Exit if not in training mode (checkpoint required)
            if mode != Mode.train:
                exit()

        return model
    
    # -------------------------------------------------------------------------------------------------------------
    def exportOnnx(model, dummy_input, onnx_name, input_names, output_names):
        """
        Export PyTorch model to ONNX format with simplification.
        
        Converts a PyTorch model to ONNX (Open Neural Network Exchange) format
        for deployment and interoperability with other frameworks. Also applies
        optimization and simplification to the exported model.
        
        Args:
            model (torch.nn.Module): The PyTorch model to export
            dummy_input (torch.Tensor or tuple): Example input for tracing
                Must have same shape and type as model's expected input
            onnx_name (str): Base name for output files (without extension)
            input_names (list): Names for model inputs
                Example: ['image', 'peak_time']
            output_names (list): Names for model outputs
                Example: ['classification', 'energy', 'direction']
                
        Side Effects:
            - Creates two ONNX files:
                1. {onnx_name}.onnx - Original exported model
                2. {onnx_name}_simp.onnx - Simplified and optimized model
            - Prints verbose export information
            
        Example:
            >>> model = ResNet(num_outputs=2)
            >>> model.eval()
            >>> dummy_input = torch.randn(1, 1, 120, 120)
            >>> ModelHelper.exportOnnx(
            ...     model,
            ...     dummy_input,
            ...     'resnet_model',
            ...     input_names=['image'],
            ...     output_names=['classification']
            ... )
            
        Requirements:
            - pip install onnx
            - pip install onnxsim
            
        Notes:
            - Model must be in eval() mode before export
            - Dummy input shape must match model's expected input
            - Simplified model is validated before saving
            - ONNX format allows deployment to:
                * TensorRT (NVIDIA)
                * OpenVINO (Intel)
                * CoreML (Apple)
                * ONNX Runtime (cross-platform)
                
        Simplification Benefits:
            - Removes redundant operations
            - Folds constant computations
            - Optimizes graph structure
            - Reduces model size
            - Improves inference speed
        """
        # Export model to ONNX format
        torch.onnx.export(
            model,
            dummy_input,
            onnx_name + ".onnx",
            verbose=True,
            input_names=input_names,
            output_names=output_names,
        )

        # Load the exported ONNX model
        model = onnx.load(onnx_name + ".onnx")

        # Apply simplification and optimization
        model_simp, check = simplify(model)

        # Validate simplified model
        assert check, "Simplified ONNX model could not be validated"
        
        # Save simplified model
        onnx.save(model_simp, onnx_name + "_simp.onnx")
    # -------------------------------------------------------------------------------------------------------------
