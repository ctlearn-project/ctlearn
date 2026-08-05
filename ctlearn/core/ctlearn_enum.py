"""
CTLearn Enumeration Types Module

This module defines enumeration types used throughout the CTLearn framework
to ensure type safety and consistency when specifying framework types, tasks,
event types, and operation modes.

Enumerations:
    FrameworkType: Deep learning framework selection (Keras or PyTorch)
    Task: Machine learning task type (classification, regression, etc.)
    EventType: Cosmic ray particle type classification
    Mode: Operation mode for the CTLearn pipeline
"""

from enum import Enum

class FrameworkType(Enum):
    """
    Deep learning framework type enumeration.
    
    This enumeration specifies which deep learning framework to use for
    model training and inference. CTLearn supports both Keras (TensorFlow backend)
    and PyTorch frameworks.
    
    Attributes:
        KERAS (int): Use Keras/TensorFlow framework (value: 1)
            - Advantages: High-level API, easy to use, good for prototyping
            - TensorFlow 2.x with Keras API
            - Suitable for production deployment
            
        PYTORCH (int): Use PyTorch framework (value: 2)
            - Advantages: Dynamic computation graphs, flexible, research-friendly
            - PyTorch 1.x or 2.x
            - Better for custom architectures and experimental models
    
    Example:
        >>> from ctlearn.core.ctlearn_enum import FrameworkType
        >>> framework = FrameworkType.PYTORCH
        >>> print(framework.name)  # 'PYTORCH'
        >>> print(framework.value)  # 'PyTorch'
    """
    KERAS = "Keras"
    PYTORCH = "PyTorch"


class Task(Enum):
    """
    Machine learning task type enumeration.
    
    This enumeration defines the different analysis tasks that CTLearn can perform
    on Cherenkov telescope data. Each task corresponds to a specific scientific
    goal in gamma-ray astronomy.
    
    Attributes:
        type (int): Particle type classification task (value: 0)
            - Classify events as gamma-ray or background (proton/electron)
            - Binary classification problem
            - Output: Class probabilities (gamma vs hadron)
            - Critical for gamma-ray source detection
            
        energy (int): Energy regression task (value: 1)
            - Estimate the energy of the primary cosmic ray
            - Regression problem with log-scale energy
            - Output: Energy in TeV (typically log10(E/TeV))
            - Essential for measuring source spectra
            
        direction (int): Generic direction reconstruction task (value: 2)
            - General direction estimation (deprecated, use specific types)
            - Maintained for backward compatibility
            
        cameradirection (int): Direction reconstruction in camera coordinates (value: 3)
            - Predict shower direction in camera frame
            - Output: (dx, dy, distance) offsets in meters
            - Must be transformed to sky coordinates for analysis
            - Faster computation, no coordinate transformations needed
            
        skydirection (int): Direction reconstruction in sky coordinates (value: 4)
            - Predict shower direction in horizontal (Alt/Az) frame
            - Output: (altitude, azimuth, angular_separation) in degrees
            - Directly usable for source localization
            - Requires telescope pointing information
            
        all (int): Multi-task learning (all tasks simultaneously) (value: 5)
            - Train model to perform all tasks jointly
            - Shared feature extraction, task-specific heads
            - Can improve performance through transfer learning
            - More complex training but potentially better features
    
    Notes:
        - Tasks can be combined in multi-task learning setups
        - Each task requires specific loss functions and evaluation metrics
        - The choice of task affects data preprocessing and model architecture
    
    Example:
        >>> from ctlearn.core.ctlearn_enum import Task
        >>> task = Task.energy
        >>> if task == Task.energy:
        ...     print("Performing energy regression")
        >>> print(task.name)  # 'energy'
    """
    type = 0
    energy = 1
    direction = 2
    cameradirection = 3
    skydirection = 4
    all = 5

class EventType(Enum):
    """
    Cosmic ray event type enumeration.
    
    This enumeration classifies the primary particle that initiated the
    air shower detected by the Cherenkov telescopes. Distinguishing between
    gamma rays and background particles is crucial for gamma-ray astronomy.
    
    Attributes:
        gamma (int): Gamma-ray event (value: 0)
            - Primary particle: High-energy photon
            - Characteristics:
                * Electromagnetic shower
                * Narrow, elliptical image
                * Low muon content
                * Preferred class for gamma-ray astronomy
            - Used for source studies and spectral analysis
            
        proton (int): Proton-induced event (value: 1)
            - Primary particle: Proton (cosmic ray)
            - Characteristics:
                * Hadronic shower
                * Irregular, fragmented image
                * High muon content
                * Most common background (~90% of cosmic rays)
            - Main source of background contamination
            
        electron (int): Electron-induced event (value: 2)
            - Primary particle: Electron or positron
            - Characteristics:
                * Electromagnetic shower (similar to gamma)
                * Can be difficult to distinguish from gamma
                * Less common than protons
                * Secondary background source
            - Often grouped with gamma for some analyses
    
    Notes:
        - In binary classification, typically gamma vs (proton + electron)
        - Event type is known only for simulated data (Monte Carlo)
        - Real observations have unknown event types (classification goal)
        - Other particles (nuclei, muons) exist but are less common
    
    Physical Context:
        - Gamma rays: Signal we want to detect
        - Protons: Dominant background (~1000x more frequent)
        - Electrons: Minor background component
        - Background rejection crucial for sensitivity
    
    Example:
        >>> from ctlearn.core.ctlearn_enum import EventType
        >>> event = EventType.gamma
        >>> if event == EventType.gamma:
        ...     print("Signal event detected")
        >>> print(event.name)  # 'gamma'
        >>> print(event.value)  # 0
    """
    gamma = 0
    proton = 1
    electron = 2
    
class Mode(Enum):
    """
    Operation mode enumeration for the CTLearn pipeline.
    
    This enumeration defines the different operational modes that CTLearn
    can run in, determining what actions are performed on the data.
    
    Attributes:
        train (int): Training mode (value: 0)
            - Train model on training dataset
            - Update model weights through backpropagation
            - Validate on validation set each epoch
            - Save checkpoints and training curves
            - Enables data augmentation
            - Uses training-specific preprocessing
            
        results (int): Results generation mode (value: 1)
            - Generate predictions on test or validation data
            - No weight updates
            - Save predictions to HDF5 files (DL2 format)
            - Compute performance metrics
            - Create evaluation plots and tables
            - Used for final model evaluation
            
        validate (int): Validation mode (value: 2)
            - Evaluate model on validation dataset
            - No weight updates
            - Compute metrics only (no predictions saved)
            - Quick performance check
            - Can be run during or after training
            
        observation (int): Real observation mode (value: 3)
            - Process real telescope observations (not simulations)
            - No ground truth labels available
            - Generate DL2 data from DL1 real data
            - Apply trained model to unknown events
            - Output used for science analysis
            
        tunning (int): Hyperparameter tuning mode (value: 4)
            - Optimize model hyperparameters
            - Multiple training runs with different configurations
            - Uses validation set for hyperparameter selection
            - May use techniques like grid search, random search, or Bayesian optimization
            - Typically automated with tools like Optuna or Ray Tune
    
    Typical Workflow:
        1. train: Develop and train models on simulated data
        2. validate: Quick performance checks during development
        3. tunning: Optimize hyperparameters for best performance
        4. results: Final evaluation and analysis on test set
        5. observation: Apply to real telescope data
    
    Notes:
        - Each mode may have different data loading behavior
        - Some preprocessing steps (augmentation) only active in train mode
        - observation mode handles real data without labels
        - Mode affects logging, checkpointing, and output format
    
    Example:
        >>> from ctlearn.core.ctlearn_enum import Mode
        >>> mode = Mode.train
        >>> if mode == Mode.train:
        ...     print("Enabling data augmentation")
        >>> elif mode == Mode.observation:
        ...     print("Processing real data")
        >>> print(mode.name)  # 'train'
    """
    train = 0
    results = 1
    validate = 2
    observation = 3
    tunning = 4
