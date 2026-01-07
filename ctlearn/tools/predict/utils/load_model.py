"""
Model loading utility module for CTLearn predictions.
This module provides a framework-agnostic interface for loading trained models,
supporting both Keras and PyTorch frameworks.
"""


def load_model(self):
    """
    Load a trained model based on the configured framework type.
    
    This function acts as a dispatcher that delegates model loading to the appropriate
    framework-specific implementation. It supports both Keras and PyTorch frameworks
    and loads the model from the checkpoint path specified in the configuration.
    
    Parameters
    ----------
    self : PredictionHandler
        The prediction handler instance containing configuration parameters including:
        - framework_type: str, either "keras" or "pytorch"
        - Model checkpoint paths and other configuration parameters
    
    Returns
    -------
    model : object
        The loaded model ready for inference. Type depends on the framework:
        - For Keras: keras.Model
        - For PyTorch: torch.nn.Module
        Returns None if the framework is not recognized.
    
    Raises
    ------
    ImportError
        If the specified framework's prediction module cannot be imported.
    
    Notes
    -----
    The function automatically detects the framework type from the configuration
    and imports the appropriate loading function dynamically to avoid unnecessary
    dependencies when using only one framework.
    """
    if self.framework_type == "keras":
        # Load Keras model using framework-specific loader
        from ctlearn.tools.predict.keras.predic_LST1_keras import load_keras_model
        return load_keras_model(self)
    
    elif self.framework_type == "pytorch":
        # Load PyTorch model using framework-specific loader
        from ctlearn.tools.predict.pytorch.predic_LST1_pytorch import load_pytorch_model
        return load_pytorch_model(self)
    
    else:
        # Log error if framework is not recognized
        self.log.error(f"Framework '{self.framework_type}' not found! Supported frameworks: 'keras', 'pytorch'")
        return None