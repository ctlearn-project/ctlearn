"""
PyTorch model prediction module for CTLearn.
This module provides functionality to load trained models and perform batch predictions
on DL1 data for multiple tasks including particle classification, energy estimation,
and direction reconstruction.
"""

from ctlearn.core.data_loader.loader import DLDataLoader
import torch
from tqdm import tqdm
import numpy as np
import inspect
from ctlearn.tools.predict.utils.load_model import load_model


def predict_with_model_pytorch(self, task):
    """
    Load and predict with a CTLearn PyTorch model.
    
    This function loads a trained model from the specified path and performs predictions
    on the provided data. It processes the data in batches and returns predictions for
    particle type classification, energy estimation, and/or direction reconstruction
    based on the configured task.
    
    Parameters
    ----------
    task : Task
        The task(s) to perform predictions for (type, energy, or direction).
    
    Returns
    -------
    predict_data : dict
        Dictionary containing prediction results with keys:
        - 'type': Particle type classification probabilities (gammaness scores)
        - 'energy': Reconstructed energy values
        - 'cameradirection': Camera coordinate offsets for direction reconstruction
    feature_vectors : None
        Feature vectors (currently not extracted, placeholder for future implementation).
    
    Notes
    -----
    The function automatically detects whether the model requires peak time information
    by inspecting the model's forward method signature. Models can accept either one
    input (image only) or two inputs (image and peak time).
    """
    # Initialize batch size from configuration parameters
    self.batch_size = self.parameters["hyp"]["batches"]

    # Create data loader for the specified task
    # The DLDataLoader is initialized separately for each task to ensure robustness
    channels = ["cleaned_image", "cleaned_peak_time"]
    if self.parameters["normalization"]["apply_log_scaling"][0]:
        channels[0] = "log_" + channels[0]
    self.dl1dh_reader.channels = channels    
    data_loader = DLDataLoader.create(
        framework="pytorch",
        DLDataReader=self.dl1dh_reader,
        indices=self.indices,
        tasks=[task],
        parameters=self.parameters,
        use_augmentation=False,
        batch_size=self.batch_size,
        sort_by_intensity=self.sort_by_intensity,
        stack_telescope_images=self.stack_telescope_images,
    )
    
    # Note: Handling of incomplete last batch
    # In PyTorch, unlike Keras, we can process incomplete batches directly
    # without needing a separate data loader. The code below is kept as reference
    # for potential future use or compatibility with other frameworks.
    
    # data_loader_last_batch = None
    # if self.last_batch_size > 0:
    #     last_batch_indices = self.indices[-self.last_batch_size:]
    #     data_loader_last_batch = DLDataLoader.create(
    #         framework="pytorch",
    #         DLDataReader=self.dl1dh_reader,
    #         indices=last_batch_indices,
    #         tasks=task,
    #         parameters=self.parameters,
    #         use_augmentation=False,
    #         batch_size=self.last_batch_size,
    #         sort_by_intensity=self.sort_by_intensity,
    #         stack_telescope_images=self.stack_telescope_images,
    #     )

    # Load the trained model from checkpoint
    model = load_model(self)
    
    # Inspect model signature to determine number of inputs
    # This allows the code to work with models that take either:
    # - Single input: image only
    # - Dual input: image and peak time
    sig = inspect.signature(model.forward)
    num_inputs = len(sig.parameters)
    
    # Initialize prediction data dictionary with empty lists
    predict_data = {}
    predict_data['type'] = []
    predict_data['energy'] = []
    predict_data["cameradirection"] = []
    
    # Set model to evaluation mode (disables dropout, batch normalization, etc.)
    model.eval()
    
    # Perform predictions without gradient computation (faster inference)
    with torch.no_grad():
        for i, x in enumerate(tqdm(data_loader, desc="Processing", total=len(data_loader))):
            # Skip empty batches
            if len(x[0]['image']) == 0:
                continue
            
            # Forward pass through the model
            classification_pred, energy_pred, direction_pred = model(
                x[0]['image'].to(self.device)
            )
            
            # Collect particle type classification predictions
            if classification_pred[0] is not None:
                # Apply softmax to get probability distribution and extract gammaness score
                gammaness = torch.softmax(classification_pred[0], dim=1).cpu().detach().numpy()
                predict_data['type'].extend(gammaness)
            
            # Collect energy estimation predictions
            if energy_pred[0] is not None:
                predict_data['energy'].extend(energy_pred[0].cpu().detach().numpy())
            
            # Collect direction reconstruction predictions
            if direction_pred[0] is not None:
                predict_data["cameradirection"].extend(direction_pred[0].cpu().detach().numpy())
            
            # Log progress every 100 batches
            if i % 100 == 0:
                self.log.info(f"Processed {i}/{len(data_loader)} events.")
    
    self.log.info("Processing completed.")
    
    # Convert lists to numpy arrays for efficient storage and further processing
    predict_data["cameradirection"] = np.array(predict_data["cameradirection"])
    predict_data["type"] = np.array(predict_data["type"])
    predict_data["energy"] = np.array(predict_data["energy"])
    
    # Return predictions and placeholder for feature vectors
    return predict_data, None