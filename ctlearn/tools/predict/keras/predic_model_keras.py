"""
Keras model prediction module for CTLearn.
This module provides functionality to load trained Keras models and perform batch predictions
on DL1 data, with optional feature vector extraction from backbone models.
"""

from ctlearn.core.data_loader.loader import DLDataLoader
import keras 
from astropy.table import Table, vstack
import numpy as np


def predict_with_model(self, model_path, task):
    """
    Load and predict with a CTLearn Keras model.
    
    This function loads a trained model from the specified path and performs predictions
    on the provided data. It handles both complete and incomplete batches, and optionally
    extracts feature vectors from the backbone model for downstream analysis.
    
    Parameters
    ----------
    model_path : str
        Path to a Keras model file (Keras 3) or directory (Keras 2).
        The model should be a complete trained CTLearn model.
    task : str
        The task for which prediction is being made (e.g., 'type', 'energy', 'cameradirection').
    
    Returns
    -------
    predict_data : astropy.table.Table
        Table containing the prediction results with columns corresponding to
        the model's output.
    feature_vectors : np.ndarray or None
        Feature vectors extracted from the backbone model if dl1_features is enabled.
        Returns None if feature extraction is not requested.
    """
    # Create data loader for the main batch processing
    # The DLDataLoader is initialized separately for each prediction task
    # Batch size is multiplied by number of replicas for distributed inference
    data_loader = DLDataLoader.create(
        framework="keras",
        DLDataReader=self.dl1dh_reader,
        indices=self.indices,
        tasks=[],
        batch_size=self.batch_size * self.strategy.num_replicas_in_sync,
        sort_by_intensity=self.sort_by_intensity,
        stack_telescope_images=self.stack_telescope_images,
    )
    
    # Handle incomplete last batch
    # Keras only processes complete batches during prediction, so we need
    # a separate data loader for remaining events that don't fill a complete batch
    data_loader_last_batch = None
    if self.last_batch_size > 0:
        # Extract indices for the last incomplete batch
        last_batch_indices = self.indices[-self.last_batch_size:]
        data_loader_last_batch = DLDataLoader.create(
            framework="keras",
            DLDataReader=self.dl1dh_reader,
            indices=last_batch_indices,
            tasks=[],
            batch_size=self.last_batch_size,
            sort_by_intensity=self.sort_by_intensity,
            stack_telescope_images=self.stack_telescope_images,
        )

    # Load the trained model from the specified path
    model = keras.saving.load_model(model_path)
    
    # Initialize variables for optional feature extraction
    backbone_model, feature_vectors = None, None
    
    if self.dl1_features:
        # Feature extraction mode: split model into backbone and head
        # This allows us to extract intermediate representations (feature vectors)
        
        # Extract the backbone model (second layer of the complete model)
        # Layer 0: Input, Layer 1: Backbone (feature extractor), Layers 2+: Head
        backbone_model = model.get_layer(index=1)
        
        # Reconstruct the head model from layers after the backbone
        # Define input with the same shape as backbone output
        backbone_output_shape = keras.Input(model.layers[2].input_shape[1:])
        x = backbone_output_shape
        
        # Connect all layers after backbone to create head model
        for layer in model.layers[2:]:
            x = layer(x)
        head = keras.Model(inputs=backbone_output_shape, outputs=x)
        
        # Extract feature vectors from backbone
        feature_vectors = backbone_model.predict(
            data_loader, verbose=self.keras_verbose
        )
        
        # Generate predictions from head using extracted features
        predict_data = Table(
            {
                task: head.predict(
                    feature_vectors, verbose=self.keras_verbose
                )
            }
        )
        
        # Process last incomplete batch if it exists
        if data_loader_last_batch is not None:
            # Extract features from last batch
            feature_vectors_last_batch = backbone_model.predict(
                data_loader_last_batch, verbose=self.keras_verbose
            )
            # Concatenate feature vectors from all batches
            feature_vectors = np.concatenate(
                (feature_vectors, feature_vectors_last_batch)
            )
            # Generate predictions for last batch and stack with main predictions
            predict_data = vstack(
                [
                    predict_data,
                    Table(
                        {
                            task: head.predict(
                                feature_vectors_last_batch,
                                verbose=self.keras_verbose,
                            )
                        }
                    ),
                ]
            )
    else:
        # Standard prediction mode without feature extraction
        # Use the complete model for end-to-end prediction
        predict_data = model.predict(data_loader, verbose=self.keras_verbose)
        
        # Convert predictions to Astropy Table
        if isinstance(predict_data, dict):
            predict_data = Table(predict_data)
        else:
            predict_data = Table({task: predict_data})
        
        # Process last incomplete batch if it exists
        if data_loader_last_batch is not None:
            # Generate predictions for last batch
            predict_data_last_batch = model.predict(
                data_loader_last_batch, verbose=self.keras_verbose
            )
            
            if isinstance(predict_data_last_batch, dict):
                predict_data_last_batch = Table(predict_data_last_batch)
            else:
                predict_data_last_batch = Table({task: predict_data_last_batch})
            
            # Stack predictions from main batches and last batch
            predict_data = vstack([predict_data, predict_data_last_batch])
    
    return predict_data, feature_vectors