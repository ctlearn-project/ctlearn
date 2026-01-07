"""
Keras prediction module for LST1 telescope data.
This module provides functionality to load trained Keras models and perform predictions
on DL1 level data for particle type classification, energy estimation, and direction reconstruction.
"""

from ctapipe.io import read_table
from astropy.table import join
import keras
from dl1_data_handler.reader import get_unmapped_image
import numpy as np


def predictions(self):
    """
    Perform predictions on input DL1 data using trained Keras models.
    
    This function processes the input file in batches, applies quality cuts,
    and generates predictions for particle type, energy, and/or direction
    depending on the configured models. The models are split into backbone
    and head components to extract feature vectors.
    
    Returns
    -------
    tuple
        Contains the following arrays:
        - event_id: Event identifiers
        - tel_azimuth: Telescope azimuth angles
        - tel_altitude: Telescope altitude angles
        - trigger_time: Event trigger times in MJD
        - prediction: Particle type classification scores (gammaness)
        - energy: Reconstructed energy values
        - cam_coord_offset_x: Camera coordinate offset in x direction
        - cam_coord_offset_y: Camera coordinate offset in y direction
        - classification_fvs: Classification feature vectors from backbone
        - energy_fvs: Energy estimation feature vectors from backbone
        - direction_fvs: Direction reconstruction feature vectors from backbone
    
    Notes
    -----
    The function processes data in batches to manage memory efficiently and
    applies quality selection criteria before making predictions.
    """
    # Initialize output arrays for storing results
    event_id, tel_azimuth, tel_altitude, trigger_time = [], [], [], []
    prediction, energy, cam_coord_offset_x, cam_coord_offset_y = [], [], [], []
    classification_fvs, energy_fvs, direction_fvs = [], [], []
    
    # Process input file in batches
    for start in range(0, self.table_length, self.batch_size):
        stop = min(start + self.batch_size, self.table_length)
        self.log.debug("Processing chunk from '%d' to '%d'.", start, stop - 1)
        
        # Read the DL1 data table for current batch
        dl1_table = read_table(
            self.input_url, self.image_table_path, start=start, stop=stop
        )
        
        # Join tables to enable quality selection
        # Join with parameter table for event parameters
        dl1_table = join(
            left=dl1_table,
            right=self.parameter_table,
            keys=["event_id"],
        )
        # Join with trigger table for timing information
        dl1_table = join(
            left=dl1_table,
            right=self.trigger_table,
            keys=["event_id"],
        )
        
        # Apply quality selection criteria
        # Initialize mask to accept all events initially
        passes_quality_checks = np.ones(len(dl1_table), dtype=bool)
        
        # Apply quality query if configured
        if self.quality_query:
            passes_quality_checks = self.quality_query.get_table_mask(dl1_table)
        
        # Filter events based on quality criteria
        dl1_table = dl1_table[passes_quality_checks]
        
        # Skip batch if no events passed quality selection
        if len(dl1_table) == 0:
            self.log.debug("No events passed the quality selection.")
            continue
        
        # Prepare input data by mapping images to model input format
        data = []
        for event in dl1_table:
            # Get the unmapped image with specified channels and transforms
            image = get_unmapped_image(event, self.channels, self.transforms)
            # Map image to model's expected input format
            data.append(self.image_mapper.map_image(image))
        input_data = {"input": np.array(data)}
        
        # Handle compatibility between Keras 2 and Keras 3
        # Keras 3 expects direct array input, not dictionary
        if int(keras.__version__.split(".")[0]) >= 3:
            input_data = input_data["input"]

        # Store event metadata
        event_id.extend(dl1_table["event_id"].data)
        tel_azimuth.extend(dl1_table["tel_az"].data)
        tel_altitude.extend(dl1_table["tel_alt"].data)
        trigger_time.extend(dl1_table["time"].mjd)
        
        # Perform particle type classification if model is loaded
        if self.load_type_model_from is not None:
            # Extract feature vectors from backbone
            classification_feature_vectors = self.backbone_type.predict_on_batch(input_data)
            classification_fvs.extend(classification_feature_vectors)
            # Generate predictions from head model
            predict_data = self.head_type.predict_on_batch(classification_feature_vectors)
            # Extract gammaness score (probability of being gamma)
            prediction.extend(predict_data[:, 1])
        
        # Perform energy estimation if model is loaded
        if self.load_energy_model_from is not None:
            # Extract feature vectors from backbone
            energy_feature_vectors = self.backbone_energy.predict_on_batch(input_data)
            energy_fvs.extend(energy_feature_vectors)
            # Generate energy predictions from head model
            predict_data = self.head_energy.predict_on_batch(energy_feature_vectors)
            energy.extend(predict_data.T[0])
        
        # Perform direction reconstruction if model is loaded
        if self.load_cameradirection_model_from is not None:
            # Extract feature vectors from backbone
            direction_feature_vectors = self.backbone_direction.predict_on_batch(input_data)
            direction_fvs.extend(direction_feature_vectors)
            # Generate direction predictions from head model
            predict_data = self.head_direction.predict_on_batch(direction_feature_vectors)
            # Extract x and y components of camera coordinate offset
            cam_coord_offset_x.extend(predict_data.T[0])
            cam_coord_offset_y.extend(predict_data.T[1])
    
    return (event_id, tel_azimuth, tel_altitude, trigger_time, prediction, energy,
            cam_coord_offset_x, cam_coord_offset_y, classification_fvs, energy_fvs, direction_fvs)


def _split_model(model):
    """
    Split a Keras model into backbone and head components.
    
    This function separates a trained model into two parts:
    - Backbone: Feature extraction layers (typically convolutional layers)
    - Head: Task-specific prediction layers (typically dense layers)
    
    This separation allows extraction of intermediate feature representations
    which can be useful for analysis or transfer learning.
    
    Parameters
    ----------
    model : keras.Model
        Complete trained Keras model to be split. The model should have:
        - Layer 0: Input layer
        - Layer 1: Backbone (feature extractor)
        - Layers 2+: Head (prediction layers)
    
    Returns
    -------
    backbone : keras.Model
        Feature extraction model that outputs intermediate representations.
    head : keras.Model
        Prediction model that takes backbone outputs and produces final predictions.
    
    Notes
    -----
    The function assumes a specific model architecture where the backbone
    is the second layer (index 1) of the complete model. This is a common
    pattern in CTLearn models where the backbone is wrapped as a single layer.
    """
    # Extract the backbone model (second layer of the complete model)
    # Layer 0 is the input, layer 1 is the backbone feature extractor
    backbone = model.get_layer(index=1)
    
    # Create a new head model using layers after the backbone
    # Define input with the same shape as backbone output
    backbone_output_shape = keras.Input(model.layers[2].input_shape[1:])
    x = backbone_output_shape
    
    # Reconstruct head by connecting all layers after backbone
    for layer in model.layers[2:]:
        x = layer(x)
    
    # Create the head model
    head = keras.Model(inputs=backbone_output_shape, outputs=x)
    
    return backbone, head


def load_keras_model(self):
    """
    Load Keras models from saved files and split them into backbone and head.
    
    This function loads trained Keras models for different tasks (particle type
    classification, energy estimation, direction reconstruction) and splits each
    into backbone and head components for efficient prediction and feature extraction.
    
    Parameters
    ----------
    self : object
        Prediction handler instance containing model paths:
        - load_type_model_from: Path to particle classification model
        - load_energy_model_from: Path to energy estimation model
        - load_cameradirection_model_from: Path to direction reconstruction model
    
    Returns
    -------
    input_shape : tuple
        Shape of the model input (height, width, channels).
        Returns the shape from the last loaded model.
    
    Notes
    -----
    The function sets the following attributes on self:
    - backbone_type, head_type: Split models for particle classification
    - backbone_energy, head_energy: Split models for energy estimation
    - backbone_direction, head_direction: Split models for direction reconstruction
    """
    input_shape = None
    
    # Load particle type classification model if configured
    if self.load_type_model_from is not None:
        self.log.info("Loading the type model from %s.", self.load_type_model_from)
        model_type = keras.saving.load_model(self.load_type_model_from)
        input_shape = model_type.input_shape[1:]
        # Split model into backbone and head
        self.backbone_type, self.head_type = _split_model(model_type)
    
    # Load energy estimation model if configured
    if self.load_energy_model_from is not None:
        self.log.info(
            "Loading the energy model from %s.", self.load_energy_model_from
        )
        model_energy = keras.saving.load_model(self.load_energy_model_from)
        input_shape = model_energy.input_shape[1:]
        # Split model into backbone and head
        self.backbone_energy, self.head_energy = _split_model(model_energy)
    
    # Load direction reconstruction model if configured
    if self.load_cameradirection_model_from is not None:
        self.log.info(
            "Loading the cameradirection model from %s.", self.load_cameradirection_model_from
        )
        model_direction = keras.saving.load_model(
            self.load_cameradirection_model_from
        )
        input_shape = model_direction.input_shape[1:]
        # Split model into backbone and head
        self.backbone_direction, self.head_direction = _split_model(model_direction)
    
    return input_shape
