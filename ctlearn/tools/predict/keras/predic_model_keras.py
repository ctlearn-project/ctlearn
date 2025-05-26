from ctlearn.core.data_loader.loader import DLDataLoader
import keras 
from astropy.table import (
    Table,
    hstack,
    vstack,
    join,
    setdiff,
)
import numpy as np

def predict_with_model(self, model_path):
    """
    Load and predict with a CTLearn model.

    Load a model from the specified path and predict the data using the loaded model.
    If a last batch loader is provided, predict the last batch and stack the results.

    Parameters
    ----------
    model_path : str
        Path to a Keras model file (Keras3) or directory (Keras2).

    Returns
    -------
    predict_data : astropy.table.Table
        Table containing the prediction results.
    feature_vectors : np.ndarray
        Feature vectors extracted from the backbone model.
    """
    # Create a new DLDataLoader for each task
    # It turned out to be more robust to initialize the DLDataLoader separately.
    data_loader = DLDataLoader.create(
        framework="keras",
        DLDataReader=self.dl1dh_reader,
        indices=self.indices,
        tasks=[],
        batch_size=self.batch_size * self.strategy.num_replicas_in_sync,
        sort_by_intensity=self.sort_by_intensity,
        stack_telescope_images=self.stack_telescope_images,
    )
    
    # Keras is only considering the last complete batch.
    # In prediction mode we don't want to loose the last
    # uncomplete batch, so we are creating an additional
    # batch generator for the remaining events.
    data_loader_last_batch = None
    if self.last_batch_size > 0:
        last_batch_indices = self.indices[-self.last_batch_size :]
        data_loader_last_batch = DLDataLoader.create(
            framework="keras",
            DLDataReader=self.dl1dh_reader,
            indices=last_batch_indices,
            tasks=[],
            batch_size=self.last_batch_size,
            sort_by_intensity=self.sort_by_intensity,
            stack_telescope_images=self.stack_telescope_images,
        )

                
    # Load the model from the specified path
    model = keras.saving.load_model(model_path)
    prediction_colname = (
        model.layers[-1].name if model.layers[-1].name != "softmax" else "type"
    )
    backbone_model, feature_vectors = None, None
    if self.dl1_features:
        # Get the backbone model which is the second layer of the model
        backbone_model = model.get_layer(index=1)
        # Create a new head model with the same layers as the original model.
        # The output of the backbone model is the input of the head model.
        backbone_output_shape = keras.Input(model.layers[2].input_shape[1:])
        x = backbone_output_shape
        for layer in model.layers[2:]:
            x = layer(x)
        head = keras.Model(inputs=backbone_output_shape, outputs=x)
        # Apply the backbone model with the data loader to retrieve the feature vectors
        feature_vectors = backbone_model.predict(
            data_loader, verbose=self.keras_verbose
        )
        # Apply the head model with the feature vectors to retrieve the prediction
        predict_data = Table(
            {
                prediction_colname: head.predict(
                    feature_vectors, verbose=self.keras_verbose
                )
            }
        )
        # Predict the last batch and stack the results to the prediction data
        if data_loader_last_batch is not None:
            feature_vectors_last_batch = backbone_model.predict(
                data_loader_last_batch, verbose=self.keras_verbose
            )
            feature_vectors = np.concatenate(
                (feature_vectors, feature_vectors_last_batch)
            )
            predict_data = vstack(
                [
                    predict_data,
                    Table(
                        {
                            prediction_colname: head.predict(
                                feature_vectors_last_batch,
                                verbose=self.keras_verbose,
                            )
                        }
                    ),
                ]
            )
    else:
        # Predict the data using the loaded model
        predict_data = model.predict(data_loader, verbose=self.keras_verbose)
        # Create a astropy table with the prediction results
        # The classification task has a softmax layer as the last layer
        # which returns the probabilities for each class in an array, while
        # the regression tasks have output neurons which returns the
        # predicted value for the task in a dictionary.
        if prediction_colname == "type":
            predict_data = Table({prediction_colname: predict_data})
        else:
            predict_data = Table(predict_data)
        # Predict the last batch and stack the results to the prediction data
        if data_loader_last_batch is not None:
            predict_data_last_batch = model.predict(
                data_loader_last_batch, verbose=self.keras_verbose
            )
            if model.layers[-1].name == "type":
                predict_data_last_batch = Table(
                    {prediction_colname: predict_data_last_batch}
                )
            else:
                predict_data_last_batch = Table(predict_data_last_batch)
            predict_data = vstack([predict_data, predict_data_last_batch])
    return predict_data, feature_vectors