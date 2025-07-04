from ctlearn.core.data_loader.loader import DLDataLoader
import keras 
from astropy.table import (
    Table,
    hstack,
    vstack,
    join,
    setdiff,
)
from tqdm import tqdm

import numpy as np
import inspect

from ctlearn.tools.predict.utils.load_model import load_model

def predict_with_model_pytorch(self, task):
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
    self.batch_size = self.parameters["hyp"]["batches"]

    data_loader = DLDataLoader.create(
        framework="pytorch",
        DLDataReader=self.dl1dh_reader,
        indices=self.indices,
        tasks=[task],
        parameters = self.parameters,
        use_augmentation = False,
        batch_size=self.batch_size,
        sort_by_intensity=self.sort_by_intensity,
        stack_telescope_images=self.stack_telescope_images,
    )
    
    # Keras is only considering the last complete batch.
    # In prediction mode we don't want to loose the last
    # uncomplete batch, so we are creating an additional
    # batch generator for the remaining events.
    # data_loader_last_batch = None
    # if self.last_batch_size > 0:
    #     last_batch_indices = self.indices[-self.last_batch_size :]
    #     data_loader_last_batch = DLDataLoader.create(
    #         framework="pytorch",
    #         DLDataReader=self.dl1dh_reader,
    #         indices=last_batch_indices,
    #         tasks=task,
    #         parameters = self.parameters,
    #         use_augmentation = False,
    #         batch_size=self.last_batch_size,
    #         sort_by_intensity=self.sort_by_intensity,
    #         stack_telescope_images=self.stack_telescope_images,
    #     )

    # Load the model from the specified path
    model = load_model(self)
    sig = inspect.signature(model.forward)
    num_inputs = len(sig.parameters)
    predict_data = {}
    predict_data['type'] = []
    predict_data['energy'] = []
    predict_data["cameradirection"] = []
    
    model.eval() 
    for x in tqdm(data_loader, desc="Processing", total=len(data_loader)):
        if len(x[0]['image'])==0:
            continue
        if num_inputs == 2:
            classification_pred, energy_pred, direction_pred = model(x[0]['image'].to(self.device) ,x[0]['peak_time'].to(self.device))
        else:
            classification_pred, energy_pred, direction_pred = model(x[0]['image'].to(self.device))
        
        if classification_pred is not None:
            predict_data['type'].extend(classification_pred.cpu().detach().numpy())
        if energy_pred is not None:
            predict_data['energy'].extend(energy_pred.cpu().detach().numpy())
        if direction_pred is not None:
            predict_data["cameradirection"].extend(direction_pred.cpu().detach().numpy())

    predict_data["cameradirection"] = np.array(predict_data["cameradirection"])
    predict_data["type"] = np.array(predict_data["type"])
    predict_data["energy"] = np.array(predict_data["energy"])
    return predict_data , None