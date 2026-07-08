"""
PyTorch prediction module for LST1 telescope data.
This module provides functionality to load trained models and perform predictions
on DL1 level data for particle type classification, energy estimation, and direction reconstruction.
"""

from ctlearn.core.pytorch.net_utils import create_model, ModelHelper
import torch
from ctlearn.core.ctlearn_enum import Task, Mode
from ctapipe.io import read_table
from astropy.table import join
from dl1_data_handler.reader import get_unmapped_image
import numpy as np
from tqdm import tqdm
from pytorch_lightning.callbacks import Callback

class GPUStatsLogger(Callback):
    """
    PyTorch Lightning callback to log GPU memory statistics during training.
    
    This callback tracks GPU memory allocation and reservation at the end of each training epoch
    and logs the statistics to TensorBoard.
    """
    
    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called at the end of each training epoch to log GPU memory statistics.
        
        Args:
            trainer: PyTorch Lightning trainer instance
            pl_module: The LightningModule being trained
        """
        mem_allocated = torch.cuda.memory_allocated()
        mem_reserved = torch.cuda.memory_reserved()
        
        trainer.logger.experiment.add_scalar(
            "gpu_mem_allocated", mem_allocated, global_step=trainer.current_epoch
        )
        trainer.logger.experiment.add_scalar(
            "gpu_mem_reserved", mem_reserved, global_step=trainer.current_epoch
        )


def predictions(self):
    """
    Perform predictions on input DL1 data using trained models.
    
    This function processes the input file in batches, applies quality cuts,
    and generates predictions for particle type, energy, and/or direction
    depending on the configured tasks.
    
    Returns:
        tuple: Contains the following arrays:
            - event_id: Event identifiers
            - tel_azimuth: Telescope azimuth angles
            - tel_altitude: Telescope altitude angles
            - trigger_time: Event trigger times
            - prediction: Particle type classification scores
            - energy: Reconstructed energy values
            - cam_coord_offset_x: Camera coordinate offset in x
            - cam_coord_offset_y: Camera coordinate offset in y
            - classification_fvs: Classification feature vectors
            - energy_fvs: Energy estimation feature vectors
            - direction_fvs: Direction reconstruction feature vectors
    """
    # Optimize batch size if requested
    if self.optim_batch_size:
        batch_size_found = False
        batch = 256
        step = 16
        
        while not batch_size_found:
            from ctlearn.tools.predict.utils.optimaze_batch_size import test_batch

            # Load a test batch to find optimal batch size
            dl1_table = read_table(
                self.input_url, self.image_table_path, start=0, stop=batch
            )
            dl1_table = join(left=dl1_table, right=self.parameter_table, keys=["event_id"])
            dl1_table = join(left=dl1_table, right=self.trigger_table, keys=["event_id"])

            # Prepare test data
            data = []
            for event in dl1_table:
                image = get_unmapped_image(dl1_table[0], self.channels, self.transforms)
                data.append(self.image_mapper.map_image(image))
            input_data = {"input": np.array(data)}
            
            imgs = input_data['input'][:, :, :, 0]
            if len(self.channels) == 2:
                peak_time = input_data['input'][:, :, :, 1]

            # Test batch size for each configured task
            for task in self.tasks:
                if task == Task.type:
                    batch_size_found = not test_batch(
                        self.type_model,
                        torch.tensor(imgs).unsqueeze(1).to(self.device),
                        torch.tensor(peak_time).unsqueeze(1).to(self.device),
                        self.device
                    )
                if task == Task.energy:
                    batch_size_found = not test_batch(
                        self.energy_model,
                        torch.tensor(imgs).unsqueeze(1).to(self.device),
                        torch.tensor(peak_time).unsqueeze(1).to(self.device),
                        self.device
                    )
                if task in [Task.cameradirection, Task.skydirection, Task.direction]:
                    batch_size_found = not test_batch(
                        self.cameradirection_model,
                        torch.tensor(imgs).unsqueeze(1).to(self.device),
                        torch.tensor(peak_time).unsqueeze(1).to(self.device),
                        self.device
                    )
            
            batch += step
            if not batch_size_found:
                self.log.info(f"Batch size: {batch} OK")
        
        self.batch_size = batch - step
        self.log.info(f"Optimized batch size: {self.batch_size}")
    
    # Initialize output arrays
    event_id, tel_azimuth, tel_altitude, trigger_time = [], [], [], []
    prediction, energy, cam_coord_offset_x, cam_coord_offset_y = [], [], [], []
    classification_fvs, energy_fvs, direction_fvs = [], [], []
    
    # Process input file in batches
    for start in tqdm(range(0, self.table_length, self.batch_size), desc="Processing input file"):
        stop = min(start + self.batch_size, self.table_length)
        self.log.debug("Processing chunk from '%d' to '%d'.", start, stop - 1)
        
        # Read and join tables
        dl1_table = read_table(self.input_url, self.image_table_path, start=start, stop=stop)
        dl1_table = join(left=dl1_table, right=self.parameter_table, keys=["event_id"])
        dl1_table = join(left=dl1_table, right=self.trigger_table, keys=["event_id"])
        
        # Apply quality selection
        passes_quality_checks = np.ones(len(dl1_table), dtype=bool)
        if self.quality_query:
            passes_quality_checks = self.quality_query.get_table_mask(dl1_table)
        dl1_table = dl1_table[passes_quality_checks]
        
        if len(dl1_table) == 0:
            self.log.debug("No events passed the quality selection.")
            continue
        
        # Prepare input data
        data = []
        for event in dl1_table:
            image = get_unmapped_image(event, self.channels, self.transforms)
            data.append(self.image_mapper.map_image(image))
        input_data = {"input": np.array(data)}
        
        # Store metadata
        event_id.extend(dl1_table["event_id"].data)
        tel_azimuth.extend(dl1_table["tel_az"].data)
        tel_altitude.extend(dl1_table["tel_alt"].data)
        trigger_time.extend(dl1_table["time"].mjd)

        # Extract and clean image data
        imgs = input_data['input'][:, :, :, 0]
        if len(self.channels) == 2:
            peak_time = input_data['input'][:, :, :, 1]
            peak_time[peak_time < 0] = 0
            peak_time[np.isnan(peak_time)] = 0
            peak_time[np.isinf(peak_time)] = 0
        
        imgs[imgs < 0] = 0
        imgs[np.isnan(imgs)] = 0
        imgs[np.isinf(imgs)] = 0

        feature_vector = True
        if self.parameters["normalization"]["apply_log_scaling"][0] == True:
            imgs = imgs.astype(np.float32)
            imgs = np.log10(imgs + 1.0)
        if self.parameters["normalization"]["apply_log_scaling"][1] == True and len(self.channels) == 2:
            peak_time = peak_time.astype(np.float32)
            peak_time = np.log10(peak_time + 1.0)
        
        # Run predictions for each configured task
        for task in self.tasks:
            if task == Task.type:
                # Particle type classification
                if len(self.channels) == 2:
                    classification_pred, energy_pred, direction_pred = self.type_model(
                        torch.tensor(imgs).unsqueeze(1).to(self.device),
                        torch.tensor(peak_time).unsqueeze(1).to(self.device)
                    )
                else:
                    classification_pred, energy_pred, direction_pred = self.type_model(
                        torch.tensor(imgs).unsqueeze(1).to(self.device)
                    )
                
                prediction.extend(torch.softmax(classification_pred[0], dim=1).cpu().detach().numpy()[:, 1])
                classification_fvs.extend(classification_pred[1].cpu().detach().numpy())
                
            elif task == Task.energy:
                # Energy estimation
                if len(self.channels) == 2:
                    classification_pred, energy_pred, direction_pred = self.energy_model(
                        torch.tensor(imgs).unsqueeze(1).to(self.device),
                        torch.tensor(peak_time).unsqueeze(1).to(self.device)
                    )
                else:
                    classification_pred, energy_pred, direction_pred = self.energy_model(
                        torch.tensor(imgs).unsqueeze(1).to(self.device)
                    )
                
                energy.extend(energy_pred[0].cpu().detach().numpy())
                if feature_vector:
                    energy_fvs.extend(energy_pred[1].cpu().detach().numpy())
                else:
                    energy_fvs.extend(np.array([[0]] * len(energy_pred[0])))

            elif task in [Task.cameradirection, Task.skydirection, Task.direction]:
                # Direction reconstruction
                if len(self.channels) == 2:
                    classification_pred, energy_pred, direction_pred = self.dirrection_model(
                        torch.tensor(imgs).unsqueeze(1).to(self.device),
                        torch.tensor(peak_time).unsqueeze(1).to(self.device)
                    )
                else:
                    classification_pred, energy_pred, direction_pred = self.dirrection_model(
                        torch.tensor(imgs).unsqueeze(1).to(self.device)
                    )
                
                cam_coord_offset_x.extend(direction_pred[0][:, 0].float().cpu().detach().numpy())
                cam_coord_offset_y.extend(direction_pred[0][:, 1].float().cpu().detach().numpy())
                if feature_vector:
                    direction_fvs.extend(direction_pred[1].cpu().detach().numpy())
                else:
                    direction_fvs.extend(np.array([[0]] * len(direction_pred[0])))

            else:
                raise ValueError(
                    f"task:{task.name} is not supported. Task must be type, direction or energy"
                )
    
    return (event_id, tel_azimuth, tel_altitude, trigger_time, prediction, energy,
            cam_coord_offset_x, cam_coord_offset_y, classification_fvs, energy_fvs, direction_fvs)


def load_pytorch_model(self):
    """
    Load PyTorch models from checkpoints for the configured tasks.
    
    This function creates and loads models for particle type classification,
    energy estimation, and/or direction reconstruction based on the tasks
    specified in the configuration.
    
    Returns:
        torch.nn.Module: The last loaded model (for compatibility)
    """
    model = None
    from ctlearn.core.pytorch.model import CTLearnPyTorchModel

    def load_pytorch_model_net(model_info, task_name, num_inputs, num_outputs):
        model_name = model_info.get("model_name", "")
        try:
            component_cls = CTLearnPyTorchModel.non_abstract_subclasses().get(model_name)
            if component_cls is not None:
                params = model_info.get("parameters", {}).copy()
                params.pop("task", None)
                params.pop("num_inputs", None)
                params.pop("num_outputs", None)
                params["parent"] = self
                component = component_cls(
                    task=task_name,
                    num_inputs=num_inputs,
                    num_outputs=num_outputs,
                    **params
                )
                return component.model
        except Exception as e:
            self.log.warning(f"Failed to load model {model_name} as Component: {e}. Falling back to create_model.")
        return create_model(model_info)

    num_inputs = 1
    
    for task in self.tasks:
        # Create model based on task type
        if task == Task.type:
            model_net = load_pytorch_model_net(self.parameters["model"]["model_type"], "type", num_inputs, 2)
            check_point_path = self.parameters["data"]["type_checkpoint"]
            
        elif task == Task.energy:
            model_net = load_pytorch_model_net(self.parameters["model"]["model_energy"], "energy", num_inputs, 1)
            check_point_path = self.parameters["data"]["energy_checkpoint"]

        elif task in [Task.cameradirection, Task.skydirection, Task.direction]:
            model_net = load_pytorch_model_net(self.parameters["model"]["model_direction"], "direction", num_inputs, 3)
            check_point_path = self.parameters["data"]["direction_checkpoint"]

        else:
            raise ValueError(
                f"task:{task.name} is not supported. Task must be type, direction or energy"
            )

        # Load the model from checkpoint
        model = ModelHelper.loadModel(
            model_net, "", check_point_path, Mode.observation, device_str=self.device_str
        )
        model.eval()
        
        # Assign model to appropriate attribute
        if task == Task.type:
            self.type_model = model
        elif task == Task.energy:
            self.energy_model = model
        elif task in [Task.cameradirection, Task.skydirection, Task.direction]:
            self.dirrection_model = model
        else:
            raise ValueError(
                f"task:{task.name} is not supported. Task must be type, direction or energy"
            )
    
    return model
