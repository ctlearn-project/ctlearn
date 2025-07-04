from ctlearn.core.pytorch.net_utils import create_model, ModelHelper
from ctlearn.tools.train.pytorch.utils import (
    str_list_to_enum_list,
    sanity_check,
    read_configuration,
    create_experiment_folder,
    expected_structure,
)
import torch
from ctlearn.core.ctlearn_enum import Task, Mode
from ctapipe.io import read_table
from astropy.table import join
from dl1_data_handler.reader import (
    get_unmapped_image
)
import numpy as np
from tqdm import tqdm

from pytorch_lightning.callbacks import Callback

from ctlearn.tools.train.pytorch.CTLearnPL import CTLearnTrainer

class GPUStatsLogger(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        mem_allocated = torch.cuda.memory_allocated()
        mem_reserved = torch.cuda.memory_reserved()
        
        trainer.logger.experiment.add_scalar(
            "gpu_mem_allocated", mem_allocated, global_step=trainer.current_epoch
        )
        trainer.logger.experiment.add_scalar(
            "gpu_mem_reserved", mem_reserved, global_step=trainer.current_epoch
        )
        
def predictions(self):
    event_id, tel_azimuth, tel_altitude, trigger_time = [], [], [], []
    prediction, energy, cam_coord_offset_x, cam_coord_offset_y = [], [], [], []
    classification_fvs, energy_fvs, direction_fvs = [], [], []
    
    for start in tqdm(range(0, self.table_length, self.batch_size), desc="Procesing input file"):
        stop = min(start + self.batch_size, self.table_length)
        self.log.debug("Processing chunk from '%d' to '%d'.", start, stop - 1)
        # Read the data
        dl1_table = read_table(
            self.input_url, self.image_table_path, start=start, stop=stop
        )
        # Join the dl1 table with the parameter table to perform quality selection
        dl1_table = join(
            left=dl1_table,
            right=self.parameter_table,
            keys=["event_id"],
        )
        dl1_table = join(
            left=dl1_table,
            right=self.trigger_table,
            keys=["event_id"],
        )
        # Initialize a boolean mask to True for all events in the sliced dl1 table
        passes_quality_checks = np.ones(len(dl1_table), dtype=bool)
        # Quality selection based on the dl1b parameter
        if self.quality_query:
            passes_quality_checks = self.quality_query.get_table_mask(dl1_table)
        # Apply the mask to filter events that are not fufilling the quality criteria
        dl1_table = dl1_table[passes_quality_checks]
        if len(dl1_table) == 0:
            self.log.debug("No events passed the quality selection.")
            continue
        data = []
        for event in dl1_table:
            # Get the unmapped image
            image = get_unmapped_image(event, self.channels, self.transforms)
            data.append(self.image_mapper.map_image(image))
        input_data = {"input": np.array(data)}
        
        event_id.extend(dl1_table["event_id"].data)
        tel_azimuth.extend(dl1_table["tel_az"].data)
        tel_altitude.extend(dl1_table["tel_alt"].data)
        trigger_time.extend(dl1_table["time"].mjd)

        imgs = input_data['input'][:,:,:,0]
        peak_time = input_data['input'][:,:,:,1]
        imgs[imgs < 0] = 0
        peak_time[peak_time < 0] = 0
        imgs[np.isnan(imgs)] = 0
        imgs[np.isinf(imgs)] = 0
        peak_time[np.isnan(peak_time)] = 0
        peak_time[np.isinf(peak_time)] = 0
        
        for task in self.tasks:
            if task == Task.type:
                imgs = (imgs - self.type_mu) / self.type_sigma
                peak_time = (peak_time - self.type_mu) / self.type_sigma
                                
                if len(self.channels) == 2:
                    classification_pred, energy_pred, direction_pred = self.type_model(
                        torch.tensor(imgs).unsqueeze(1).to(self.device) , torch.tensor(peak_time).unsqueeze(1).to(self.device) 
                    )
                else:
                    classification_pred, energy_pred, direction_pred = self.type_model(input_data['input'][:,:,:,0])
                    
                prediction.extend(torch.softmax(classification_pred[0],dim=1).cpu().detach().numpy()[:,0])
                classification_fvs.extend(classification_pred[1].cpu().detach().numpy())
                
            elif task == Task.energy:
                
                imgs = (imgs - self.energy_mu) / self.energy_sigma
                peak_time = (peak_time - self.energy_mu) / self.energy_sigma
                
                if len(self.channels) == 2:
                    classification_pred, energy_pred, direction_pred = self.energy_model(
                        torch.tensor(imgs).unsqueeze(1).to(self.device) , torch.tensor(peak_time).unsqueeze(1).to(self.device) 
                    )
                else:
                    classification_pred, energy_pred, direction_pred = self.energy_model(input_data['input'][:,:,:,0])
                    
                energy.extend(energy_pred[0].cpu().detach().numpy())
                energy_fvs.extend(energy_pred[1].cpu().detach().numpy())
            elif task == Task.cameradirection or task == Task.skydirection or task == Task.direction:
                
                imgs = (imgs - self.dir_mu) / self.dir_sigma
                peak_time = (peak_time - self.dir_mu) / self.dir_sigma
                
                if len(self.channels) == 2:
                    classification_pred, energy_pred, direction_pred = self.dirrection_model(
                        torch.tensor(imgs).unsqueeze(1).to(self.device) , torch.tensor(peak_time).unsqueeze(1).to(self.device) 
                    )
                else:
                    classification_pred, energy_pred, direction_pred = self.dirrection_model(input_data['input'][:,:,:,0])
                    
                cam_coord_offset_x.extend(direction_pred[0][0][:,0].float().cpu().detach().numpy())
                cam_coord_offset_y.extend(direction_pred[0][0][:,1].float().cpu().detach().numpy())
                direction_fvs.extend(direction_pred[1].cpu().detach().numpy())
            else:
                raise ValueError(
                    f"task:{task.name} is not supported. Task must be type, direction or energy"
                )
    return event_id, tel_azimuth, tel_altitude, trigger_time, prediction, energy, cam_coord_offset_x, cam_coord_offset_y, classification_fvs, energy_fvs, direction_fvs

def load_pytorch_model(self):
    for task in self.tasks:
        if task == Task.type:
            model_net = create_model(self.parameters["model"]["model_type"])
            
        elif task == Task.energy:
            model_net = create_model(self.parameters["model"]["model_energy"])

        elif task == Task.cameradirection or task == Task.skydirection or task == Task.direction:
            model_net = create_model(self.parameters["model"]["model_direction"])

        else:
            raise ValueError(
                f"task:{task.name} is not supported. Task must be type, direction or energy"
            )

        # ------------------------------------------------------------------------------
        # Load Checkpoints
        # ------------------------------------------------------------------------------
        
        if task == Task.type:
            check_point_path = self.parameters["data"]["type_checkpoint"]

        elif task == Task.energy:
            check_point_path = self.parameters["data"]["energy_checkpoint"]

        elif task == Task.cameradirection or task == Task.skydirection or task == Task.direction:
            check_point_path = self.parameters["data"]["direction_checkpoint"]

        else:
            raise ValueError(
                f"task:{task.name} is not supported. Task must be type, direction or energy"
            )
        # Load the checkpoint
        
        model = ModelHelper.loadModel(
            model_net, "", check_point_path, Mode.observation, device_str=self.device_str
        )
        
        model.eval()
        
    return model