import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from ctlearn.core.pytorch.nets.loss_functions.loss_functions import (
    FocalLoss,
    VectorLoss,
    AngularDistance,
    cosine_direction_loss,
)
from ctlearn.core.pytorch.nets.optimizer.optimizer import one_cycle
from tqdm import tqdm
import os
import math
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import confusion_matrix
from torchmetrics.classification import ConfusionMatrix, MulticlassPrecision, MulticlassF1Score

from io import BytesIO
import torch.nn.functional as F
from ctlearn.core.pytorch.utils import utils
from ctlearn.core.pytorch.visualization.vis_utils import (
    plot_confusion_matrix,
    plot_energy_resolution_error,
    plot_direction_resolution_error,
)

# import utils_torch
import pytorch_lightning as pl
from torchmetrics import Accuracy
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from ctlearn.core.ctlearn_enum import Task, Mode
import gc
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from ctlearn.core.pytorch.nets.loss_functions.loss_functions import evidential_regression_loss
import inspect

import matplotlib
matplotlib.use('Agg')  
torch.cuda.empty_cache()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class CTLearnTrainer(pl.Trainer):
    def __init__(self,**kwargs):
        super(CTLearnTrainer, self).__init__(**kwargs)
        self.multi_gpu=False
        if self.world_size > 1:
            self.multi_gpu=True

    def predict(
        self,
        model,
        dataloaders=None,
        return_predictions=False,
        h5_file_name="./results.r1.dl2.h5",
        task: Task = None,
        mode: Mode = None,
        **kwargs,
    ):

        # ----------------------------------------------------------------
        # Prediction
        # ----------------------------------------------------------------
        self.class_predictions = []
        self.energy_predictions = []
        self.direction_predictions = []
        self.event_id_list = []
        self.obs_id_list = []
        self.labels_energy_list = []
        self.labels_direction_list = []
        self.labels_true_alt_az_list = []
        self.hillas_list = []
        self.pointing_dir = {}
        self.h5_file_name = h5_file_name
        self.task = task
        self.mode = mode
        # super(CTLearnTrainer, self).__init__(kwargs)
        print("Using CustomTrainer's predict method.")

        if dataloaders is None:
            raise ValueError("A dataloader must be provided for prediction.")

        model.eval()
        with torch.no_grad():
            # Call your custom logic here
            results = model.generate_results(
                input_data_loader=dataloaders,
                h5_file_name=None,
                task=task,
                mode=mode,
            )

        if return_predictions:
            return results

        print("Prediction complete.")

    def get_log_dir(self) -> str:
        return self.logger.log_dir

class CTLearnPL(pl.LightningModule):

    # def setup(self,stage):

    #     # self.dummy = None 
        
    #     self.dummy = self.occupy_free_gpu_memory(self.device)

    def occupy_free_gpu_memory(self,device):
        # Get free and total GPU memory using CUDA APIs
        free_mem, total_mem = torch.cuda.mem_get_info(device)
        
        # Leave a margin (e.g., 100 MB)
        margin = 20000 * 1024 * 1024  # 1000 MB
        mem_to_allocate = max(0, free_mem - margin)
        
        if mem_to_allocate > 0:
            try:
                # Each float32 element takes 4 bytes
                num_elements = mem_to_allocate // 4
                dummy_tensor = torch.zeros(num_elements, dtype=torch.float32, device=device)
                print(f"Occupied {mem_to_allocate/1024**2:.2f} MB of GPU memory on {device}")
                return dummy_tensor
            except RuntimeError as e:
                print(f"Could not allocate memory: {e}")
                return None
        else:
            print("Not enough free GPU memory to occupy.")
            return None

    def __init__(
        self,
        model,
        save_folder,
        task: Task,
        mode: Mode,
        parameters,
        train_loader=None,
        val_loader=None,
        test_val_loader=None,        
        num_inputs=1,
        k=3,
  
    ):
        super(CTLearnPL, self).__init__()
        
        self.task = task
        self.mode = mode

        self.save_folder = save_folder
        self.model = model

        # torch.autograd.set_detect_anomaly(True)

        self._device = torch.device(parameters["arch"]["device"])
        self.device_type = parameters["arch"]["device"]

        self.model.to(self.device)


 
        # # Ejecutar la función
        # self.dummy = None 
        
        # self.occupy_free_gpu_memory(self.device)
        self.k = k  # Number of top results to save
        # Get the number of inputs of the net
        sig = inspect.signature(model.forward)
        num_inputs = len(sig.parameters)
        # Detect if model is diffusive
        if hasattr(self.model,"T"):
            self.is_difussion=True
        else:
            self.is_difussion=False
        print("Diffusion: ",self.is_difussion)
        
        self.num_inputs = num_inputs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_val_loader = test_val_loader
        self.class_names = ["gamma", "proton"]

        # Hyperparameters
        self.set_hyperparameters(parameters)

        self.optimizer = None
        self.scheduler = None
        # self.class_weights = torch.tensor([1.0, 1.3], dtype=torch.float32).to(self.device).contiguous()

        # Loss Function
        class_weights = (
             torch.tensor([parameters['class_weight'][0],parameters['class_weight'][1]], dtype=torch.float32).to(self.device).contiguous()
        )  # [1.0, 1.3]

        self.criterion_class = nn.CrossEntropyLoss(
            weight=class_weights, reduction="mean"
        )


        self.criterion_energy_value = torch.nn.L1Loss(reduction="sum")
        self.criterion_direction = torch.nn.SmoothL1Loss()  # nn.MSELoss()
        self.criterion_magnitud = torch.nn.L1Loss(reduction="mean") 

        self.criterion_direction_none = torch.nn.SmoothL1Loss(reduction="none")  # nn.MSELoss()
        self.criterion_magnitud_none = torch.nn.L1Loss(reduction="none") 
        self.criterion_alt_az_l1_none = torch.nn.L1Loss(reduction="none")
        self.criterion_energy_value_none = torch.nn.L1Loss(reduction="none")

        self.criterion_direction = torch.nn.L1Loss(reduction="mean")  # nn.MSELoss()
        self.criterion_vector = VectorLoss(alpha=0.1, reduction="mean")
        self.criterion_alt_az_l1 = torch.nn.L1Loss(reduction="mean")

        self.criterion_alt_az = evidential_regression_loss(lamb=0.01, reduction="mean")


        self.best_loss = float("inf")
        self.best_accuracy = 0
        # Best Metrics Tracking
        self.best_losses = [(float("inf"), None)] * k  
        self.best_accuracies = [(0, None)] * k   
        self.best_validation_accuracy = 0

        self.correct_classification = 0

        self.f1_score_val = MulticlassF1Score(
            num_classes=2,
            dist_sync_on_step=True,
        )

        self.f1_score_train = MulticlassF1Score(
            num_classes=2,
            dist_sync_on_step=True,
        )

        self.precision_val = MulticlassPrecision(num_classes=2,dist_sync_on_step=True,)
        self.precision_train = MulticlassPrecision(num_classes=2,dist_sync_on_step=True,)
 
        self.class_train_accuracy = Accuracy(
            task="multiclass",
            num_classes=parameters["model"]["model_type"]["parameters"]["num_outputs"],
            dist_sync_on_step=True  # GPUs Sync
        )

        self.class_val_accuracy = Accuracy(
            task="multiclass",
            num_classes=parameters["model"]["model_type"]["parameters"]["num_outputs"],
            dist_sync_on_step=True  # GPUs Sync            
        )
 
        self.confusion_matrix = ConfusionMatrix(num_classes=2, task="multiclass",dist_sync_on_step=True)

        self.loss_train_sum = 0.0
        self.num_train_batches = 0

        # ----------------------------------------------------------------
        # List used to plot on tensorboard
        # ----------------------------------------------------------------

        # ----------------------------------------------------------------
        # Validation
        # ----------------------------------------------------------------
        self.loss_val_sum = 0.0
        self.num_val_batches = 0


        self.val_angular_diff_list = []
        self.val_energy_diff_list = []

        self.val_energy_pred_list = []

        self.val_alt_pred_list = []
        self.val_az_pred_list = []
        self.val_alt_label_list = []
        self.val_az_label_list = []

        self.loss_val_distance = 0.0
        self.loss_val_dx_dy = 0.0
        self.loss_val_angular_error = 0.0

        self.val_energy_label_list = []
        self.val_hillas_intensity_list = []

        # ----------------------------------------------------------------
        # Training
        # ----------------------------------------------------------------
        self.loss_train_distance = 0.0
        self.loss_train_dx_dy = 0.0
        self.loss_train_angular_error = 0.0

        self.alt_off_list = []
        self.az_off_list = []

        # ----------------------------------------------------------------
        # Predictions
        # ----------------------------------------------------------------

        self.predictions = []

        self.class_predictions = []
        self.energy_predictions = []
        self.direction_predictions = []
        self.event_id_list = []
        self.obs_id_list = []
        self.labels_energy_list = []
        self.labels_direction_list = []
        self.labels_true_alt_az_list = []
        self.hillas_list = []
        self.pointing = []
    # ----------------------------------------------------------------------------------------------------------
    def train_dataloader(self):
        return self.train_loader
    # ----------------------------------------------------------------------------------------------------------
    def set_hyperparameters(self, parameters):
        # Hyperparameters
        self.learning_rate = float(parameters["hyp"]["learning_rate"])
        self.adam_epsilon = float(parameters["hyp"]["adam_epsilon"])
        self.momentum = float(parameters["hyp"]["momentum"])
        self.weight_decay = float(parameters["hyp"]["weight_decay"])
        self.num_epochs = int(parameters["hyp"]["epochs"])

        self.start_epoch = parameters["hyp"]["start_epoch"]
        self.steps_epoch = parameters["hyp"]["steps_epoch"]
        self.lrf = float(parameters["hyp"]["lrf"])
        self.optimizer_type = str(parameters["hyp"]["optimizer"]).lower()
        self.l2_lambda = float(parameters["hyp"]["l2_lambda"])
    # ----------------------------------------------------------------------------------------------------------
    def forward(self, x, y):
        return self.model(x, y)
    # ----------------------------------------------------------------------------------------------------------
    def save_checkpoint(self, save_folder, metric_value, filename_prefix, is_loss=True):
        """Save checkpoint and manage top k checkpoints for loss or accuracy."""
        filename = os.path.join(
            save_folder, f"{filename_prefix}_{metric_value:.16f}.pth"
        )
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metric_value": metric_value,
            },
            filename,
        )

        # Determine the list to update
        current_list = self.best_losses if is_loss else self.best_accuracies
        current_list.append((metric_value, filename))
        current_list.sort(reverse=not is_loss)
        if len(current_list) > self.k:
            removed_metric, removed_file = current_list.pop()
            if removed_file and os.path.exists(removed_file):
                # TODO: Add a Lock to avoid problems when using multiple GPUS
                try:
                    os.remove(removed_file)  # Remove the worst performing file
                except FileNotFoundError:
                    pass  # File doesn't exist, continue execution


    # ----------------------------------------------------------------------------------------------------------
    def compute_type_loss(
        self, classification_pred, labels_class, test_val=False, training=False
    ):
        target = labels_class.to(torch.int64)

        # Cálculo de la loss con F.cross_entropy
        loss_class = F.cross_entropy(classification_pred, target, weight=self.class_weights, reduction='mean')

        # Calculate accuracy
        predicted = torch.softmax(classification_pred, dim=1)
        predicted = predicted.argmax(dim=1)
        

        accuracy = 0
        precision = 0
        loss = loss_class
  
        if training:

            self.class_train_accuracy.update(predicted, labels_class)
            accuracy = self.class_train_accuracy.compute().item()
            self.f1_score_train.update(predicted, labels_class)
            self.precision_train.update(predicted, labels_class)
            precision = self.precision_train.compute().item()
        else:

            self.class_val_accuracy.update(predicted, labels_class)
            accuracy = self.class_val_accuracy.compute().item()
            self.confusion_matrix.update(predicted, labels_class)
            self.f1_score_val.update(predicted, labels_class)
            self.precision_val.update(predicted, labels_class)
            precision = self.precision_val.compute().item()
                
        return loss, accuracy, predicted, precision
    # ----------------------------------------------------------------------------------------------------------
    def compute_camera_direction_loss(self, direction_pred, labels_direction, training=False):
        
        labels_dx_dy = labels_direction[:, 0:2]
        label_distance = labels_direction[:, 2]


        if isinstance(direction_pred, tuple):
            direction_pred = list(direction_pred)

            pred_dx_dy = direction_pred[0][:,0:2].unsqueeze(-1)
            pred_distance = direction_pred[0][:,2].unsqueeze(-1)
        else:
            pred_dx_dy = direction_pred[:,0:2].unsqueeze(-1)
            pred_distance = direction_pred[:,2].unsqueeze(-1)


        loss_angular_diff = cosine_direction_loss(pred_dx_dy[:,0],pred_dx_dy[:,1], labels_dx_dy[:, 0],labels_dx_dy[:, 1])


        _, angular_diff = AngularDistance(
            pred_dx_dy[:,0],
            labels_dx_dy[:, 0],
            pred_dx_dy[:,1],
            labels_dx_dy[:, 1],
            reduction="None",
        )

        vector_cam_distance = torch.sqrt(pred_dx_dy[:,0]**2 + pred_dx_dy[:,1]**2)
        loss_dx_dy = self.criterion_alt_az_l1(pred_dx_dy, labels_dx_dy)
        loss_distance = self.criterion_magnitud(label_distance, pred_distance)
        loss_distance_dx_dy = self.criterion_magnitud(label_distance, vector_cam_distance)

        loss = loss_dx_dy + loss_distance + loss_distance_dx_dy + loss_angular_diff
        return loss, loss_dx_dy, loss_distance, loss_distance_dx_dy, loss_angular_diff, angular_diff
    # ----------------------------------------------------------------------------------------------------------
    def compute_energy_loss(
        self, energy_pred, labels_energy, test_val=False, training=False
    ):

        loss_energy = self.criterion_energy_value(energy_pred, labels_energy)
        loss = loss_energy
        #-----------------------------------------
        # k=0.6
        # e_thrs= -0.3
        # loss_energy = self.criterion_energy_value_none(energy_pred, labels_energy)
        # energy_tev = torch.pow(10,labels_energy)  
        # energy_weight=k*torch.exp(1+(np.log10(1/k)*(energy_tev-e_thrs)))
        # weight_loss = energy_weight * (loss_energy)

        # loss = weight_loss.sum()
        #-----------------------------------------

        if training == False:
            energy_pred = pow(10, energy_pred)
            labels_energy = pow(10, labels_energy)
            energy_diff = torch.abs(energy_pred - labels_energy)
            energy_diff = energy_diff.float().cpu().detach().numpy()
        else:
            energy_diff = None

        return loss, energy_diff
    # ----------------------------------------------------------------------------------------------------------
    def compute_energy_loss_diffusion(self, x_1, y,training=False, x_2 = None
        # self, energy_pred, labels_energy, test_val=False, training=False
    ):
        
        loss = 0
        # y = y.squeeze(-1)
        y_embed = self.model.target_embedder(y)
        k=0.6
        e_thrs= -0.3
        for t in range(self.model.T):
            # Add noise to target (landmarks)
            alpha_bar_t = self.model.alpha_bar[t]
            noise = torch.randn_like(y_embed)
            z_t = torch.sqrt(alpha_bar_t) * y_embed + torch.sqrt(1 - alpha_bar_t) * noise

            if x_2 is None:
                # Denoise step
                z, _ = self.model.blocks[t](x_1, z_t, None)  # W_embed not needed
            else:
                z, _ = self.model.blocks[t](x_1 ,x_2, z_t, None)  # W_embed not needed


            # Loss to clean target
            loss_l2 = F.mse_loss(z, y_embed)
            
            labels_energy = y

            # Weighted by SNR difference
            step_loss = 2.5 * self.model.eta * self.model.snr_diff[t] * loss_l2

            # Final step: use regression head
            if t == self.model.T - 1:

                energy_pred = self.model.regress(z)
                
                # labels_energy_tev = torch.pow(10,labels_energy)
                # energy_pred_tev = torch.pow(10,energy_pred)

                # loss_energy = self.criterion_energy_value(energy_pred_tev, labels_energy_tev)

                loss_energy = self.criterion_energy_value(energy_pred, labels_energy)
                # loss_energy = F.mse_loss(energy_pred, labels_energy,reduction="sum")
                # loss_energy = self.criterion_energy_value_none(energy_pred, labels_energy)
            
                # energy_weight=k*torch.exp(1+(np.log10(1/k)*(energy_tev-e_thrs)))
                # weight_loss = energy_weight * (loss_energy)

                # step_loss = step_loss + weight_loss.mean()
                
                if training == False:
                    energy_pred = pow(10, energy_pred)
                    labels_energy = pow(10, labels_energy)
                    energy_diff = torch.abs(energy_pred - labels_energy)
                    energy_diff = energy_diff.float().cpu().detach().numpy()
                else:
                    energy_diff = None

                step_loss = step_loss + loss_energy

            loss = loss + step_loss

        return loss, energy_diff, energy_pred
    # ----------------------------------------------------------------------------------------------------------
    def compute_camera_direction_loss_diffusion(self, x_1, y, labels_energy_value, x_2 = None):

        loss = 0
        y = y.squeeze(-1)
        y_embed = self.model.target_embedder(y)
        k=0.6
        e_thrs= -0.3

        for t in range(self.model.T):
            # Add noise to target (landmarks)
            alpha_bar_t = self.model.alpha_bar[t]
            noise = torch.randn_like(y_embed)
            z_t = torch.sqrt(alpha_bar_t) * y_embed + torch.sqrt(1 - alpha_bar_t) * noise

            if x_2 is None:
                # Denoise step
                z, _ = self.model.blocks[t](x_1, z_t, None)  # W_embed not needed
            else:
                z, _ = self.model.blocks[t](x_1,x_2, z_t, None)  # W_embed not needed

            preds = self.model.regress(z)
            # Loss to clean target
            loss_l2 = F.mse_loss(preds, y)

            # Weighted by SNR difference
            step_loss = 2.5 * self.model.eta * self.model.snr_diff[t] * loss_l2

            # Final step: use regression head
            if t == self.model.T - 1:
                labels_dx_dy = y[:, 0:2]
                label_distance = y[:, 2]
                direction_pred = self.model.regress(z)
                if isinstance(direction_pred, tuple):
                    # Not Tested 
                    direction_pred = list(direction_pred)

                    pred_dx_dy = direction_pred[0][:,0:2].unsqueeze(-1)
                    pred_distance = direction_pred[0][:,2].unsqueeze(-1)
                else:
                    pred_dx_dy = direction_pred[:,0:2] 
                    pred_distance = direction_pred[:,2] 

                # loss_angular_diff = cosine_direction_loss(pred_dx_dy[:,0],pred_dx_dy[:,1], labels_dx_dy[:, 0],labels_dx_dy[:, 1])
                loss_angular_diff = cosine_direction_loss(pred_dx_dy[:,0],pred_dx_dy[:,1], labels_dx_dy[:, 0],labels_dx_dy[:, 1],reduction="none")
                _, angular_diff = AngularDistance(
                    pred_dx_dy[:,0],
                    labels_dx_dy[:, 0],
                    pred_dx_dy[:,1],
                    labels_dx_dy[:, 1],
                    reduction="None",
                   
                )

                vector_cam_distance = torch.sqrt(pred_dx_dy[:,0]**2 + pred_dx_dy[:,1]**2)
                # loss_dx_dy = self.criterion_alt_az_l1(pred_dx_dy, labels_dx_dy)
                # loss_distance = self.criterion_magnitud(label_distance, pred_distance)
                # loss_distance_dx_dy = self.criterion_magnitud(label_distance, vector_cam_distance)
                loss_dx_dy = self.criterion_alt_az_l1_none(pred_dx_dy, labels_dx_dy).mean(dim=1)
                loss_distance = self.criterion_magnitud_none(label_distance, pred_distance)
                loss_distance_dx_dy = self.criterion_magnitud_none(label_distance, vector_cam_distance)

                energy = torch.pow(10,labels_energy_value)
                # k=4.3
                # e_thrs= 4
                # energy_weight = k*(1/(1+torch.exp(-(1/k)*(energy-e_thrs))))
                # weight_loss = energy_weight * (loss_dx_dy + loss_distance + loss_distance_dx_dy + loss_angular_diff)
   


                energy_weight=k*torch.exp(1+(np.log10(1/k)*(energy-e_thrs)))
                # weight_loss = energy_weight * (loss_dx_dy + loss_distance + loss_distance_dx_dy + loss_angular_diff)
                weight_loss = energy_weight * (loss_dx_dy + loss_distance + loss_distance_dx_dy)

                loss_distance=loss_distance.mean()
                loss_dx_dy = loss_dx_dy.mean()
                loss_distance_dx_dy = loss_distance_dx_dy.mean()

                loss_angular_diff = loss_angular_diff.mean()

                step_loss = step_loss + weight_loss.mean()
                # step_loss = step_loss + loss_dx_dy + loss_distance + loss_distance_dx_dy
            loss = loss + step_loss

        return loss, loss_dx_dy, loss_distance, loss_distance_dx_dy, loss_angular_diff, angular_diff
    # ----------------------------------------------------------------------------------------------------------
    def compute_sky_direction_loss_diffusion(self, x, y, labels_energy_value):

        loss = 0
        y = y.squeeze(-1)
        y_embed = self.model.target_embedder(y)

        for t in range(self.model.T):
            # Add noise to target (landmarks)
            alpha_bar_t = self.model.alpha_bar[t]
            noise = torch.randn_like(y_embed)
            z_t = torch.sqrt(alpha_bar_t) * y_embed + torch.sqrt(1 - alpha_bar_t) * noise

            # Denoise step
            z, _ = self.model.blocks[t](x, z_t, None)  # W_embed not needed
            
            preds = self.model.regress(z)
            # Loss to clean target
            loss_l2 = F.mse_loss(preds, y)

            # Weighted by SNR difference
            step_loss = 2.5 * self.model.eta * self.model.snr_diff[t] * loss_l2

            # Final step: use regression head
            if t == self.model.T - 1:


                labels_dx_dy = y[:, 0:2]
                label_distance = y[:, 2]
                direction_pred = self.model.regress(z)

                # if len(direction_pred)>1 and type(direction_pred)!=torch.Tensor:
                #     direction_pred = list(direction_pred)
                #     pred_az_atl = direction_pred[0][:,0:2]
                #     pred_separation = direction_pred[0][:,2]
                #     # direction_pred[0]= direction_pred[0][:,0:2]
                # else:

                #     pred_az_atl = direction_pred[:, 0:2]
                #     pred_separation = direction_pred[:, 2]
                
                # labels_az_alt = labels_direction[:, 0:2]
                # label_separation = labels_direction[:, 2]
                # loss_separation = self.criterion_direction(pred_separation, label_separation)

                # # loss_vector = self.criterion_vector(pred_dir_cartesian, labels_direction_cartesian)
                # # vect_magnitud = torch.sqrt(torch.sum(pred_dir_cartesian**2, dim=1))
                # # loss_magnitud = torch.abs(1.0-vect_magnitud).sum()

                # # alt_az = utils_torch.cartesian_to_alt_az(direction[:,0:3])
                # if len(direction_pred)>1 and type(direction_pred)!=torch.Tensor:
                #     loss_alt_az = self.criterion_alt_az(direction_pred, labels_direction)
                # else: 
                #     loss_alt_az = self.criterion_alt_az_l1(pred_az_atl, labels_az_alt)

                # if len(direction_pred)>1 and type(direction_pred)!=torch.Tensor:
                
                #     loss_angular_error, _ = AngularDistance(
                #         (direction_pred[0][:, 1]),
                #         labels_az_alt[:, 1],
                #         (direction_pred[0][:, 0]),
                #         labels_az_alt[:, 0],
                #         reduction="sum",
                #     )

                # else:

                #     loss_angular_error, _ = AngularDistance(
                #         (direction_pred[:, 1]),
                #         labels_az_alt[:, 1],
                #         (direction_pred[:, 0]),
                #         labels_az_alt[:, 0],
                #         reduction="sum",
                #     )

                # if training == False:
                #     if len(direction_pred)>1 and type(direction_pred)!=torch.Tensor:
                #         _, angular_diff = AngularDistance(
                #             (direction_pred[0][:, 1]),
                #             labels_az_alt[:, 0],
                #             (direction_pred[0][:, 0]),
                #             labels_az_alt[:, 1],
                #             reduction=None,
                #         )
                #     else: 
                #         _, angular_diff = AngularDistance(
                #             (direction_pred[:, 1]),
                #             labels_az_alt[:, 0],
                #             (direction_pred[:, 0]),
                #             labels_az_alt[:, 1],
                #             reduction=None,
                #         )
                # else:
                #     angular_diff = None

                # loss = loss_alt_az + 0.001*(loss_separation + loss_angular_error)

        #     loss = loss + step_loss

        # return loss, loss_dx_dy, loss_distance, loss_distance_dx_dy, loss_angular_diff, angular_diff
    # ----------------------------------------------------------------------------------------------------------    
    def compute_class_weights(self, y, n_classes):
        # y: tensor de etiquetas, por ejemplo torch.tensor([0,0,1,1,1,2])
        counts = torch.bincount(y, minlength=n_classes)
        total = counts.sum()
        # Evita división por cero:
        counts = torch.clamp(counts, min=1)
        weights = total / (counts * n_classes)
        return weights    
    # ----------------------------------------------------------------------------------------------------------        
    def compute_type_loss_diffusion(self, x,y, training=False):

        loss = 0
        # Get the embedding of the true label (e.g. "this is a 3")
        uy = self.model.W_embed[y]
        for t in range(self.model.T):

            # Get the current noise level from the schedule
            alpha_bar_t = self.model.alpha_bar[t]

            # Generate random noise with the same shape as uy
            noise = torch.randn_like(uy)

            # Add noise to the label embedding → this is our "noisy target"
            z_t = torch.sqrt(alpha_bar_t) * uy + torch.sqrt(1 - alpha_bar_t) * noise

            # Pass the image and the noisy label through one block
            z_pred, _ = self.model.blocks[t](x, z_t, self.model.W_embed)

            # Compute how far the output is from the clean label embedding
            loss_l2 = F.mse_loss(z_pred, uy)

            # Weight the loss using the signal-to-noise ratio
            step_loss = 0.5 * self.model.eta * self.model.snr_diff[t] * loss_l2


            accuracy = 0
            precision = 0
            predicted = None
            # If it's the final layer, add classification and KL losses
            if t == self.model.T - 1:
                # Get predictions from classifier
                logits = self.model.classifier(z_pred)

                # class_weights = self.compute_class_weights(y, n_classes=2).to(y.device)
                # loss_ce = F.cross_entropy(logits, y, class_weights)

                # Cross-entropy loss: how wrong the predicted class is
                loss_ce = F.cross_entropy(logits, y)

                # KL-like loss: penalize if embedding is too far from origin
                loss_kl = 0.5 * uy.pow(2).sum(dim=1).mean()

                # Add all parts together
                # loss = loss + loss_ce + loss_kl    
                step_loss = step_loss + loss_ce + loss_kl
                classification_pred,*_ = self.model(x)

                # Calculate accuracy
                predicted = torch.softmax(classification_pred, dim=1)
                predicted = predicted.argmax(dim=1)
                
                if training:
                    self.class_train_accuracy.update(predicted, y)
                    accuracy = self.class_train_accuracy.compute().item()
                    self.f1_score_train.update(predicted, y)
                    self.precision_train.update(predicted, y)
                    precision = self.precision_train.compute().item()
                else:
                    self.class_val_accuracy.update(predicted, y)
                    accuracy = self.class_val_accuracy.compute().item()
                    self.confusion_matrix.update(predicted, y)
                    self.f1_score_val.update(predicted, y)
                    self.precision_val.update(predicted, y)
                    precision = self.precision_val.compute().item()
            loss = loss + step_loss

        return loss, accuracy, predicted, precision       
    # ----------------------------------------------------------------------------------------------------------
    # def on_train_start(self):
    #      self.dummy_tensor = self.occupy_free_gpu_memory(self.device)
    # ----------------------------------------------------------------------------------------------------------
    def on_train_batch_start(self, batch, batch_idx):
        if batch_idx == 5 and not hasattr(self, "dummy_tensor"):
             self.dummy_tensor = self.occupy_free_gpu_memory(self.device)
    # ----------------------------------------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        # ------------------------------------------------------------------
        # Read inputs (features) and labels
        # ------------------------------------------------------------------
        features, labels, t = batch
        loss = 0
        if len(features) > 0:
            imgs = features["image"]
                        
            if self.task == Task.type:
                labels_class = labels["type"]

            labels_energy_value = labels["energy"]
            if self.task == Task.energy:
                labels_energy_value = labels_energy_value.to(self.device)
           

            if self.task == Task.cameradirection:
                labels_direction = labels["direction"]

            imgs = imgs.to(self.device)
            
            # ------------------------------------------------------------------
            # Predictions based on one backbone or two back bones
            # ------------------------------------------------------------------
            if not self.is_difussion:
                if self.num_inputs == 2:
                    peak_time = features["peak_time"]
                    peak_time = peak_time.to(self.device)
                    classification_pred, energy_pred, direction_pred = self.model(
                        imgs, peak_time
                    )
                else:
                    classification_pred, energy_pred, direction_pred = self.model(imgs)

            # ------------------------------------------------------------------
            # Particle type
            # ---------------------------------------
            if self.task == Task.type:
                if self.is_difussion:
                    loss, accuracy, predicted, precision = self.compute_type_loss_diffusion(imgs,labels_class,training=True)
                
                else:
                    classification_pred_ = classification_pred[0]
                    feature_vector = classification_pred[1]
                    
                    loss, accuracy, predicted, precision = self.compute_type_loss(
                        classification_pred_, labels_class, test_val=False, training=True
                )
                # Log batch loss and accuracy on the progress bar
                self.log(
                    "train_acc",
                    accuracy * 100,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    logger=False,
                )
                self.log(
                    "train_precision",
                    precision*100,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    logger=False,
                )

            # ---------------------------------------
            # Direction
            # ---------------------------------------
            if self.task == Task.cameradirection:

                if self.is_difussion:
                    if self.num_inputs == 2:
                        peak_time = features["peak_time"]
                        peak_time = peak_time.to(self.device)
                        loss, loss_dx_dy, loss_distance, loss_distance_dx_dy, loss_angular_diff, angular_diff = self.compute_camera_direction_loss_diffusion(imgs, labels_direction,labels_energy_value, peak_time)
                    else:

                        loss, loss_dx_dy, loss_distance, loss_distance_dx_dy, loss_angular_diff, angular_diff = self.compute_camera_direction_loss_diffusion(imgs, labels_direction,labels_energy_value)
                else:
                    if len(direction_pred)==2:
                        direction_pred = direction_pred[0]
                    loss, loss_dx_dy, loss_distance, loss_distance_dx_dy, loss_angular_diff, angular_diff = self.compute_camera_direction_loss( direction_pred, labels_direction, training=True)

                self.loss_train_distance += loss_distance.item() 
                self.loss_train_dx_dy += loss_dx_dy.item()
                self.loss_train_angular_error += loss_angular_diff.item()

            self.loss_val_separation = 0.0
            self.loss_val_dx_dy = 0.0
            self.loss_val_angular_error = 0.0

            # ---------------------------------------
            # Energy
            # ---------------------------------------
            if self.task == Task.energy:

                if self.is_difussion:
                    if self.num_inputs == 1:
                        loss, *_ = self.compute_energy_loss_diffusion(imgs,labels_energy_value/1.0,training=True,x_2=None)
                    else:
                        peak_time = features["peak_time"]
                        loss, *_ = self.compute_energy_loss_diffusion(imgs,labels_energy_value/1.0,training=True,x_2=peak_time)
                    
                else:
                    if len(energy_pred)==2:
                        energy_pred = energy_pred[0]
                        
                    loss, *_ = self.compute_energy_loss(
                        energy_pred, labels_energy_value, test_val=False, training=True
                    )
            # ---------------------------------------

            # L2 Regularization
            l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
            loss = loss + self.l2_lambda * l2_norm

            # Display on Bar progress
            self.log(
                "train_loss",
                loss.item(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=False,
            )

            if not np.isnan(loss.item()):

                self.loss_train_sum += loss.item()
                self.num_train_batches += 1

            # torch.cuda.empty_cache()
        return loss
    # ----------------------------------------------------------------------------------------------------------
    def on_train_epoch_end(self):
 
        if self.trainer.world_size > 1:
 
            # dist.barrier()
 
            # Gather y Sync values with the GPUs. 
            total_loss_train= self.all_gather(self.loss_train_sum).sum().item()
            total_batches_val = self.all_gather(torch.tensor(self.num_train_batches, device=self.device)).sum().item()

        else:            
            total_loss_train = self.loss_train_sum
            total_batches_val = self.num_train_batches
 
        if total_batches_val==0:
 
            return 0

        if self.trainer.is_global_zero:
 
            global_loss = total_loss_train / total_batches_val

            self.logger.experiment.add_scalars(
                "loss/Global Training Loss",
                {
                    "loss": global_loss,
                },
                self.current_epoch,
            )
 
            self.logger.experiment.add_scalars(
                "Learning rate",
                {
                    "Learning rate": self.scheduler.get_last_lr()[0],
                },
                self.current_epoch,
            )
            # Optionally print the global loss for immediate feedback
            print(f"Epoch {self.current_epoch}: Global Training Loss: {global_loss:.4f}")

        # ---------------------------------------
        # Particle Type
        # ---------------------------------------
        if self.task == Task.type:
            ii = 0 
            # f1_score = self.f1_score_train.compute().detach().cpu().numpy()*100.0
            # self.f1_score_train.reset()

            # precision = self.precision_train.compute().detach().cpu().numpy()*100.0
            # self.precision_train.reset()

            epoch_accuracy = self.class_train_accuracy.compute().detach().cpu().item() * 100        
            self.class_train_accuracy.reset()

            # # Compute the accuracy and reset the metric states after each epoch
            if self.trainer.is_global_zero:                   
            #     self.log("train_acc_epoch", epoch_accuracy, on_step=False, prog_bar=True, sync_dist=True)
            #     # Log
            #     self.logger.experiment.add_scalars(
            #         "Metrics/Training",
            #         {
            #             "acc": epoch_accuracy,
            #             "f1":f1_score,
            #             "precision":precision,
            #         },
            #         self.current_epoch,
            #     )
                print(
                    f"Epoch {self.current_epoch}: Global Training Accuracy: {epoch_accuracy:.4f}"
                )
                filename_prefix = (
                    f"Epoch_{self.current_epoch}_{self.task.name}_train_acc"
                )
                if self.logger.log_dir:
                    self.save_checkpoint(
                        self.logger.log_dir,
                        epoch_accuracy,
                        filename_prefix=filename_prefix,
                        is_loss=False,
                    )
        # ---------------------------------------
        # Direction
        # ---------------------------------------
        if self.task == Task.cameradirection and self.trainer.is_global_zero:

            filename_prefix = (
                f"Epoch_{self.current_epoch}_{self.task.name}_train_loss"
            )
            self.save_checkpoint(
                self.logger.log_dir,
                global_loss,
                filename_prefix=filename_prefix,
                is_loss=True,
            )
            # Log scalar values
            self.logger.experiment.add_scalars(
                "loss/ Loss Training",
                {
                    "loss": global_loss,
                    "loss_distance": self.loss_train_distance
                    / self.num_train_batches,
                    "loss_alt_az": self.loss_train_dx_dy / self.num_train_batches,
                    "loss_angular_error": self.loss_train_angular_error
                    / self.num_train_batches,
                },
                self.current_epoch,
            )
            self.loss_train_distance = 0.0
            self.loss_train_dx_dy = 0.0
            self.loss_train_angular_error = 0.0
        # ---------------------------------------
        # Energy
        # ---------------------------------------
        if self.task == Task.energy and self.trainer.is_global_zero:
            filename_prefix = (
                f"Epoch_{self.current_epoch}_{self.task.name}_train_loss"
            )
            self.save_checkpoint(
                self.logger.log_dir,
                global_loss,
                filename_prefix=filename_prefix,
                is_loss=True,
            )
        # ---------------------------------------
        # Delete all the lists and set to 0
        # the values used to estimate the losses
        # ---------------------------------------
        self.reset_values()
        print("END train epoch")
        # Reset
        self.loss_train_sum = 0
        self.num_train_batches = 0
        self.training_step_outputs = []
    # ----------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss=0
        self.model.eval()
        
        # ------------------------------------------------------------------
        # Read inputs (features) and labels
        # ------------------------------------------------------------------
        features, labels, t = batch

        if len(features) > 0:
            
            imgs = features["image"]

            batch_size = imgs.shape[0]
            if self.task == Task.type:
                labels_class = labels["type"]

            labels_energy_value = labels["energy"]
            hillas_intensity = features["hillas"]["hillas_intensity"]

            if self.task == Task.cameradirection:
                labels_direction = labels["direction"]


            # ------------------------------------------------------------------
            # Predictions based on one backbone or two back bones
            # ------------------------------------------------------------------
            if not self.is_difussion:
                if self.num_inputs == 2:
                    peak_time = features["peak_time"]
                    classification_pred, energy_pred, direction_pred = self.model(
                        imgs, peak_time
                    )
                else:
                    classification_pred, energy_pred, direction_pred = self.model(imgs)                
            # ------------------------------------------------------------------
            # Compute Loss functions based on different tasks
            # ------------------------------------------------------------------
            # Particle Type
            # ---------------------------------------
            if self.task == Task.type:
                if self.is_difussion:
                    loss, accuracy, predicted, precision = self.compute_type_loss_diffusion(imgs,labels_class,training=False)
                else:
                    classification_pred_ = classification_pred[0]
                    feature_vector = classification_pred[1]
                    loss, accuracy, predicted, precision = self.compute_type_loss(
                            classification_pred_,
                            labels_class,
                            test_val=False,
                            training=False,
                        )
                    
                # Log batch loss and accuracy on the progress bar
                # if dataloader_idx == 0:
                #     if self.is_difussion:
                #         loss, accuracy, predicted, precision =self.compute_type_loss_diffusion(
                #         classification_pred_,
                #         labels_class,
                #         test_val=False,
                #         training=False,
                #         )
                #     else:
                #         loss, accuracy, predicted, precision = self.compute_type_loss(
                #             classification_pred_,
                #             labels_class,
                #             test_val=False,
                #             training=False,
                #         )
                        
                    self.log(
                        "val_acc",
                        accuracy * 100,
                        on_step=True,
                        on_epoch=False,
                        prog_bar=True,
                        logger=False,
                        batch_size=batch_size,
                    )
                    self.log(
                        "val_prec",
                        precision*100,
                        on_step=True,
                        on_epoch=False,
                        prog_bar=True,
                        logger=False,
                        batch_size=batch_size,
                    )                    
            # ---------------------------------------
            # Direction
            # ---------------------------------------
            if self.task == Task.cameradirection: 

                if len(direction_pred)==2:
                    direction_pred = direction_pred[0]

                loss, loss_dx_dy, loss_distance, loss_distance_dx_dy, loss_angular_diff, angular_error= self.compute_camera_direction_loss( direction_pred, labels_direction, training=False)

                # ------------------------------------------------------------------------
                # Convert the offset to altitud and azimuth
                # ------------------------------------------------------------------------
                
                if dataloader_idx == 0:
                    self.loss_val_distance += loss_distance.item()
                    self.loss_val_dx_dy +=  loss_dx_dy.item()
                    self.loss_val_angular_error += loss_angular_diff.item()
                    self.val_angular_diff_list.extend(angular_error)
                # -------------------------------------------------------
                
                    pred_dx = direction_pred[:, 0].float().cpu().detach().numpy()
                    pred_dy = direction_pred[:, 1].float().cpu().detach().numpy()
                
                    cam_x = pred_dx
                    cam_y = pred_dy

                    pred_alt, pred_az = self.val_loader.cam_to_alt_az(labels["tel_ids"].cpu().detach().numpy(), labels["focal_length"].cpu().detach().numpy(), labels["pix_rotation"].cpu().detach().numpy(),labels["tel_az"].cpu().detach().numpy(),labels["tel_alt"].cpu().detach().numpy(), cam_x, cam_y)
                    
                    labels_dx_dy = labels_direction[:, 0:2]
                    
                    true_alt, true_az = self.val_loader.cam_to_alt_az(labels["tel_ids"].cpu().detach().numpy(), labels["focal_length"].cpu().detach().numpy(), labels["pix_rotation"].cpu().detach().numpy(),labels["tel_az"].cpu().detach().numpy(),labels["tel_alt"].cpu().detach().numpy(), labels_dx_dy[:,0].float().cpu().detach().numpy(), labels_dx_dy[:,1].float().cpu().detach().numpy())

                    self.val_alt_pred_list.extend(np.radians(pred_alt))
                    self.val_az_pred_list.extend(np.radians(pred_az))
                    self.val_alt_label_list.extend(np.radians(true_alt))
                    self.val_az_label_list.extend(np.radians(true_az))

            # ---------------------------------------
            # Energy
            # ---------------------------------------
            if self.task == Task.energy:
                if self.is_difussion:
                    if self.num_inputs == 1:
                        loss, energy_diff, energy_pred_tev = self.compute_energy_loss_diffusion(imgs,labels_energy_value/1.0,training=False,x_2=None)
                    else:
                        peak_time = features["peak_time"]
                        loss, energy_diff, energy_pred_tev = self.compute_energy_loss_diffusion(imgs,labels_energy_value/1.0,training=False,x_2=peak_time)
                else:
                    if len(energy_pred)==2:
                        energy_pred = energy_pred[0]

                    loss, energy_diff = self.compute_energy_loss(
                        energy_pred, labels_energy_value/1.0, test_val=False, training=False
                    )
                    energy_pred_tev = torch.pow(10, energy_pred*1.0)

                if dataloader_idx == 0:

                    self.val_energy_diff_list.extend(energy_diff)
                    self.val_energy_pred_list.extend(
                        energy_pred_tev[:, 0].float().cpu().detach().numpy()
                    )

            # ---------------------------------------
            # Collect the True Energy and Hillas Intensity
            # ---------------------------------------

            energy_label_tev = torch.pow(10, labels_energy_value)

            if dataloader_idx == 0:
                self.val_energy_label_list.extend(
                    energy_label_tev[:, 0].float().cpu().detach().numpy().flatten().tolist()
                )
                self.val_hillas_intensity_list.extend(
                    hillas_intensity[:, 0].float().cpu().detach().numpy().flatten().tolist()
                )

            # ---------------------------------------
            # Log validation loss
            # ---------------------------------------

            if dataloader_idx == 0:
                self.loss_val_sum += loss.item()
                self.num_val_batches += 1


        loss_key = "val_loss" if dataloader_idx == 0 else "test_loss"
        if loss is not None:
            self.log(loss_key, loss,on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)

        # Release cuda memory
        # torch.cuda.empty_cache()
        return loss
    # ----------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def on_validation_epoch_end(self):  

        if self.trainer.world_size > 1:
            dist.barrier()
            # Gather y Sync values with the GPUs. 
            total_loss_val = self.all_gather(self.loss_val_sum).sum().item()
            total_batches_val = self.all_gather(torch.tensor(self.num_val_batches, device=self.device)).sum().item()

        else:
            total_loss_val = self.loss_val_sum 
            total_batches_val = self.num_val_batches 


        if self.trainer.is_global_zero:
            # Calcular la pérdida promedio global
            global_loss_val = total_loss_val / max(1, total_batches_val)
            
            self.logger.experiment.add_scalars(
                "loss/Global Validation Loss",
                {
                    "loss": global_loss_val,
                },
                self.current_epoch,
            )

            # Print the global loss for immediate feedback
            print(f"Epoch {self.current_epoch}: Global Validation Loss: {global_loss_val:.4f}")
        # ---------------------------------------
        # Particle Type
        # ---------------------------------------
        if self.task == Task.type:
            conf_matrix = self.confusion_matrix.compute().detach().cpu().numpy()
            f1_score_val = self.f1_score_val.compute().detach().cpu().numpy()*100.0
            precision_val = self.precision_val.compute().detach().cpu().numpy()*100.0

            # Compute the accuracy and reset the metric states after each epoch
            epoch_accuracy_val = self.class_val_accuracy.compute().item() * 100

            self.class_val_accuracy.reset() 
            self.confusion_matrix.reset()
            self.f1_score_val.reset()
            self.precision_val.reset()
            
            # ---------------------------------------
            # Create Confusion Matrix
            # ---------------------------------------        
            if self.trainer.is_global_zero:

                # Log
                self.logger.experiment.add_scalars(
                    "Metrics/Validation",
                    {
                        "acc": epoch_accuracy_val,
                        "f1": f1_score_val,
                        "precision":precision_val,
                    },
                    self.current_epoch,
                )

                print(
                    f"Epoch {self.current_epoch}: Global Validation Accuracy: {epoch_accuracy_val:.4f}"
                )
                print(
                    f"Epoch {self.current_epoch}: Global Validation F1 Score: {f1_score_val:.4f}"
                )     
                print(
                    f"Epoch {self.current_epoch}: Global Validation Precision: {precision_val:.4f}"
                )                      
                filename_prefix = "confusion_matrix_val"
                cm_file_name = f"{filename_prefix}_{self.current_epoch}_{epoch_accuracy_val:.4f}_Validation"
                
                if self.logger.log_dir:
                    plot_confusion_matrix(
                        conf_matrix,
                        self.class_names,
                        cm_file_name,
                        self.logger.log_dir,  
                    )
                # Compute class-wise accuracies
                class_accuracies = (conf_matrix.diagonal() / conf_matrix.sum(axis=1))*100

                self.logger.experiment.add_scalars(
                    "confusion_matrix",
                    {
                        "val_acc_gamma": class_accuracies[0],
                        "val_acc_proton": class_accuracies[1],
                        "val_global_acc": epoch_accuracy_val,
                    },
                    self.current_epoch,
                )
                # return 0
        # ---------------------------------------
        # Direction
        # ---------------------------------------
        if self.task == Task.cameradirection and self.trainer.is_global_zero:
            
            self.print_direction_error(self.val_angular_diff_list, "Validation")

            plt.close("all")
            fig_direction_error = plot_direction_resolution_error(
                self.val_alt_pred_list,
                self.val_az_pred_list,
                self.val_alt_label_list,
                self.val_az_label_list,
                self.val_energy_label_list,
                self.val_hillas_intensity_list,
            )
            self.logger.experiment.add_figure(
                "Direction Resolution Error/Validation",
                fig_direction_error,
                self.current_epoch,
            )  # Log the plot

            # Save the figure
            fig_direction_error.savefig(
                os.path.join(
                    self.logger.log_dir,
                    "angular_resolution_validation_"
                    + str(self.current_epoch)
                    + "_"
                    + str(global_loss_val)
                    + ".png",
                ),
                format="png",
            )
            plt.close(fig_direction_error)  # Close the figure to release memory
            plt.close("all")

            # Log scalar values
            self.logger.experiment.add_scalars(
                "loss/Loss Validation",
                {
                    "loss": global_loss_val,
                    "loss_distance": self.loss_val_distance / self.num_val_batches,
                    "loss_dx_dy": self.loss_val_dx_dy / self.num_val_batches,
                    "loss_angular_error": self.loss_val_angular_error
                    / self.num_val_batches,
                },
                self.current_epoch,
            )



        # ---------------------------------------
        # Energy
        # ---------------------------------------
        if self.task == Task.energy and self.trainer.is_global_zero:

            self.print_energy_error(self.val_energy_diff_list, "Validation")
            plt.close("all")
            fig_energy_error = plot_energy_resolution_error(
                self.val_energy_pred_list,
                self.val_energy_label_list,
                self.val_hillas_intensity_list,
            )
            self.logger.experiment.add_figure(
                "Energy Resolution Error/Validation",
                fig_energy_error,
                self.current_epoch,
            )  # Log the plot
            fig_energy_error.savefig(
                os.path.join(
                    self.logger.log_dir,
                    "error_resolution_validation_"
                    + str(self.current_epoch)
                    + "_"
                    + str(global_loss_val)
                    + ".png",
                ),
                format="png",
            )

            plt.close(fig_energy_error)  # Close the figure to release memory
            plt.close("all")
            
    # ----------------------------------------------------------------------------------------------------------
    def reset_values(self):
        # ---------------------------------------
        # Reset
        # ---------------------------------------
        self.loss_val_sum = 0
        self.num_val_batches = 0

        # Reset Energy and hillas intensity
        # --------------------------------------------------
        # Validation
        # --------------------------------------------------
        self.val_energy_label_list.clear()
        self.val_hillas_intensity_list.clear()

        # --------------------------------------------------
        # Reset Direction
        # --------------------------------------------------
        # Validation
        # --------------------------------------------------
        self.val_angular_diff_list.clear()


        self.val_alt_pred_list.clear()
        self.val_az_pred_list.clear()
        self.val_alt_label_list.clear()
        self.val_az_label_list.clear()

        self.loss_val_distance = 0.0
        self.loss_val_dx_dy = 0.0
        self.loss_val_angular_error = 0.0

        # --------------------------------------------------
        # Reset Energy
        # --------------------------------------------------
        # Validation
        # --------------------------------------------------
        self.val_energy_diff_list.clear()
        self.val_energy_pred_list.clear()


    # ----------------------------------------------------------------------------------------------------------
    def print_direction_error(self, angular_diff_list, type_val: str):
        # Count the angular error in ranges [20º-0.1º]
        error_20 = len([num for num in angular_diff_list if num < 20])
        error_10 = len([num for num in angular_diff_list if num < 10])
        error_5 = len([num for num in angular_diff_list if num < 5])
        error_2 = len([num for num in angular_diff_list if num < 2])
        error_1 = len([num for num in angular_diff_list if num < 1])
        error_0_5 = len([num for num in angular_diff_list if num < 0.5])
        error_0_25 = len([num for num in angular_diff_list if num < 0.25])
        error_0_1 = len([num for num in angular_diff_list if num < 0.1])
        # Log
        self.logger.experiment.add_scalars(
            "Direction Error/" + type_val,
            {
                "0: error: 20": error_20,
                "1: error: 10": error_10,
                "2: error: 5": error_5,
                "3: error: 2": error_2,
                "4: error: 1": error_1,
                "5: error: 0.5": error_0_5,
                "6: error: 0.25": error_0_25,
                "7: error: 0.1": error_0_1,
            },
            self.current_epoch,
        )
        # Print
        print(type_val + " Direction Error < 20º: ", error_20)
        print(type_val + " Direction Error < 10º: ", error_10)
        print(type_val + " Direction Error < 5º: ", error_5)
        print(type_val + " Direction Error < 2º: ", error_2)
        print(type_val + " Direction Error < 1º: ", error_1)
        print(type_val + " Direction Error < 0.50º: ", error_0_5)
        print(type_val + " Direction Error < 0.25º: ", error_0_25)
        print(type_val + " Direction Error < 0.10º: ", error_0_1)
    # ----------------------------------------------------------------------------------------------------------
    def print_energy_error(self, energy_diff_list, type_val: str):

        
        error_30 = len([num for num in energy_diff_list if num < 30])
        error_20 = len([num for num in energy_diff_list if num < 20])
        error_10 = len([num for num in energy_diff_list if num < 10])
        error_5 = len([num for num in energy_diff_list if num < 5])
        error_1 = len([num for num in energy_diff_list if num < 1])
        error_0_5 = len([num for num in energy_diff_list if num < 0.5])
        error_0_25 = len([num for num in energy_diff_list if num < 0.25])
        error_0_1 = len([num for num in energy_diff_list if num < 0.1])
        error_0_01 = len([num for num in energy_diff_list if num < 0.01])
        # Log
        self.logger.experiment.add_scalars(
            "Energy Error/" + type_val,
            {
                "0: error: 30": error_30,
                "1: error: 20": error_20,
                "2: error: 10": error_10,
                "3: error: 5": error_5,
                "4: error: 1": error_1,
                "5: error: 0.5": error_0_5,
                "6: error: 0.25": error_0_25,
                "7: error: 0.1": error_0_1,
                "8: error: 0.01": error_0_01,
            },
            self.current_epoch,
        )
        # Print
        print(f"Total: {len(energy_diff_list)}")
        print(type_val + " Energy Error < 30:", error_30)
        print(type_val + " Energy Error < 20:", error_20)
        print(type_val + " Energy Error < 10:", error_10)
        print(type_val + " Energy Error < 5:", error_5)
        print(type_val + " Energy Error < 1:", error_1)
        print(type_val + " Energy Error < 0.5:", error_0_5)
        print(type_val + " Energy Error < 0.1:", error_0_1)
        print(type_val + " Energy Error < 0.01:", error_0_01)
    # ----------------------------------------------------------------------------------------------------------
    def create_confusion_matrix(
        self, epoch, accuracy, all_val_preds, all_val_labels, val_type: str
    ):
        all_labels = all_val_preds.numpy()
        all_preds = all_val_labels.numpy()
        filename_prefix = "confusion_matrix_val"
        cm_file_name = f"{filename_prefix}_{epoch}_{accuracy:.4f}_{val_type}"
        cm = confusion_matrix(all_labels, all_preds)

        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Remove NaNs
        cm_norm = np.nan_to_num(cm_norm, nan=0.0)
        accuracies = cm_norm.diagonal() * 100.0
        if self.logger.log_dir:
            plot_confusion_matrix(
                cm,
                self.class_names,
                accuracies,
                cm_file_name,
                self.logger.log_dir,  # self.save_folder
            )

        return accuracies
    # ----------------------------------------------------------------------------------------------------------
    def configure_optimizers(self):

        if self.optimizer_type == "sgd":
            print("Using SGD...")
            # Optimizer SGD
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=True,
            )
        elif self.optimizer_type == "adam":
            print("Using Adam...")
            # Optimizer Adam
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                eps=self.adam_epsilon,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type == "adamw":
            print("Using AdamW...")
            # Optimizer Adam
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                eps=self.adam_epsilon,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(
                f"Unsupported optimizer type: {self.optimizer_type}. Supported types are 'sgd' and 'adam'."
            )

        # Cosine phase
        # Note: The schedule is updated after execute one epoch. -> self.epochs* self.batches
        lf = one_cycle(
            1, self.lrf, self.num_epochs * self.steps_epoch
        )  # cosine 1->hyp['lrf']
        # self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)

        # TODO: Add the option to choose the kind of scheduler
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=self.lrf,total_steps=self.num_epochs * self.steps_epoch)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs
        )
        self.scheduler.last_epoch = self.start_epoch
        # Prepare scheduler dictionary as expected by PyTorch Lightning
        lr_dict = {
            "scheduler": self.scheduler,
            "interval": "epoch",  # Use for each epoch, dont use for each step
            "frequency": 1,
            "name": "OneCycleLR",
            "monitor": "val_loss",
        }
        # Return the optimizer and learning rate scheduler
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": lr_dict,
        }
    # ----------------------------------------------------------------------------------------------------------
