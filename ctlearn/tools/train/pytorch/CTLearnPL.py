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
# from ctlearn.nets.loss_functions.loss_functions import evidential_classification
 

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
                h5_file_name=h5_file_name,
                task=task,
                mode=mode,
            )

        if return_predictions:
            return results

        print("Prediction complete.")

    def get_log_dir(self) -> str:
        return self.logger.log_dir

class CTLearnPL(pl.LightningModule):

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

        self.k = k  # Number of top results to save
        # Get the number of inputs of the net
        sig = inspect.signature(model.forward)
        num_inputs = len(sig.parameters)
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
             torch.tensor([1.0, 1.3], dtype=torch.float32).to(self.device).contiguous()
        )  # [1.0, 1.3]

        self.criterion_class = nn.CrossEntropyLoss(
            weight=class_weights, reduction="mean"
        )

        self.alpha = torch.tensor([1.0, 1.2], dtype=torch.float32)  
        gamma = 2.0  # Increase the penalty on misclassified examples.

        self.criterion_class = FocalLoss(alpha=self.alpha,gamma=gamma)


        self.criterion_energy_class = nn.CrossEntropyLoss(
            reduction="sum"
        )   
        self.criterion_energy_value = torch.nn.L1Loss(reduction="mean")
        self.criterion_direction = torch.nn.SmoothL1Loss()  # nn.MSELoss()
        self.criterion_magnitud = torch.nn.L1Loss(reduction="mean") 
        

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
        self.criterion_class.set_alpha(self.alpha.to(self.device))

        target = labels_class.to(torch.int64)
        loss_class = self.criterion_class(classification_pred, target)

        # Calculate accuracy
        predicted = torch.softmax(classification_pred, dim=1)
        predicted = predicted.argmax(dim=1)
        

        accuracy = 0
        precision = 0
        loss = loss_class
        # loss = alpha*loss_class+(1-alpha)*loss_triplet
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
    def compute_direction_loss(self, direction_pred, labels_direction, training=False):
        
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
        # return loss, loss_separation, loss_alt_az, loss_angular_error, angular_diff


        
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
        if training == False:
            energy_pred = pow(10, energy_pred)
            labels_energy = pow(10, labels_energy)
            energy_diff = torch.abs(energy_pred - labels_energy)
            energy_diff = energy_diff.float().cpu().detach().numpy()
        else:
            energy_diff = None

        return loss, energy_diff
    # ----------------------------------------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):

        # ------------------------------------------------------------------
        # Read inputs (features) and labels
        # ------------------------------------------------------------------
        features, labels = batch
        loss = 0


        # self.trainer.datamodule.train_dataloader().cam_to_alt_az(labels["tel_ids"], labels["focal_length"], labels["pix_rotation"],labels["tel_az"],labels["tel_alt"], cam_x, cam_y)
        if len(features) > 0:
            imgs = features["image"]

            if self.task == Task.type:
                labels_class = labels["type"]

            if self.task == Task.energy:
                labels_energy_value = labels["energy"]
                labels_energy_value = labels_energy_value.to(self.device)

            if self.task == Task.cameradirection:
                labels_direction = labels["direction"]

            imgs = imgs.to(self.device)
            
            # ------------------------------------------------------------------
            # Predictions based on one backbone or two back bones
            # ------------------------------------------------------------------
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
                if len(direction_pred)==2:
                    direction_pred = direction_pred[0]

                loss, loss_dx_dy, loss_distance, loss_distance_dx_dy, loss_angular_diff, angular_diff = self.compute_direction_loss( direction_pred, labels_direction, training=True)

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
                loss, *_ = self.compute_energy_loss(
                    energy_pred, labels_energy_value, training=True
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

            torch.cuda.empty_cache()
        return loss
    # ----------------------------------------------------------------------------------------------------------
    def on_train_epoch_end(self):
        
        if self.trainer.world_size > 1:
            dist.barrier()
            # Gather y Sync values with the GPUs. 
            total_loss_train= self.all_gather(self.loss_train_sum).sum().item()
            total_batches_val = self.all_gather(torch.tensor(self.num_train_batches, device=self.device)).sum().item()

        else:            
            total_loss_train = self.loss_train_sum
            total_batches_val = self.num_train_batches
 
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

            f1_score = self.f1_score_train.compute().detach().cpu().numpy()*100.0
            self.f1_score_train.reset()

            precision = self.precision_train.compute().detach().cpu().numpy()*100.0
            self.precision_train.reset()

            epoch_accuracy = self.class_train_accuracy.compute().detach().cpu().item() * 100        
            self.class_train_accuracy.reset()

            # Compute the accuracy and reset the metric states after each epoch
            if self.trainer.is_global_zero:                   
                self.log("train_acc_epoch", epoch_accuracy, on_step=False, prog_bar=True)
                # Log
                self.logger.experiment.add_scalars(
                    "Metrics/Training",
                    {
                        "acc": epoch_accuracy,
                        "f1":f1_score,
                        "precision":precision,
                    },
                    self.current_epoch,
                )
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
        if self.task == Task.cameradirection:
            if self.trainer.is_global_zero:
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
        if self.task == Task.energy:
            if self.trainer.is_global_zero:
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
        features, labels = batch
        if len(features) > 0:
            imgs = features["image"]

            if self.task == Task.type:
                labels_class = labels["particletype"]

            labels_energy_value = labels["energy"]
            hillas_intensity = features["hillas"]["hillas_intensity"]

            if self.task == Task.cameradirection:
                labels_direction = labels["direction"]


            # ------------------------------------------------------------------
            # Predictions based on one backbone or two back bones
            # ------------------------------------------------------------------
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
                classification_pred_ = classification_pred[0]
                feature_vector = classification_pred[1]
                # Log batch loss and accuracy on the progress bar
                if dataloader_idx == 0:
                    loss, accuracy, predicted, precision = self.compute_type_loss(
                        classification_pred_,
                        labels_class,
                        test_val=False,
                        training=False,
                    )
                    self.log(
                        "val_acc",
                        accuracy * 100,
                        on_step=True,
                        on_epoch=False,
                        prog_bar=True,
                        logger=False,
                    )
                    self.log(
                        "val_prec",
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

                if len(direction_pred)==2:
                    direction_pred = direction_pred[0]

                loss, loss_dx_dy, loss_distance, loss_distance_dx_dy, loss_angular_diff, angular_error= self.compute_direction_loss( direction_pred, labels_direction, training=False)

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

                    pred_alt, pred_az = self.val_loader.cam_to_alt_az(labels["tel_ids"], labels["focal_length"], labels["pix_rotation"],labels["tel_az"],labels["tel_alt"], cam_x, cam_y)

                    true_alt = labels["true_alt"]
                    true_az = labels["true_az"]
                    self.val_alt_pred_list.extend(np.radians(pred_alt))
                    self.val_az_pred_list.extend(np.radians(pred_az))
                    self.val_alt_label_list.extend(np.radians(true_alt))
                    self.val_az_label_list.extend(np.radians(true_az))

            # ---------------------------------------
            # Energy
            # ---------------------------------------
            if self.task == Task.energy:

                energy_pred_tev = torch.pow(10, energy_pred)

                if dataloader_idx == 0:

                    loss, energy_diff = self.compute_energy_loss(
                        energy_pred, labels_energy_value, test_val=False, training=False
                    )

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
                    hillas_intensity.float().cpu().detach().numpy().flatten().tolist()
                )

            # ---------------------------------------
            # Log validation loss
            # ---------------------------------------

            if dataloader_idx == 0:
                self.loss_val_sum += loss.item()
                self.num_val_batches += 1


        loss_key = "val_loss" if dataloader_idx == 0 else "test_loss"
        if loss is not None:
            self.log(loss_key, loss,on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Release cuda memory
        torch.cuda.empty_cache()
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
        if self.task == Task.type and self.trainer.is_global_zero:
            conf_matrix = self.confusion_matrix.compute().detach().cpu().numpy()
            f1_score_val = self.f1_score_val.compute().detach().cpu().numpy()*100.0
            precision_val = self.precision_val.compute().detach().cpu().numpy()*100.0

            # Compute the accuracy and reset the metric states after each epoch
            epoch_accuracy_val = self.class_val_accuracy.compute().item() * 100

            if self.trainer.is_global_zero:
                self.class_val_accuracy.reset()
                     
                self.confusion_matrix.reset()
                self.f1_score_val.reset()
           
                self.precision_val.reset()
     

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
          
                # ---------------------------------------
                # Create Confusion Matrix
                # ---------------------------------------        
                if self.trainer.is_global_zero:

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
                    "error_resulution_validation_"
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
