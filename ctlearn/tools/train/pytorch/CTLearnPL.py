import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from ctlearn.core.pytorch.nets.loss_functions.loss_functions import (
    FocalLoss,
    VectorLoss,
    AngularDistance,
    AngularError,
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
import json
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
        num_channels=1,
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
        self.num_channels = num_channels
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

        self.alpha = torch.tensor([1.0, 1.2], dtype=torch.float32) # torch.tensor([1.0, 1.1], dtype=torch.float32).to(self.device)  # Aumentamos la clase 1
        gamma = 2.0  # Aumenta la penalización en ejemplos mal clasificados  
        self.criterion_class = FocalLoss(alpha=self.alpha,gamma=gamma)


        self.criterion_energy_class = nn.CrossEntropyLoss(
            reduction="sum"
        )  # torch.nn.L1Loss(reduction='sum')
        self.criterion_energy_value = torch.nn.L1Loss(reduction="sum")
        self.criterion_direction = torch.nn.SmoothL1Loss()  # nn.MSELoss()
        # self.criterion_energy = torch.nn.MSELoss()

        self.criterion_direction = torch.nn.L1Loss(reduction="sum")  # nn.MSELoss()
        self.criterion_vector = VectorLoss(alpha=0.1, reduction="sum")
        self.criterion_alt_az_l1 = torch.nn.L1Loss(reduction="sum")
        self.criterion_alt_az = evidential_regression_loss(lamb=0.01, reduction="sum")


        self.best_loss = float("inf")
        self.best_accuracy = 0
        # Best Metrics Tracking
        self.best_losses = [(float("inf"), None)] * k  # (loss, filename)
        self.best_accuracies = [(0, None)] * k  # (accuracy, filename)
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

        self.f1_score_test = MulticlassF1Score(
            num_classes=2,
            dist_sync_on_step=True,
        )

        self.precision_val = MulticlassPrecision(num_classes=2,dist_sync_on_step=True,)
        self.precision_train = MulticlassPrecision(num_classes=2,dist_sync_on_step=True,)
        self.precision_test = MulticlassPrecision(num_classes=2,dist_sync_on_step=True,)

        self.class_train_accuracy = Accuracy(
            task="multiclass",
            num_classes=parameters["model"]["model_type"]["parameters"]["num_outputs"],
            # compute_on_step=True,  # Compute for each step and epoch 
            dist_sync_on_step=True  # GPUs Sync
        )

        self.class_val_accuracy = Accuracy(
            task="multiclass",
            num_classes=parameters["model"]["model_type"]["parameters"]["num_outputs"],
            # compute_on_step=True,  # Compute for each step and epoch 
            dist_sync_on_step=True  # GPUs Sync            
        )
        self.class_test_val_accuracy = Accuracy(
            task="multiclass",
            num_classes=parameters["model"]["model_type"]["parameters"]["num_outputs"],
            # compute_on_step=True,  # Compute for each step and epoch 
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

        self.loss_val_separation = 0.0
        self.loss_val_alt_az = 0.0
        self.loss_val_angular_error = 0.0

        self.val_energy_label_list = []
        self.val_hillas_intensity_list = []
        # ----------------------------------------------------------------
        # Test Validation
        # ----------------------------------------------------------------
        self.loss_test_val_sum = 0.0
        self.num_test_val_batches = 0
        self.all_test_val_preds = torch.tensor([])
        self.all_test_val_labels = torch.tensor([])

        self.test_val_angular_diff_list = []
        self.test_val_energy_diff_list = []

        self.test_val_energy_pred_list = []

        self.test_val_alt_pred_list = []
        self.test_val_az_pred_list = []
        self.test_val_alt_label_list = []
        self.test_val_az_label_list = []

        self.loss_test_val_separation = 0.0
        self.loss_test_val_alt_az = 0.0
        self.loss_test_val_angular_error = 0.0

        self.test_val_energy_label_list = []
        self.test_val_hillas_intensity_list = []

        # ----------------------------------------------------------------
        # Training
        # ----------------------------------------------------------------
        self.loss_train_separation = 0.0
        self.loss_train_alt_az = 0.0
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


        # loss_triplet = criterion(anchor_out, positive_out, negative_out)

        # Calculate accuracy
        predicted = torch.softmax(classification_pred, dim=1)
        predicted = predicted.argmax(dim=1)
        
        # lamb = min(1, self.trainer.current_epoch / 10)
        # class_weights= self.class_weights.to(self.device)
        # class_weights = None
        # loss_class = 0.2*evidential_classification(classification_pred, target, lamb=lamb) + 0.8 * loss_class
        # loss_class = self.evidence_loss(classification_pred, target, class_weights, lamb=lamb) 
        # Calculate accuracy
        # predicted = torch.softmax(classification_pred, dim=1)
        # predicted = classification_pred.argmax(dim=1)

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
            # Test
            if test_val:

                self.class_test_val_accuracy.update(predicted, labels_class)
                accuracy = self.class_test_val_accuracy.compute().item()
                self.f1_score_test.update(predicted, labels_class)
                self.precision_test.update(predicted, labels_class)
                precision = self.precision_test.compute().item()
            # Validation 
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
        
        if len(direction_pred)>1 and type(direction_pred)!=torch.Tensor:
            direction_pred = list(direction_pred)
            pred_az_atl = direction_pred[0][:,0:2]
            pred_separation = direction_pred[0][:,2]
            # direction_pred[0]= direction_pred[0][:,0:2]
        else:

            pred_az_atl = direction_pred[:, 0:2]
            pred_separation = direction_pred[:, 2]
        
        labels_az_alt = labels_direction[:, 0:2]
        label_separation = labels_direction[:, 2]
        loss_separation = self.criterion_direction(pred_separation, label_separation)

        # loss_vector = self.criterion_vector(pred_dir_cartesian, labels_direction_cartesian)
        # vect_magnitud = torch.sqrt(torch.sum(pred_dir_cartesian**2, dim=1))
        # loss_magnitud = torch.abs(1.0-vect_magnitud).sum()

        # alt_az = utils_torch.cartesian_to_alt_az(direction[:,0:3])
        if len(direction_pred)>1 and type(direction_pred)!=torch.Tensor:
            loss_alt_az = self.criterion_alt_az(direction_pred, labels_direction)
        else: 
            loss_alt_az = self.criterion_alt_az_l1(pred_az_atl, labels_az_alt)

        if len(direction_pred)>1 and type(direction_pred)!=torch.Tensor:
        
            loss_angular_error, _ = AngularDistance(
                (direction_pred[0][:, 1]),
                labels_az_alt[:, 1],
                (direction_pred[0][:, 0]),
                labels_az_alt[:, 0],
                reduction="sum",
            )

        else:

            loss_angular_error, _ = AngularDistance(
                (direction_pred[:, 1]),
                labels_az_alt[:, 1],
                (direction_pred[:, 0]),
                labels_az_alt[:, 0],
                reduction="sum",
            )

        if training == False:
            if len(direction_pred)>1 and type(direction_pred)!=torch.Tensor:
                _, angular_diff = AngularDistance(
                    (direction_pred[0][:, 1]),
                    labels_az_alt[:, 0],
                    (direction_pred[0][:, 0]),
                    labels_az_alt[:, 1],
                    reduction=None,
                )
            else: 
                _, angular_diff = AngularDistance(
                    (direction_pred[:, 1]),
                    labels_az_alt[:, 0],
                    (direction_pred[:, 0]),
                    labels_az_alt[:, 1],
                    reduction=None,
                )
        else:
            angular_diff = None

        loss = loss_alt_az + 0.001*(loss_separation + loss_angular_error)

        return loss, loss_separation, loss_alt_az, loss_angular_error, angular_diff
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

        # if training:
        #     accuracy = self.class_train_energy_accuracy(predicted, labels_energy_class)
        # else:
        #     if test_val:
        #         accuracy = self.class_test_energy_accuracy(predicted, labels_energy_class)
        #     else:
        #         accuracy = self.class_validation_energy_accuracy(predicted, labels_energy_class)

        return loss, energy_diff
        #
        # return loss, energy_diff
    # ----------------------------------------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):

        # ------------------------------------------------------------------
        # Read inputs (features) and labels
        # ------------------------------------------------------------------
        features, labels = batch
        loss = 0
        if len(features) > 0:
            imgs = features["image"]

            if self.task == Task.type:
                labels_class = labels["type"]

            if self.task == Task.energy:
                labels_energy_value = labels["energy"]
                labels_energy_value = labels_energy_value.to('cuda')
                # labels_energy_class = labels["energy_class"]

            if self.task == Task.direction:
                labels_direction = labels["direction"]

                labels_direction_cartesian = labels["direction_cartesian"]


            imgs = imgs.to('cuda')

            # ------------------------------------------------------------------
            # Predictions based on one backbone or two back bones
            # ------------------------------------------------------------------
            if self.num_channels == 2:
                peak_time = features["peak_time"]
                peak_time = peak_time.to('cuda')
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
            if self.task == Task.direction:
                if len(direction_pred)==2:
                    direction_pred = direction_pred[0]

                loss, loss_separation, loss_alt_az, loss_angular_error, _ = (
                    self.compute_direction_loss(
                        direction_pred, labels_direction, training=True
                    )
                )

                self.loss_train_separation += loss_separation.item()
                self.loss_train_alt_az += loss_alt_az.item()
                self.loss_train_angular_error += loss_angular_error.item()

            self.loss_val_separation = 0.0
            self.loss_val_alt_az = 0.0
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
            # TODO: No Tested 
            total_loss_train = self.loss_train_sum.item()
            total_batches_val = self.num_train_batches.item()
 
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
        if self.task == Task.direction:
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
                        "loss_separation": self.loss_train_separation
                        / self.num_train_batches,
                        "loss_alt_az": self.loss_train_alt_az / self.num_train_batches,
                        "loss_angular_error": self.loss_train_angular_error
                        / self.num_train_batches,
                    },
                    self.current_epoch,
                )
                self.loss_train_separation = 0.0
                self.loss_train_alt_az = 0.0
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

            # if self.task == Task.energy:
                # labels_energy_class = labels["energy_class"]
            labels_energy_value = labels["energy"]
            hillas_intensity = features["hillas"]["hillas_intensity"]
            # hillas = {key: tensor.to(self.device)
            #             for key, tensor in hillas.items()}
            if self.task == Task.direction:
                labels_direction = labels["direction"]
                # labels_alt_az = labels['alt_az']
                # labels_direction_cartesian = labels["direction_cartesian"]

            # ------------------------------------------------------------------
            # Predictions based on one backbone or two back bones
            # ------------------------------------------------------------------
            if self.num_channels == 2:
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
                else:
                    loss, accuracy, predicted, precision = self.compute_type_loss(
                        classification_pred_, labels_class, test_val=True, training=False
                    )
                    self.log(
                        "test_val_acc",
                        accuracy * 100,
                        on_step=True,
                        on_epoch=False,
                        prog_bar=True,
                        logger=False,
                    )

                    self.log(
                        "test_prec",
                        precision*100, 
                        on_step=True,
                        on_epoch=False,
                        prog_bar=True,
                        logger=False,
                    )   
            # ---------------------------------------
            # Direction
            # ---------------------------------------
            if self.task == Task.direction: 

                if len(direction_pred)==2:
                    direction_pred = direction_pred[0]
                # loss, angular_diff = self.compute_direction_loss(direction_pred, labels_direction, training=False)
                loss, loss_separation, loss_alt_az, loss_angular_error, angular_diff = (
                    self.compute_direction_loss(
                        direction_pred, labels_direction, training=False
                    )
                )
                # ------------------------------------------------------------------------
                # Convert the offset to altitud and azimuth
                # ------------------------------------------------------------------------
                reco_az, reco_alt = [], []
                if "tel_alt" in features:
                    pointing_alt = features["tel_alt"].float().cpu().detach().numpy()
                    pointing_az = features["tel_az"].float().cpu().detach().numpy()
                elif "tel_alt" in labels:
                    pointing_alt = labels["tel_alt"].float().cpu().detach().numpy()
                    pointing_az = labels["tel_az"].float().cpu().detach().numpy()
                else:
                    raise ValueError(f"Telescope altitud and azimuth not found.")

                pointing_alt = np.rad2deg(pointing_alt)
                pointing_az = np.rad2deg(pointing_az)

                fix_pointing = SkyCoord(
                    pointing_az * u.deg,
                    pointing_alt * u.deg,
                    frame="altaz",
                    unit="deg",
                )
                alt_off = direction_pred[:, 1].float().cpu().detach().numpy()
                az_off = direction_pred[:, 0].float().cpu().detach().numpy()

                reco_direction = utils.recover_alt_az(fix_pointing, alt_off, az_off)

                reco_alt = np.deg2rad(reco_direction.alt.to_value())[0, :]
                reco_az = np.deg2rad(reco_direction.az.to_value())[0, :]

                true_alt = labels["alt_az"][:, 0].float().cpu().detach().numpy()
                true_az = labels["alt_az"][:, 1].float().cpu().detach().numpy()

                if dataloader_idx == 0:
                    # self.alt_off_list.extend(alt_off)
                    # self.az_off_list.extend(az_off)

                    self.loss_val_separation += loss_separation.item()
                    self.loss_val_alt_az += loss_alt_az.item()
                    self.loss_val_angular_error += loss_angular_error.item()

                    self.val_angular_diff_list.extend(angular_diff)
                    # -------------------------------------------------------

                    self.val_alt_pred_list.extend(reco_alt)
                    self.val_az_pred_list.extend(reco_az)
                    self.val_alt_label_list.extend(true_alt)
                    self.val_az_label_list.extend(true_az)

                else:

                    self.loss_test_val_separation += loss_separation.item()
                    self.loss_test_val_alt_az += loss_alt_az.item()
                    self.loss_test_val_angular_error += loss_angular_error.item()

                    self.test_val_angular_diff_list.extend(angular_diff)
                    # self.test_val_alt_pred_list.extend(direction_pred[:,1].float().cpu().detach().numpy())
                    # self.test_val_az_pred_list.extend(direction_pred[:,0].float().cpu().detach().numpy())
                    # self.test_val_alt_label_list.extend(labels_direction[:,1].float().cpu().detach().numpy())
                    # self.test_val_az_label_list.extend(labels_direction[:,0].float().cpu().detach().numpy())

                    self.test_val_alt_pred_list.extend(reco_alt)
                    self.test_val_az_pred_list.extend(reco_az)
                    self.test_val_alt_label_list.extend(true_alt)
                    self.test_val_az_label_list.extend(true_az)
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
                else:
                    loss, energy_diff = self.compute_energy_loss(
                        energy_pred, labels_energy_value, test_val=True, training=False
                    )

                    self.test_val_energy_diff_list.extend(energy_diff)
                    self.test_val_energy_pred_list.extend(
                        energy_pred_tev[:, 0].float().cpu().detach().numpy()
                    )

                # ---------------------------------------
            # ---------------------------------------
            # Collect the True Energy and Hillas Intensity
            # ---------------------------------------

            energy_label_tev = torch.pow(10, labels_energy_value)

            if dataloader_idx == 0:
                self.val_energy_label_list.extend(
                    energy_label_tev[:, 0].float().cpu().detach().numpy()
                )
                self.val_hillas_intensity_list.extend(
                    hillas_intensity.float().cpu().detach().numpy()
                )
            else:
                self.test_val_energy_label_list.extend(
                    energy_label_tev[:, 0].float().cpu().detach().numpy()
                )
                self.test_val_hillas_intensity_list.extend(
                    hillas_intensity.float().cpu().detach().numpy()
                )

            # ---------------------------------------
            # Log validation loss

            if dataloader_idx == 0:
                self.loss_val_sum += loss.item()
                self.num_val_batches += 1
            else:
                self.loss_test_val_sum += loss.item()
                self.num_test_val_batches += 1

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
            total_loss_test = self.all_gather(self.loss_test_val_sum).sum().item()
            total_batches_val = self.all_gather(torch.tensor(self.num_val_batches, device=self.device)).sum().item()
            total_batches_test = self.all_gather(torch.tensor(self.num_test_val_batches, device=self.device)).sum().item()


        else:
            # TODO: No Tested 
            total_loss_val = self.loss_val_sum 
            total_loss_test = self.loss_test_val_sum 
            total_batches_val = self.num_val_batches 
            total_batches_test = self.num_test_val_batches 

            # total_loss_val = self.loss_val_sum.item()
            # total_loss_test = self.loss_test_val_sum.item()
            # total_batches_val = self.num_val_batches.item()
            # total_batches_test = self.num_test_val_batches.item()

        # Calcular la pérdida promedio global
        global_loss_val = total_loss_val / max(1, total_batches_val)
        global_loss_test = total_loss_test / max(1, total_batches_test)
       

        self.logger.experiment.add_scalars(
            "loss/Global Validation Loss",
            {
                "loss": global_loss_val,
            },
            self.current_epoch,
        )

        self.logger.experiment.add_scalars(
            "loss/Global Test Loss",
            {
                "loss": global_loss_test,
            },
            self.current_epoch,
        )

        # Print the global loss for immediate feedback
        print(
            f"Epoch {self.current_epoch}: Global Validation Loss: {global_loss_val:.4f}"
        )
        print(f"Epoch {self.current_epoch}: Global Test Loss: {global_loss_test:.4f}")
        # ---------------------------------------
        # Particle Type
        # ---------------------------------------
        if self.task == Task.type:
            conf_matrix = self.confusion_matrix.compute().detach().cpu().numpy()
            f1_score_val = self.f1_score_val.compute().detach().cpu().numpy()*100.0
            f1_score_test = self.f1_score_test.compute().detach().cpu().numpy()*100.0            
            precision_val = self.precision_val.compute().detach().cpu().numpy()*100.0
            precision_test = self.precision_test.compute().detach().cpu().numpy()*100.0


            # Compute the accuracy and reset the metric states after each epoch
            epoch_accuracy_val = self.class_val_accuracy.compute().item() * 100
            epoch_accuracy_test = self.class_test_val_accuracy.compute().item() * 100

            if self.trainer.is_global_zero:
                self.class_val_accuracy.reset()
                self.class_test_val_accuracy.reset()                
                self.confusion_matrix.reset()
                self.f1_score_val.reset()
                self.f1_score_test.reset()
                self.precision_val.reset()
                self.precision_test.reset()

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
                self.logger.experiment.add_scalars(
                    "Metrics/Test",
                    {
                        "acc": epoch_accuracy_test,
                        "f1": f1_score_test,
                        "precision":precision_test,
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

                print(
                    f"Epoch {self.current_epoch}: Global Test Accuracy: {epoch_accuracy_test:.4f}"
                )
                print(
                    f"Epoch {self.current_epoch}: Global Test F1 Score: {f1_score_test:.4f}"
                )     
                print(
                    f"Epoch {self.current_epoch}: Global Test Precision: {precision_test:.4f}"
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
        if self.task == Task.direction:

            self.print_direction_error(self.val_angular_diff_list, "Validation")
            self.print_direction_error(self.test_val_angular_diff_list, "Test")
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
            fig_direction_error = plot_direction_resolution_error(
                self.test_val_alt_pred_list,
                self.test_val_az_pred_list,
                self.test_val_alt_label_list,
                self.test_val_az_label_list,
                self.test_val_energy_label_list,
                self.test_val_hillas_intensity_list,
            )
            self.logger.experiment.add_figure(
                "Direction Resolution Error/Test",
                fig_direction_error,
                self.current_epoch,
            )  # Log the plot

            # Save the figure
            fig_direction_error.savefig(
                os.path.join(
                    self.logger.log_dir,
                    "angular_resolution_test_"
                    + str(self.current_epoch)
                    + "_"
                    + str(global_loss_test)
                    + ".png",
                ),
                format="png",
            )
            plt.close(fig_direction_error)  # Close the figure to release memory
            # Log scalar values
            self.logger.experiment.add_scalars(
                "loss/Loss Validation",
                {
                    "loss": global_loss_val,
                    "loss_separation": self.loss_val_separation / self.num_val_batches,
                    "loss_alt_az": self.loss_val_alt_az / self.num_val_batches,
                    "loss_angular_error": self.loss_val_angular_error
                    / self.num_val_batches,
                },
                self.current_epoch,
            )

            if self.test_dataloader != None:

                # Log scalar values
                self.logger.experiment.add_scalars(
                    "loss/Loss Test",
                    {
                        "loss": global_loss_test,
                        "loss_separation": self.loss_test_val_separation
                        / self.num_test_val_batches,
                        "loss_alt_az": self.loss_test_val_alt_az
                        / self.num_test_val_batches,
                        "loss_angular_error": self.loss_test_val_angular_error
                        / self.num_test_val_batches,
                    },
                    self.current_epoch,
                )

        # ---------------------------------------
        # Energy
        # ---------------------------------------
        if self.task == Task.energy:

            self.print_energy_error(self.val_energy_diff_list, "Validation")
            self.print_energy_error(self.test_val_energy_diff_list, "Test")
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
            fig_energy_error = plot_energy_resolution_error(
                self.test_val_energy_pred_list,
                self.test_val_energy_label_list,
                self.test_val_hillas_intensity_list,
            )
            self.logger.experiment.add_figure(
                "Energy Resolution Error/Test", fig_energy_error, self.current_epoch
            )  # Log the plot
            fig_energy_error.savefig(
                os.path.join(
                    self.logger.log_dir,
                    "error_resulution_test_"
                    + str(self.current_epoch)
                    + "_"
                    + str(global_loss_test)
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
        self.loss_test_val_sum = 0
        self.num_test_val_batches = 0
        # Reset Energy and hillas intensity
        # --------------------------------------------------
        # Validation
        # --------------------------------------------------
        self.val_energy_label_list.clear()

        self.val_hillas_intensity_list.clear()

        # --------------------------------------------------
        # Test validation
        # --------------------------------------------------
        self.test_val_energy_label_list.clear()

        self.test_val_hillas_intensity_list.clear()

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

        self.loss_val_separation = 0.0
        self.loss_val_alt_az = 0.0
        self.loss_val_angular_error = 0.0
        # --------------------------------------------------
        # Test validation
        # --------------------------------------------------
        self.test_val_angular_diff_list.clear()

        self.test_val_alt_pred_list.clear()
        self.test_val_az_pred_list.clear()
        self.test_val_alt_label_list.clear()
        self.test_val_az_label_list.clear()


        self.loss_test_val_separation = 0.0
        self.loss_test_val_alt_az = 0.0
        self.loss_test_val_angular_error = 0.0
        # --------------------------------------------------
        # Reset Energy
        # --------------------------------------------------
        # Validation
        # --------------------------------------------------
        self.val_energy_diff_list.clear()
        self.val_energy_pred_list.clear()

        # --------------------------------------------------
        # Test validation
        # --------------------------------------------------
        self.test_val_energy_diff_list.clear()
        self.test_val_energy_pred_list.clear()
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

        # accuracies = (np.diag(cm) / np.sum(cm, axis=0))*100.0

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
        # # Empty the list
        # all_val_preds = torch.tensor([])
        # all_val_labels = torch.tensor([])
        return accuracies
    # ----------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def generate_results(
        self,
        input_data_loader,
        h5_file_name="./results.r1.dl2.h5",
        task=None,
        mode=None,
    ):
        self.model.to(self.device)        
        self.model.eval()
        # data_loader = DataLoader(
        #     input_data_loader.dataset, batch_size=128, shuffle=False)
        dataset = input_data_loader.dataset
        data_loader = input_data_loader
        with torch.no_grad():
            pbar = tqdm(total=len(data_loader), desc="DL2 conv", leave=True)

            class_predictions = []
            class_feature_vector = []
            class_predictions_class = []
            energy_predictions = []
            direction_predictions = []
            direction_predictions_mu = []
            direction_predictions_sigma = []
            direction_pred_mu=[]
            direction_pred_sigma=[] 

            event_id_list = []
            obs_id_list = []
            labels_energy_list = []
            labels_direction_list = []
            labels_true_alt_az_list = []
            hillas_list = []
            total = 0
            correct = 0
            predicted_class = None
            direction_pred = None
            direction_pred = None
            # tel_pointing_dir=[]
            if mode == Mode.observation:  # "observation":
                dataset.pointing_dir["pointing_alt"] = []
                dataset.pointing_dir["pointing_az"] = []
                dataset.pointing_dir["dragon_time"] = []
                dataset.pointing_dir["utc_time"] = []
                dataset.pointing_dir["src_x"] = []
                dataset.pointing_dir["src_y"] = []

            cnt = 0
            for batch_idx, (features, labels) in enumerate(data_loader):
                
                # TODO: Check that features is not empty 
                if len(features)==0:
                    continue
                imgs = features["image"].to(self.device).contiguous()

                labels_class = (
                    labels["particletype"].float().to(self.device).contiguous()
                )
                hillas = features["hillas"]
                hillas = {
                    key: tensor.cpu().detach().numpy() for key, tensor in hillas.items()
                }

                tel_alt = features["tel_alt"].float().cpu().detach().numpy()
                tel_az = features["tel_az"].float().cpu().detach().numpy()

                if "time" in features:
                    tel_time = features["time"].double().cpu().detach().numpy()

                if "src_x" and "src_y" in features:
                    src_x = features["src_x"].float().cpu().detach().numpy()
                    src_y = features["src_y"].float().cpu().detach().numpy()

                if self.num_channels == 2:
                    peak_time = features["peak_time"].to(self.device).contiguous()
                    classification_pred, energy_pred, direction_pred = self.model(
                        imgs, peak_time
                    )
                else:
                    classification_pred, energy_pred, direction_pred = self.model(imgs)

                # Convert to numpy
                if task == Task.type:
                    classification_pred_ = classification_pred[0]
                    feature_vector = classification_pred[1].cpu().detach().numpy()
                    predicted = torch.softmax(classification_pred_, dim=1)
                    predicted_class = predicted.argmax(dim=1)
                    correct += (predicted_class == labels_class).sum().item()
                    predicted = predicted.cpu().detach().numpy()
                    predicted_class = predicted_class.cpu().detach().numpy()
                    total += labels_class.size(0)

                if task == Task.energy:  
                    energy = energy_pred.cpu().detach().numpy()
                    # energy = energy[0]
                    
                if task == Task.direction: 
                    if len(direction_pred)==2:
                         direction_pred_mu = direction_pred[0].cpu().detach().numpy()
                         direction_pred_sigma = direction_pred[1].cpu().detach().numpy()
                    else:
                        direction_pred = direction_pred.cpu().detach().numpy()

                obs_id = hillas["obs_id"]
                event_id = hillas["event_id"]
                labels_energy = labels["energy"].cpu().detach().numpy()
                labels_energy = 1 * (labels_energy)

                labels_direction = labels["direction"].cpu().detach().numpy()
                label_true_alt_az = labels["alt_az"].cpu().detach().numpy()

                # ------------------------------------------------------------------
                id = list(range(features["image"].shape[0]))
                if task == Task.type: 
                    class_predictions.extend(predicted[:, :])
                    class_feature_vector.extend(feature_vector[:, :])
                    class_predictions_class.extend(predicted_class[:])

                if task == Task.energy:  
                    energy_predictions.extend(energy[:, :])
                if task == Task.direction:  

                    if len(direction_pred)==2:
                        direction_predictions_mu.extend(direction_pred_mu[:,0:2])
                        direction_predictions_sigma.extend(direction_pred_sigma[:,0:2])   

                    else:
                        dir = direction_pred[:, 0:2]
                        direction_predictions.extend(dir)

                    if mode == Mode.observation:  
                        dataset.pointing_dir["pointing_alt"].extend(tel_alt)
                        dataset.pointing_dir["pointing_az"].extend(tel_az)
                        dataset.pointing_dir["utc_time"].extend(tel_time)
                        time = Time(tel_time, format="mjd")
                        datetime_utc = time.to_datetime()
                        # Convert to UNIX timestamp (dragon_time)
                        # dragon_time = datetime_utc[:].timestamp()
                        # Vectorized application of timestamp()
                        dragon_time = np.vectorize(lambda dt: dt.timestamp())(
                            datetime_utc
                        )
                        dataset.pointing_dir["dragon_time"].extend(dragon_time)

                        dataset.pointing_dir["src_x"].extend(src_x)
                        dataset.pointing_dir["src_y"].extend(src_y)

                obs_id_list.extend(obs_id)
                event_id_list.extend(event_id)

                if mode != Mode.observation:
                    labels_energy_list.extend(labels_energy[:, 0])
                    dir = labels_direction[:, 0:2]
                    labels_direction_list.extend(dir)
                    labels_true_alt_az_list.extend(label_true_alt_az)

                hillas_vector = np.array(utils.create_key_value_array(hillas, id)).T
                hillas_list.extend(hillas_vector)
                # ------------------------------------------------------------------
                if task == Task.type and mode != Mode.observation:  
                                    
                    val_acc = 100 * correct / total if total > 0 else 0.0
                    pbar.set_postfix({"val_accuracy": f"{val_acc:.2f}%"})
                    pbar.refresh()
                pbar.update(1)

                # if task == Task.type and mode != Mode.observation:  
                #     pbar.set_postfix(
                #         {
                #             "val_accuracy": f"{100 * correct / total:.2f}%",
                #         }
                #     )
                # if batch_idx % 10 == 0:
                #     pbar.update(10)

                # TESTING
                # h5_file_name="./test.dl2.h5"
                # cnt += 1
                # if cnt>5:
                #    break


        
        predictions = {
            "type": np.array(class_predictions),
            "type_feature_vector": np.array(class_feature_vector), 
            "type_class": np.array(class_predictions_class),
            "energy": np.array(energy_predictions),
            "direction": np.array(direction_predictions),
            "direction_mu": np.array(direction_predictions_mu),
            "direction_sigma": np.array(direction_predictions_sigma),
        }

        if not hasattr(dataset, "class_names"):
            dataset.class_names = ["gamma", "proton"]

        data = {
            "effective_focal_length": dataset.optics.effective_focal_length,
            "obs_id": np.array(obs_id_list),
            "event_id": np.array(event_id_list),
            "pointing": dataset.pointing_dir,
            "true_shower_primary_id": dataset.true_shower_primary_id,
            "include_nsb_patches": dataset.include_nsb_patches,
            "simulation_info": dataset.simulation_info,
            "parameter_names": dataset.hillas_names,
            "parameter_data": hillas_list,
            "mode": dataset.observation_mode,
            "class_names": dataset.class_names,
            "energy_unit": dataset.energy_unit,
            "selected_telescopes": dataset.selected_telescopes,
        }

        labels = {
            "true_energy": np.array(labels_energy_list),
            "true_direction": np.array(labels_direction_list),
            "true_alt_az": np.array(labels_true_alt_az_list),
        }

        if task == Task.type:  # "type":
            average_accuracy = 100.0 * (correct / total)
            print("Validation Accuracy: {:.2f}%".format(average_accuracy))
        
        gc.enable()
        del self.model
        del data_loader
        del input_data_loader
        del self.val_loader
        del self.scheduler

        gc.collect()

        # Save h5 file dl2 format
        utils.write_output(h5_file_name, data, predictions, labels, task, mode)

        gc.enable()
        del data
        del predictions
        del labels
        gc.collect()
        torch.cuda.empty_cache()
        
        return None
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
    # TODO: Rewrite this function in order to create a mosaic with values to study the possible problems in classification and regression
    # def missing_analysis(self):

    #     self.model.eval()
    #     total = 0
    #     correct = 0
    #     validation_loss = 0.0
    #     with torch.no_grad():
    #         pbar = tqdm(total=len(self.val_loader),
    #                     desc='Validating', leave=True)
    #         all_preds = torch.tensor([])
    #         all_labels = torch.tensor([])
    #         for batch_idx, (features, labels) in enumerate(tqdm(self.val_loader)):

    #             images = features['image'].to(self.device)
    #             labels = labels['particletype'].to(self.device)
    #             hillas_cpu = features["hillas"]
    #             # hillas = {key: tensor.to(self.device) for key, tensor in hillas_cpu.items()}
    #             hillas = {key: tensor.cpu().detach().numpy()
    #                       for key, tensor in hillas_cpu.items()}
    #             filename_list = features["filename"].cpu().detach().numpy()

    #             if self.num_channels == 2:
    #                 peak_time = features['peak_time'].to(self.device)
    #                 outputs = self.model(images, peak_time)
    #             else:
    #                 outputs = self.model(images)

    #             # outputs = self.model(images)
    #             outputs = torch.squeeze(outputs, dim=1)
    #             # labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=2).float()
    #             loss = self.criterion(outputs, labels)
    #             validation_loss += loss.item()
    #             predicted = torch.sigmoid(outputs).round()

    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()

    #             all_preds = torch.cat(
    #                 (all_preds, predicted.float().cpu()), dim=0)
    #             all_labels = torch.cat(
    #                 (all_labels, labels.float().cpu()), dim=0)

    #             images_list = []
    #             text_list = []
    #             cnt = 0
    #             for id_image in range(len(labels)):
    #                 # if (labels[id_image] == predicted[id_image]):
    #                 if not (labels[id_image] == predicted[id_image]):
    #                     cnt += 1
    #                     filename = self.convert_ascii_list_to_string(
    #                         filename_list[id_image])
    #                     # Transform image to opencv (WxHxC)
    #                     image = cv2.UMat(images[id_image].cpu(
    #                     ).detach().permute(1, 2, 0).numpy())

    #                     text_list.append("Label: "+str(self.class_names[int(labels[id_image])])+" "+"Predicted: "+str(
    #                         self.class_names[int(predicted[id_image])])+"\n"+filename)
    #                     # Transform into C,W,H
    #                     image = cv2.UMat.get(image)
    #                     images_list.append(image)
    #                     if cnt >= 16:
    #                         break

    #             canvas = self.create_image_mosaic(
    #                 images_list, text_list, batch_idx)
    #             # cv2.imshow("Missing",canvas)
    #             # cv2.waitKey(0)

    #             pbar.set_postfix({
    #                 # 'val_loss': f'{validation_loss/total:.4f}',
    #                 'val_accuracy': f'{100 * correct / total:.2f}%'
    #             })
    #             pbar.update(1)

    #     average_accuracy = 100 * (correct / total)
    #     print('Validation Accuracy: {:.2f}%'.format(average_accuracy))
    #     # Save checkpoint if this is the best accuracy or loss so far
    #     self.save_checkpoint(
    #         average_accuracy, 'validation_accuracy', is_loss=False)

    #     if average_accuracy > self.best_validation_accuracy:
    #         all_labels = all_labels.numpy()
    #         all_preds = all_preds.numpy()
    #         filename_prefix = "./confusion_matrix"
    #         cm_file_name = f'{filename_prefix}_{average_accuracy:.4f}.pth'
    #         cm = confusion_matrix(all_labels, all_preds)

    #         # accuracies = (np.diag(cm) / np.sum(cm, axis=0))*100.0

    #         cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #         accuracies = cm_norm.diagonal() * 100.0
    #         self.plot_confusion_matrix(
    #             cm, self.class_names, accuracies, cm_file_name)

    #     return average_accuracy
