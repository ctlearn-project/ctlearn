"""
Tool to train a PyTorch-based ``CTLearnModel`` on R1/DL1a data using the ``DLDataReader`` and ``PyTorchDataLoader``.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

from ctlearn.core.pytorch.dataset import PyTorchDataset
from ctlearn.core.model import CTLearnModel
from ctlearn.tools.train_model import TrainCTLearnModel


class TrainCTLearnPyTorchModel(TrainCTLearnModel):
    """
    Tool to train a ``~ctlearn.core.model.CTLearnModel`` PyTorch-based model on R1/DL1a data.

    The tool trains a CTLearn PyTorch-based model on the input data (R1 calibrated waveforms or DL1a images) and
    saves the trained model in the output directory. The input data is loaded from the input directories
    for signal and background events using the ``~dl1_data_handler.reader.DLDataReader`` and
    ``~dl1_data_handler.loader.DLDataLoader``. The tool supports the following reconstruction tasks:
    - Classification of the primary particle type (gamma/proton)
    - Regression of the primary particle energy
    - Regression of the primary particle arrival direction based on the offsets in camera coordinates
    - Regression of the primary particle arrival direction based on the offsets in sky coordinates
    """

    name = "ctlearn-train-pytorch-model"
    description = __doc__

    examples = """
    To train a PyTorch-based CTLearn model for the classification of the primary particle type:
    > ctlearn-train-pytorch-model \\
        --signal /path/to/your/gammas_dl1_dir/ \\
        --pattern-signal "gamma_*_run1.dl1.h5" \\
        --pattern-signal "gamma_*_run10.dl1.h5" \\
        --background /path/to/your/protons_dl1_dir/ \\
        --pattern-background "proton_*_run1.dl1.h5" \\
        --pattern-background "proton_*_run10.dl1.h5" \\
        --output /path/to/your/type/ \\
        --reco type \\

    To train a PyTorch-based CTLearn model for the regression of the primary particle energy:
    > ctlearn-train-pytorch-model \\
        --signal /path/to/your/gammas_dl1_dir/ \\
        --pattern-signal "gamma_*_run1.dl1.h5" \\
        --pattern-signal "gamma_*_run10.dl1.h5" \\
        --output /path/to/your/energy/ \\
        --reco energy \\
    
    To train a PyTorch-based CTLearn model for the regression of the primary particle
    arrival direction based on the offsets in camera coordinates:
    > ctlearn-train-pytorch-model \\
        --signal /path/to/your/gammas_dl1_dir/ \\
        --pattern-signal "gamma_*_run1.dl1.h5" \\
        --pattern-signal "gamma_*_run10.dl1.h5" \\
        --output /path/to/your/direction/ \\
        --reco cameradirection \\

    To train a PyTorch-based CTLearn model for the regression of the primary particle
    arrival direction based on the offsets in sky coordinates:
    > ctlearn-train-pytorch-model \\
        --signal /path/to/your/gammas_dl1_dir/ \\
        --pattern-signal "gamma_*_run1.dl1.h5" \\
        --pattern-signal "gamma_*_run10.dl1.h5" \\
        --output /path/to/your/direction/ \\
        --reco skydirection \\
    """

    def setup_framework(self):
        # Determine available hardware device (Multi-GPU / Single GPU / CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.num_devices = torch.cuda.device_count()
            self.log.info("Using CUDA device(s). Available GPUs: %d", self.num_devices)
        else:
            self.device = torch.device("cpu")
            self.num_devices = 1
            self.log.info("Using CPU device.")


        # Set seed globally for reproducibility across random operations
        torch.manual_seed(self.random_seed)
        # Create a dedicated Generator for the DataLoader
        g = torch.Generator()
        g.manual_seed(self.random_seed)

        # Init the PyTorchDataLoader
        self.training_dataset = PyTorchDataset(
            DLDataReader=self.dl1dh_reader,
            indices=self.training_indices,
            tasks=self.reco_tasks,
            sort_by_intensity=self.sort_by_intensity,
            stack_telescope_images=self.stack_telescope_images,
        )
        self.training_loader = DataLoader(
            dataset=self.training_dataset,
            batch_size=self.batch_size * self.num_devices,
            shuffle=True, # Enables shuffling
            generator=g, # Controls the shuffling seed deterministically
            pin_memory=torch.cuda.is_available() # Accelerates memory copy from host CPU to GPU
        )
        self.validation_dataset = PyTorchDataset(
            DLDataReader=self.dl1dh_reader,
            indices=self.validation_indices,
            tasks=self.reco_tasks,
            sort_by_intensity=self.sort_by_intensity,
            stack_telescope_images=self.stack_telescope_images,
        )
        self.validation_loader = DataLoader(
            dataset=self.validation_dataset,
            batch_size=self.batch_size * self.num_devices,
            shuffle=False, # Disables shuffling
            pin_memory=torch.cuda.is_available() # Accelerates memory copy from host CPU to GPU
        )

        # Set up TensorBoard writers for train and validation and CSV logging path
        self.train_writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "train"))
        self.val_writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "validation"))
        self.csv_log_path = os.path.join(self.output_dir, "training_log.csv")

        # Initialize TorchMetrics according to reconstruction tasks
        self.train_metrics = self._get_task_metrics()
        self.val_metrics = self._get_task_metrics()

        # Build CSV dynamic header matching Keras logger format
        self.csv_headers = ["epoch"]
        if "type" in self.reco_tasks:
            self.csv_headers.extend(["accuracy", "auc"])
        if "energy" in self.reco_tasks:
            self.csv_headers.append("mae_energy")
        if "cameradirection" in self.reco_tasks:
            self.csv_headers.append("mae_cameradirection")
        if "skydirection" in self.reco_tasks:
            self.csv_headers.append("mae_skydirection")
        self.csv_headers.append("loss")

        # Add validation metrics to header
        if "type" in self.reco_tasks:
            self.csv_headers.extend(["val_accuracy", "val_auc"])
        if "energy" in self.reco_tasks:
            self.csv_headers.append("val_mae_energy")
        if "cameradirection" in self.reco_tasks:
            self.csv_headers.append("val_mae_cameradirection")
        if "skydirection" in self.reco_tasks:
            self.csv_headers.append("val_mae_skydirection")
        self.csv_headers.append("val_loss")

        # Write CSV header if file doesn't exist
        if not os.path.exists(self.csv_log_path):
            with open(self.csv_log_path, "w") as f:
                f.write(",".join(self.csv_headers) + "\n")

    def start(self):
        self.log.info("Setting up the PyTorch model.")
        base_model = CTLearnModel.from_name(
            f"PyTorch{self.model_type}",
            input_shape=self.training_dataset.input_shape,
            tasks=self.reco_tasks,
            parent=self,
        ).model

        base_model.to(self.device)

        if self.device.type == "cuda" and self.num_devices > 1:
            self.model = nn.DataParallel(base_model)
        else:
            self.model = base_model

        optimizers = {
            "Adadelta": lambda params: torch.optim.Adadelta(params, lr=self.learning_rate),
            "Adam": lambda params: torch.optim.Adam(params, lr=self.learning_rate, eps=self.adam_epsilon),
            "RMSProp": lambda params: torch.optim.RMSprop(params, lr=self.learning_rate),
            "SGD": lambda params: torch.optim.SGD(params, lr=self.learning_rate),
        }
        self.opt = optimizers[self.optimizer["name"]](self.model.parameters())

        # Setup Learning Rate Scheduler
        self.scheduler = None
        if self.lr_reducing is not None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.opt,
                mode="min",
                factor=self.lr_reducing["factor"],
                patience=self.lr_reducing["patience"],
                threshold=self.lr_reducing["min_delta"],
                min_lr=self.lr_reducing["min_lr"],
            )

        self.loss_fns = self._get_loss_functions()

        self.log.info("Training and evaluating...")
        
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_weights = None
        model_save_path = os.path.join(self.output_dir, "ctlearn_model.pth")
        state_dict_save_path = os.path.join(self.output_dir, "ctlearn_state_dict.pth")

        for epoch_idx in range(self.n_epochs):
            train_loss, train_metric_vals = self._train_epoch()
            val_loss, val_metric_vals = self._validate_epoch()

            # Record CSV Row (0-indexed epochs matching Keras CSVLogger)
            row_dict = {"epoch": epoch_idx, "loss": train_loss, "val_loss": val_loss}
            for k, v in train_metric_vals.items():
                row_dict[k] = v
            for k, v in val_metric_vals.items():
                row_dict[f"val_{k}"] = v

            row_str = ",".join(str(row_dict[col]) for col in self.csv_headers)
            with open(self.csv_log_path, "a") as f:
                f.write(row_str + "\n")

            # TensorBoard metrics training and validation
            self.train_writer.add_scalar("loss", train_loss, epoch_idx)
            for k, v in train_metric_vals.items():
                self.train_writer.add_scalar(k, v, epoch_idx)
            self.train_writer.flush()
            self.val_writer.add_scalar("loss", val_loss, epoch_idx)
            for k, v in val_metric_vals.items():
                self.val_writer.add_scalar(k, v, epoch_idx)
            self.val_writer.flush()

            # Unwrap model if wrapped with DataParallel / DistributedDataParallel
            unwrapped_model = (
                self.model.module
                if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel))
                else self.model
            )

            # Checkpoint saving & Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                state_dict = (
                    self.model.module.state_dict()
                    if isinstance(self.model, nn.DataParallel)
                    else self.model.state_dict()
                )
                best_model_weights = state_dict
                if self.save_best_validation_only:
                    # Save state_dict (tensors only)
                    torch.save(state_dict, state_dict_save_path)
                    # Save full model (nn.Module object)
                    torch.save(unwrapped_model, model_save_path)
            else:
                patience_counter += 1

            if not self.save_best_validation_only:
                state_dict = (
                    self.model.module.state_dict()
                    if isinstance(self.model, nn.DataParallel)
                    else self.model.state_dict()
                )
                # Save state_dict (tensors only)
                torch.save(state_dict, state_dict_save_path)
                # Save full model (nn.Module object)
                torch.save(unwrapped_model, model_save_path)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            if self.early_stopping is not None:
                if patience_counter >= self.early_stopping["patience"]:
                    self.log.info("Early stopping triggered at epoch %d.", epoch_idx)
                    if (
                        self.early_stopping["restore_best_weights"]
                        and best_model_weights is not None
                    ):
                        unwrapped_model = (
                            self.model.module
                            if isinstance(self.model, nn.DataParallel)
                            else self.model
                        )
                        unwrapped_model.load_state_dict(best_model_weights)
                    break

        # Close the TensorBoard writers
        self.train_writer.close()
        self.val_writer.close()
        self.log.info("Training and evaluating finished successfully!")

    def _train_epoch(self):
        self.model.train()
        for m in self.train_metrics.values():
            m.reset()

        total_loss = 0.0

        for batch_x, batch_y in self.training_loader:
            batch_x = self._to_device(batch_x)
            batch_y = self._to_device(batch_y)

            self.opt.zero_grad()
            outputs = self.model(batch_x)

            loss = self._compute_combined_loss(outputs, batch_y)
            loss.backward()
            self.opt.step()

            total_loss += loss.item()
            self._update_metrics(self.train_metrics, outputs, batch_y)

        avg_loss = total_loss / len(self.training_loader)
        metric_results = {k: m.compute().item() for k, m in self.train_metrics.items()}
        return avg_loss, metric_results

    def _validate_epoch(self):
        self.model.eval()
        for m in self.val_metrics.values():
            m.reset()

        total_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in self.validation_loader:
                batch_x = self._to_device(batch_x)
                batch_y = self._to_device(batch_y)

                outputs = self.model(batch_x)
                loss = self._compute_combined_loss(outputs, batch_y)

                total_loss += loss.item()
                self._update_metrics(self.val_metrics, outputs, batch_y)

        avg_loss = total_loss / len(self.validation_loader)
        metric_results = {k: m.compute().item() for k, m in self.val_metrics.items()}
        return avg_loss, metric_results

    def _get_task_metrics(self):
        """Instantiates TorchMetrics matching Keras metric definitions."""
        metrics = {}
        if "type" in self.reco_tasks:
            num_classes = getattr(self.dl1dh_reader, "num_classes", 2)
            metrics["accuracy"] = torchmetrics.Accuracy(
                task="multiclass" if num_classes > 2 else "binary",
                num_classes=num_classes,
            ).to(self.device)
            metrics["auc"] = torchmetrics.AUROC(
                task="multiclass" if num_classes > 2 else "binary",
                num_classes=num_classes,
            ).to(self.device)
        if "energy" in self.reco_tasks:
            metrics["mae_energy"] = torchmetrics.MeanAbsoluteError().to(self.device)
        if "cameradirection" in self.reco_tasks:
            metrics["mae_cameradirection"] = torchmetrics.MeanAbsoluteError().to(self.device)
        if "skydirection" in self.reco_tasks:
            metrics["mae_skydirection"] = torchmetrics.MeanAbsoluteError().to(self.device)
        return metrics
    
    def _update_metrics(self, metrics, outputs, targets):
        for task in self.reco_tasks:
            out = outputs[task] if isinstance(outputs, dict) else outputs
            tgt = targets[task] if isinstance(targets, dict) else targets
            task_metrics = metrics[task] if task in metrics else metrics

            if task == "type":
                if out.ndim > 1 and out.shape[-1] > 1:
                    out = out[:, 1]  # Positive class probability for binary metrics
            else:
                if out.ndim > 1 and out.shape[-1] == 1:
                    out = out.squeeze(-1)
                if tgt.ndim > 1 and tgt.shape[-1] == 1:
                    tgt = tgt.squeeze(-1)

            if isinstance(task_metrics, dict):
                for metric_obj in task_metrics.values():
                    metric_obj.update(out, tgt)
            else:
                task_metrics.update(out, tgt)

    def _compute_combined_loss(self, outputs, targets):
        total_loss = 0.0

        for task in self.reco_tasks:
            # Extract output tensor safely
            task_output = outputs[task] if isinstance(outputs, dict) else outputs
            task_target = targets[task] if isinstance(targets, dict) else targets

            # Align shapes for regression tasks (energy, direction)
            if task != "type":
                if task_output.ndim > 1 and task_output.shape[-1] == 1:
                    task_output = task_output.squeeze(-1)
                if task_target.ndim > 1 and task_target.shape[-1] == 1:
                    task_target = task_target.squeeze(-1)

            task_loss = self.loss_fns[task](task_output, task_target)
            total_loss += task_loss

        return total_loss

    def _get_loss_functions(self):
        loss_fns = {}
        if "type" in self.reco_tasks:
            weight = None
            if self.dl1dh_reader.class_weight is not None:
                class_weights = self.dl1dh_reader.class_weight
                
                # Convert dict to a list ordered by class index
                if isinstance(class_weights, dict):
                    class_weights = [class_weights[k] for k in sorted(class_weights.keys())]

                weight = torch.tensor(
                    class_weights,
                    dtype=torch.float32,
                    device=self.device,
                )
            
            loss_fns["type"] = torch.nn.CrossEntropyLoss(weight=weight)
        if "energy" in self.reco_tasks:
            loss_fns["energy"] = nn.L1Loss()
        if "cameradirection" in self.reco_tasks:
            loss_fns["cameradirection"] = nn.L1Loss()
        if "skydirection" in self.reco_tasks:
            loss_fns["skydirection"] = nn.L1Loss()
        return loss_fns

    def _to_device(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {k: self._to_device(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._to_device(v) for v in data]
        return data


def main():
    tool = TrainCTLearnPyTorchModel()
    tool.run()


if __name__ == "__main__":
    main()