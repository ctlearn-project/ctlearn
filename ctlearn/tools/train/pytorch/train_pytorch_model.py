from ctapipe.core.traits import Path, Bool, Unicode, ComponentName
from torch.utils.data import DataLoader
from ctlearn.tools.train.pytorch.CTLearnPL import CTLearnTrainer, CTLearnPL
from ctlearn.core.model import CTLearnModel
import ctlearn.core.pytorch.model
import sys
import os

is_debug = '--debug' in sys.argv or any(arg.startswith('--log-level=DEBUG') for arg in sys.argv)

if not is_debug:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
    os.environ['NCCL_DEBUG'] = 'WARN'       # Suppress verbose NCCL networking logs

import warnings
warnings.filterwarnings("ignore", ".*AccumulateGrad node's stream does not match.*")
warnings.filterwarnings("ignore", ".*torch.distributed.nn.functional.all_gather is deprecated.*")
warnings.filterwarnings("ignore", ".*does not have many workers which may be a bottleneck.*")
warnings.filterwarnings("ignore", ".*isinstance.treespec, LeafSpec.*")
warnings.filterwarnings("ignore", ".*This axis already has a converter set and is updating.*")

if not is_debug:
    # Silence noisy external library warnings
    warnings.filterwarnings("ignore", ".*NoneDefaultNotAllowedWarning.*")
    warnings.filterwarnings("ignore", ".*MergeConflictWarning.*")
    warnings.filterwarnings("ignore", ".*'ctlearn.tools.train_model' found in sys.modules.*")

try:
    import torch
    if hasattr(torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch"):
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
except ImportError:
    raise ImportError("pytorch is not installed in your environment!")

try:
    from pytorch_lightning.loggers import TensorBoardLogger
except ImportError:
    raise ImportError("pytorch_lightning is not installed in your environment!")

from ctlearn.tools.train.base_train_model import TrainCTLearnModel
from ctlearn.core.ctlearn_enum import Task, Mode
from .utils import (
    str_list_to_enum_list,
    sanity_check,
    read_configuration,
    create_experiment_folder,
    expected_structure,
)

from ctlearn.core.pytorch.net_utils import create_model, ModelHelper
from ctlearn.core.data_loader.loader import DLDataLoader
from pytorch_lightning.callbacks import Callback
import os 
import numpy as np 
import json


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

# from ctlearn.tools.train_model import
class TrainPyTorchModel(TrainCTLearnModel):
    """
    Tool to train a ``~ctlearn.core.model.CTLearnModel`` on R1/DL1a data using PyTorch.

    The tool sets up the PyTorch model using ... The PyTorch model is trained
    on the input data (R1 calibrated waveforms or DL1a images) and saved in the output directory.
    """

    name = "ctlearn-train-pytorch-model"
    description = __doc__

    examples = """
    To train a CTLearn PyTorch model for the classification of the primary particle type:
    > ctlearn-train-pytorch-model \\
        --signal /path/to/your/gammas_dl1_dir/ \\
        --pattern-signal "gamma_*_run1.dl1.h5" \\
        --pattern-signal "gamma_*_run10.dl1.h5" \\
        --background /path/to/your/protons_dl1_dir/ \\
        --pattern-background "proton_*_run1.dl1.h5" \\
        --pattern-background "proton_*_run10.dl1.h5" \\
        --output /path/to/your/type/ \\
        --reco type \\

    To train a CTLearn PyTorch model for the regression of the primary particle energy:
    > ctlearn-train-pytorch-model \\
        --signal /path/to/your/gammas_dl1_dir/ \\
        --pattern-signal "gamma_*_run1.dl1.h5" \\
        --pattern-signal "gamma_*_run10.dl1.h5" \\
        --output /path/to/your/energy/ \\
        --reco energy \\

    To train a CTLearn PyTorch model for the regression of the primary particle
    arrival direction based on the offsets in camera coordinates:
    > ctlearn-train-pytorch-model \\
        --signal /path/to/your/gammas_dl1_dir/ \\
        --pattern-signal "gamma_*_run1.dl1.h5" \\
        --pattern-signal "gamma_*_run10.dl1.h5" \\
        --output /path/to/your/direction/ \\
        --reco cameradirection \\

    To train a CTLearn PyTorch model for the regression of the primary particle
    arrival direction based on the offsets in sky coordinates:
    > ctlearn-train-pytorch-model \\
        --signal /path/to/your/gammas_dl1_dir/ \\
        --pattern-signal "gamma_*_run1.dl1.h5" \\
        --pattern-signal "gamma_*_run10.dl1.h5" \\
        --output /path/to/your/direction/ \\
        --reco skydirection \\
    """

    config_file = Path(
        exits=True,
        default_value=None,
        allow_none=True,
        directory_ok=True,
        file_ok=True,
        help="Configuration file.",
    ).tag(config=True)

    disable_progress_bar = Bool(
        default_value=False,
        help="Disable PyTorch Lightning progress bar.",
    ).tag(config=True)

    model_type = ComponentName(
        CTLearnModel, default_value="PyTorchResNet"
    ).tag(config=True)

    aliases = {
        **TrainCTLearnModel.aliases,
        "config_file": "TrainPyTorchModel.config_file",
        "disable_progress_bar": "TrainPyTorchModel.disable_progress_bar",
        "model-type": "TrainPyTorchModel.model_type",
        "model-name": "TrainPyTorchModel.model_type",
    }

    def __init__(self, **kwargs):
        
        # Setup GPU 
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["NCCL_DEBUG"] = "WARN"
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        torch.set_float32_matmul_precision('medium')        
 
        super().__init__(**kwargs)
 

    def setup(self):
 
        super().setup()

        # Create tasks Enum List
        self.tasks = str_list_to_enum_list(self.reco_tasks)

        for task_ in self.tasks:
            print("Task:", task_.name)

        if self.config_file is not None:
            self.log.info("Loading configuration from legacy PyTorch config file: %s", self.config_file)
            legacy_params = read_configuration(self.config_file)
            sanity_check(legacy_params, expected_structure)
            
            def get_conf_val(key1, key2, trait_name, default_val):
                in_config = False
                for cls_name in ["TrainPyTorchModel", "TrainCTLearnModel"]:
                    if cls_name in self.config and trait_name in self.config[cls_name]:
                        in_config = True
                        break
                if not in_config:
                    return legacy_params.get(key1, {}).get(key2, default_val)
                return getattr(self, trait_name)
                
            self.type_checkpoint = get_conf_val("data", "type_checkpoint", "type_checkpoint", self.type_checkpoint)
            self.energy_checkpoint = get_conf_val("data", "energy_checkpoint", "energy_checkpoint", self.energy_checkpoint)
            self.direction_checkpoint = get_conf_val("data", "direction_checkpoint", "direction_checkpoint", self.direction_checkpoint)
            self.load_onnx_model = get_conf_val("data", "load_onnx_model", "load_onnx_model", self.load_onnx_model)
            
            self.experiment_number = get_conf_val("run_details", "experiment_number", "experiment_number", self.experiment_number)
            self.save_onnx = get_conf_val("run_details", "save_onnx", "save_onnx", self.save_onnx)
            self.save_k_checkpoints = get_conf_val("hyp", "save_k", "save_k_checkpoints", self.save_k_checkpoints)
            self.device_str = get_conf_val("arch", "device", "device", self.device)
            self.batch_size = get_conf_val("hyp", "batches", "batch_size", self.batch_size)
            self.pin_memory = get_conf_val("dataset", "pin_memory", "pin_memory", self.pin_memory)
            self.num_workers = get_conf_val("dataset", "num_workers", "num_workers", self.num_workers)
            self.persistent_workers = get_conf_val("dataset", "persistent_workers", "persistent_workers", self.persistent_workers)
            self.devices = get_conf_val("arch", "devices", "devices", self.devices)
            self.strategy = get_conf_val("arch", "strategy", "strategy", self.strategy)
            
            # Augmentations
            self.use_augmentation = get_conf_val("augmentation", "use_augmentation", "use_augmentation", self.use_augmentation)
            self.aug_prob = get_conf_val("augmentation", "aug_prob", "aug_prob", self.aug_prob)
            self.rot_prob = get_conf_val("augmentation", "rot_prob", "rot_prob", self.rot_prob)
            self.trans_prob = get_conf_val("augmentation", "trans_prob", "trans_prob", self.trans_prob)
            self.flip_hor_prob = get_conf_val("augmentation", "flip_hor_prob", "flip_hor_prob", self.flip_hor_prob)
            self.flip_ver_prob = get_conf_val("augmentation", "flip_ver_prob", "flip_ver_prob", self.flip_ver_prob)
            self.mask_prob = get_conf_val("augmentation", "mask_prob", "mask_prob", self.mask_prob)
            self.mask_dvr_prob = get_conf_val("augmentation", "mask_dvr_prob", "mask_dvr_prob", self.mask_dvr_prob)
            self.noise_prob = get_conf_val("augmentation", "noise_prob", "noise_prob", self.noise_prob)
            self.max_rot = get_conf_val("augmentation", "max_rot", "max_rot", self.max_rot)
            self.max_trans = get_conf_val("augmentation", "max_trans", "max_trans", self.max_trans)

            # Normalizations
            self.use_clean = get_conf_val("normalization", "use_clean", "use_clean", self.use_clean)
            self.use_clean_dvr = get_conf_val("normalization", "use_clean_dvr", "use_clean_dvr", self.use_clean_dvr)
            self.type_mu = get_conf_val("normalization", "type_mu", "type_mu", self.type_mu)
            self.type_sigma = get_conf_val("normalization", "type_sigma", "type_sigma", self.type_sigma)
            self.dir_mu = get_conf_val("normalization", "dir_mu", "dir_mu", self.dir_mu)
            self.dir_sigma = get_conf_val("normalization", "dir_sigma", "dir_sigma", self.dir_sigma)
            self.energy_mu = get_conf_val("normalization", "energy_mu", "energy_mu", self.energy_mu)
            self.energy_sigma = get_conf_val("normalization", "energy_sigma", "energy_sigma", self.energy_sigma)
            
            # Cut-offs
            self.leakage_intensity_cutoff = get_conf_val("cut-off", "leakage_intensity", "leakage_intensity_cutoff", self.leakage_intensity_cutoff)
            self.intensity_cutoff = get_conf_val("cut-off", "intensity", "intensity_cutoff", self.intensity_cutoff)

            self.hyp_configs = legacy_params.get("hyp", {})
        else:
            self.log.info("No legacy config file provided. Using standard Traitlets configuration.")
            self.device_str = self.device
            self.hyp_configs = {
                "epochs": self.n_epochs,
                "batches": self.batch_size,
                "dynamic_batches": True,
                "optimizer": self.optimizer.get("name", "Adam"),
                "momentum": self.optimizer_momentum,
                "weight_decay": self.optimizer_weight_decay,
                "learning_rate": self.optimizer.get("base_learning_rate", 0.0001),
                "lrf": self.lrf,
                "start_epoch": 0,
                "steps_epoch": 100,
                "l2_lambda": self.l2_lambda,
                "adam_epsilon": self.optimizer.get("adam_epsilon", 1.0e-8),
                "gradient_clip_val": self.gradient_clip_val,
                "save_k": self.save_k_checkpoints,
            }

        self.save_k = self.save_k_checkpoints

        self.parameters = {
            "data": {
                "train_gamma_proton": None,
                "validation_gamma_proton": None,
                "train_gamma": None,
                "validation_gamma": None,
                "test_gamma": None,
                "test_proton": None,
                "test_electron": None,
                "test_validation_gamma": None,
                "test_validation_gamma_proton": None,
                "type_checkpoint": self.type_checkpoint,
                "energy_checkpoint": self.energy_checkpoint,
                "direction_checkpoint": self.direction_checkpoint,
                "load_onnx_model": self.load_onnx_model,
            },
            "run_details": {
                "mode": "train",
                "task": self.reco_tasks[0] if self.reco_tasks else "all",
                "test_type": "gamma",
                "experiment_number": self.experiment_number,
            },
            "cut-off": {
                "leakage_intensity": self.leakage_intensity_cutoff,
                "intensity": self.intensity_cutoff,
            },
            "model": self.model_type,
            "hyp": self.hyp_configs,
            "augmentation": {
                "use_augmentation": self.use_augmentation,
                "aug_prob": self.aug_prob,
                "rot_prob": self.rot_prob,
                "trans_prob": self.trans_prob,
                "flip_hor_prob": self.flip_hor_prob,
                "flip_ver_prob": self.flip_ver_prob,
                "mask_prob": self.mask_prob,
                "mask_dvr_prob": self.mask_dvr_prob,
                "noise_prob": self.noise_prob,
                "max_rot": self.max_rot,
                "max_trans": self.max_trans,
            },
            "normalization": {
                "apply_log_scaling": self.apply_log_scaling,
                "use_clean": self.use_clean,
                "use_clean_dvr": self.use_clean_dvr,
                "type_mu": self.type_mu,
                "type_sigma": self.type_sigma,
                "dir_mu": self.dir_mu,
                "dir_sigma": self.dir_sigma,
                "energy_mu": self.energy_mu,
                "energy_sigma": self.energy_sigma,
            },
            "dataset": {
                "num_workers": self.num_workers,
                "pin_memory": self.pin_memory,
                "persistent_workers": self.persistent_workers,
            },
            "arch": {
                "device": self.device_str,
                "precision_type": self.precision_type,
                "precision_energy": self.precision_energy,
                "precision_direction": self.precision_direction,
                "devices": self.devices,
                "strategy": self.strategy,
            }
        }

        print(f"Using Devices: {self.devices}")

        # Set up the data loaders for training and validation
        indices = list(range(self.dl1dh_reader._get_n_events()))
        # Shuffle the indices before the training/validation split
        np.random.seed(self.random_seed)
        np.random.shuffle(indices)
        n_validation_examples = int(
            self.validation_split * self.dl1dh_reader._get_n_events()
        )
        training_indices = indices[n_validation_examples:]
        validation_indices = indices[:n_validation_examples]

        if not ("class_weight" in self.parameters):
            self.parameters['class_weight'] = self.dl1dh_reader.class_weight
            self.log.info(f"Class weights not provided. Using class weights from data reader: {self.parameters['class_weight']}")
        elif len(self.parameters['class_weight']) != len(self.dl1dh_reader.class_names):
            raise ValueError(f"Number of class weights provided ({len(self.parameters['class_weight'])}) does not match number of classes in data ({len(self.dl1dh_reader.class_names)}).")
        else:
            self.log.info(f"Using class weights from configuration file: {self.parameters['class_weight']}")
            
        print("BASE TRAIN FRAMEWORK", self.framework_type)

        self.train_dataset = DLDataLoader.create(
            framework=self.framework_type,
            DLDataReader=self.dl1dh_reader,
            indices=training_indices,
            tasks=self.reco_tasks,
            batch_size=self.batch_size,
            random_seed=self.random_seed,
            sort_by_intensity=self.sort_by_intensity,
            stack_telescope_images=self.stack_telescope_images,
            parameters=self.parameters,
            use_augmentation=self.use_augmentation,
            is_training=True,
        )
        self.training_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=None,
            batch_sampler=None,
            num_workers=self.num_workers,       
            pin_memory=self.pin_memory, 
            prefetch_factor=4 if self.num_workers > 0 else None,    
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False
        )
        
        print(len(self.training_loader))
        
        self.validation_dataset = DLDataLoader.create(
            framework=self.framework_type,
            DLDataReader=self.dl1dh_reader,
            indices=validation_indices,
            tasks=self.reco_tasks,
            batch_size=self.batch_size,
            random_seed=self.random_seed,
            sort_by_intensity=self.sort_by_intensity,
            stack_telescope_images=self.stack_telescope_images,
            parameters=self.parameters,
            use_augmentation=False,
            is_training=False,
        )
        self.validation_loader = DataLoader(
            dataset=self.validation_dataset,
            batch_size=None,
            batch_sampler=None,
            num_workers=self.num_workers,       
            pin_memory=self.pin_memory, 
            prefetch_factor=4 if self.num_workers > 0 else None,    
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False
        )
        
        print(len(self.validation_dataset))


    def start(self):
        super().start()
        for task in self.tasks:

            # Create the experiment folder
            save_folder = create_experiment_folder(
                f"run_{task.name}_training_", next_number=self.experiment_number
            )

            # Save the resolved configuration parameters to a config file in the experiment directory
            config_filename = os.path.join(save_folder, "resolved_config.yml")
            with open(config_filename, "w") as outfile:
                import yaml
                yaml.dump(self.parameters, outfile, default_flow_style=False)

            # ------------------------------------------------------------------------------
            # Select the model and precision
            # ------------------------------------------------------------------------------
            import torch.nn as nn
            import torch
            torch.backends.cudnn.enabled = False

            if task == Task.type:
                precision = self.parameters["arch"]["precision_type"]
            elif task == Task.energy:
                precision = self.parameters["arch"]["precision_energy"]
            elif task == Task.cameradirection or task == Task.skydirection:
                precision = self.parameters["arch"]["precision_direction"]
            else:
                raise ValueError(
                    f"task:{task.name} is not supported. Task must be type, direction or energy"
                )

            class ONNXModelWrapper(nn.Module):
                def __init__(self, onnx_model_net, active_task, onnx_input_shape):
                    super().__init__()
                    self.onnx_model_net = onnx_model_net
                    self.active_task = active_task
                    self.onnx_input_shape = onnx_input_shape
                    
                    self.onnx_channel_last = False
                    self.expected_channels = 1
                    if len(onnx_input_shape) == 4:
                        if onnx_input_shape[3] in [1, 2, 3] and onnx_input_shape[1] > onnx_input_shape[3]:
                            self.onnx_channel_last = True
                            self.expected_channels = onnx_input_shape[3]
                        else:
                            self.expected_channels = onnx_input_shape[1]

                def forward(self, x, y=None):
                    if self.expected_channels == 2 and y is not None:
                        x = torch.cat([x, y], dim=1)
                        
                    import inspect
                    sig = inspect.signature(self.onnx_model_net.forward)
                    num_onnx_inputs = len(sig.parameters)
                    
                    if num_onnx_inputs == 2 and y is not None:
                        if self.onnx_channel_last:
                            x = x.permute(0, 2, 3, 1)
                            y = y.permute(0, 2, 3, 1)
                        out = self.onnx_model_net(x, y)
                    else:
                        if self.onnx_channel_last:
                            x = x.permute(0, 2, 3, 1)
                        out = self.onnx_model_net(x)
                        
                    if isinstance(out, (tuple, list)):
                        if len(out) == 3:
                            return out
                        val = out[0]
                    else:
                        val = out
                        
                    if self.active_task == Task.type:
                        return val, None, None
                    elif self.active_task == Task.energy:
                        return None, val, None
                    else:
                        return None, None, val

            if self.load_onnx_model:
                self.log.info(f"Loading ONNX model from {self.load_onnx_model} for training...")
                self.log.warning(
                    "WARNING: Training an ONNX model from scratch (untrained) in PyTorch using onnx2pytorch "
                    "often leads to frozen gradients and the loss not improving. This is because onnx2pytorch "
                    "is designed primarily for inference, and many operations break the computational graph. "
                    "It is highly recommended to train the native PyTorch model instead by removing the "
                    "--load_onnx_model flag."
                )
                import onnx
                from onnx2pytorch import ConvertModel
                try:
                    onnx_proto = onnx.load(self.load_onnx_model)
                    onnx_model = ConvertModel(onnx_proto)
                    onnx_input_shape = [dim.dim_value for dim in onnx_proto.graph.input[0].type.tensor_type.shape.dim]
                    model_net = ONNXModelWrapper(onnx_model, task, onnx_input_shape)
                except Exception as e:
                    self.log.error(f"Failed to load ONNX model: {e}")
                    raise e
            else:
                self.log.info("Setting up the PyTorch model.")
                num_inputs = 1
                if isinstance(self.parameters.get("model"), dict):
                    num_inputs = self.parameters["model"].get(f"model_{task.name}", {}).get("parameters", {}).get("num_inputs", 1)
                
                model_input_shape = list(self.train_dataset.input_shape)
                if len(model_input_shape) == 3:
                    # Convert (H, W, C) to PyTorch's (C, H, W)
                    model_input_shape = [model_input_shape[2], model_input_shape[0], model_input_shape[1]]
                elif len(model_input_shape) >= 4:
                    # e.g., (Telescopes, H, W, C) -> (Telescopes, C, H, W)
                    model_input_shape = list(model_input_shape[:-3]) + [model_input_shape[-1], model_input_shape[-3], model_input_shape[-2]]
                
                model_net = CTLearnModel.from_name(
                    self.model_type,
                    input_shape=tuple(model_input_shape),
                    tasks=[task.name],
                    parent=self,
                ).model

            # if hasattr(model_net, 'T'):
            #     self.training_loader.set_T(model_net.T)
            # ------------------------------------------------------------------------------
            # Load Checkpoints
            # ------------------------------------------------------------------------------
            if task == Task.type:
                check_point_path = self.parameters["data"]["type_checkpoint"]

            elif task == Task.energy:
                check_point_path = self.parameters["data"]["energy_checkpoint"]

            elif task == Task.cameradirection or task == Task.skydirection:
                check_point_path = self.parameters["data"]["direction_checkpoint"]

            else:
                raise ValueError(
                    f"task:{task.name} is not supported. Task must be type, direction or energy"
                )
            # Load the checkpoint if provided
            if check_point_path:
                model_net = ModelHelper.loadModel(
                    model_net, "", check_point_path, Mode.train, device_str=self.device_str
                )
           
            # Setup the TensorBoard logger
            log_dir = save_folder
            
            tb_logger = TensorBoardLogger(
                save_dir=log_dir,
                name="exp_"
                + str(self.experiment_number)
                + "_"
                + task.name
                + "_train",
                default_hp_metric=False,
            )

            extra_trainer_args = {}
            test_limit = os.environ.get("CTLEARN_TEST_LIMIT")
            if test_limit:
                try:
                    limit = int(test_limit)
                except ValueError:
                    limit = 5
                extra_trainer_args["limit_train_batches"] = limit
                extra_trainer_args["limit_val_batches"] = limit

            # Setup the Trainer
            trainer_pl = CTLearnTrainer(
                max_epochs=self.parameters["hyp"]["epochs"],
                accelerator=self.parameters["arch"]["device"],
                devices=self.devices,
                strategy= self.parameters["arch"]["strategy"],
                default_root_dir=log_dir,
                log_every_n_steps=1,
                logger=tb_logger,
                num_sanity_val_steps=0,
                precision=precision,
                gradient_clip_val=self.parameters["hyp"]["gradient_clip_val"],
                callbacks=[GPUStatsLogger()],
                sync_batchnorm=True,
                enable_progress_bar=not self.disable_progress_bar,
                **extra_trainer_args
            )
 
            # Setup Lighting 
            lightning_model = CTLearnPL(
                model=model_net,
                save_folder=trainer_pl.get_log_dir(),
                task=task,
                mode = Mode.train,
                parameters=self.parameters,
                k=self.save_k,
                train_loader= self.training_loader,
                val_loader= self.validation_loader,
                train_dataset=self.train_dataset,
                val_dataset=self.validation_dataset
            )
            
            if trainer_pl.is_global_zero:
                # Save configuration file.
                if not os.path.exists(trainer_pl.get_log_dir()):
                    os.makedirs(trainer_pl.get_log_dir())
                
                with open(os.path.join(trainer_pl.get_log_dir(),"parameters.json"), "w") as f:
                    def path_serializer(obj):
                        import pathlib
                        if isinstance(obj, pathlib.Path):
                            return str(obj)
                        raise TypeError(f"Type {type(obj)} not serializable")
                    json.dump(self.parameters, f, indent=4, default=path_serializer)
        
                print(f"Run tensorboard server: tensorboard --load_fast=false --host=0.0.0.0 --logdir={trainer_pl.get_log_dir()}/")

                print(f"Accelerator: {trainer_pl.accelerator}")   
                print(f"Num. Devices: {trainer_pl.num_devices}")  
                 
            trainer_pl.fit(
                model=lightning_model,
                train_dataloaders=self.training_loader,
                val_dataloaders=[self.validation_loader],
            )    

            # Export to ONNX if requested
            if self.save_onnx:
                self.log.info("Converting PyTorch model into ONNX format...")
                try:
                    # Load the best model weights if available
                    if trainer_pl.checkpoint_callback and trainer_pl.checkpoint_callback.best_model_path:
                        best_model_path = trainer_pl.checkpoint_callback.best_model_path
                        self.log.info(f"Loading best model from {best_model_path} for ONNX export...")
                        model_net = ModelHelper.loadModel(
                            model_net, "", best_model_path, Mode.train, device_str=self.device_str
                        )
                    
                    # Create dummy input dynamically from a batch
                    batch = next(iter(self.training_loader))
                    features, labels, t = batch
                    
                    import inspect
                    sig = inspect.signature(model_net.forward)
                    num_inputs_sig = len(sig.parameters)
                    
                    # Transfer dummy inputs to same device as model
                    if num_inputs_sig == 2:
                        dummy_image = torch.randn_like(features["image"][:1]).to(self.device_str)
                        dummy_peak = torch.randn_like(features["peak_time"][:1]).to(self.device_str)
                        dummy_input = (dummy_image, dummy_peak)
                        input_names = ["image", "peak_time"]
                    else:
                        dummy_input = torch.randn_like(features["image"][:1]).to(self.device_str)
                        input_names = ["image"]
                        
                    output_names = [task.name]
                    
                    # Put model in evaluation mode
                    model_net.eval()
                    
                    onnx_path = os.path.join(save_folder, f"ctlearn_model_{task.name}")
                    ModelHelper.exportOnnx(
                        model_net,
                        dummy_input,
                        onnx_path,
                        input_names=input_names,
                        output_names=output_names
                    )
                    self.log.info(f"ONNX model saved successfully to {onnx_path}.onnx and {onnx_path}_simp.onnx")
                except Exception as e:
                    self.log.error(f"Failed to export model to ONNX: {e}")

    def finish(self):
        super().finish()
        print("Pytorch finish")

    def show_version(self):
        print("Pytorch 2.3")
