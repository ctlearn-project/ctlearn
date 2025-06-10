from ctapipe.core.traits import (
    Bool,
    CaselessStrEnum,
    Path,
    Float,
    Int,
    List,
    Dict,
    classes_with_traits,
    ComponentName,
    Unicode,
)

from ctlearn.tools.train.pytorch.CTLearnPL import CTLearnTrainer, CTLearnPL
try:
    import torch
    
except ImportError:
    raise ImportError("pytorch is not installed in your environment!")

try:
    import pytorch_lightning
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
        allow_none=False,
        directory_ok=True,
        file_ok=True,
        help="Configuration file.",
    ).tag(config=True)

    aliases = {
        **TrainCTLearnModel.aliases,
        "config_file": "TrainPyTorchModel.config_file",
    }

    def __init__(self, **kwargs):
        
        # Setup GPU 
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["NCCL_DEBUG"] = "WARN"
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["NCCL_DEBUG"] = "INFO"
        torch.set_float32_matmul_precision('medium')        
 
        super().__init__(**kwargs)
 

    def setup(self):
 
        super().setup()

        # Create tasks Enum List
        self.tasks = str_list_to_enum_list(self.reco_tasks)

        for task_ in self.tasks:
            print("Task:", task_.name)

        print(self.config_file)
        self.parameters = read_configuration(self.config_file)
        sanity_check(self.parameters, expected_structure)

        self.experiment_number = self.parameters["run_details"]["experiment_number"]
        self.save_k = self.parameters["hyp"]["save_k"]
        self.device_str = self.parameters["arch"]["device"]
        self.device = torch.device(self.device_str)

        self.batch_size = self.parameters["hyp"]["batches"]
        self.pin_memory = self.parameters["dataset"]["pin_memory"]

        self.num_workers = self.parameters["dataset"]["num_workers"]
        self.persistent_workers = self.parameters["dataset"]["persistent_workers"]

        self.devices =  self.parameters["arch"]["devices"]
        self.save_k = self.parameters["hyp"]["save_k"]


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

        # --------------------------------------------------------------------
        # Reduce for testing 
        # --------------------------------------------------------------------
        # Limit the number of examples (optional)
        # max_training_samples = 500  # or whatever number you want
        # max_validation_samples = 200  # or whatever number you want

        # training_indices = training_indices[:max_training_samples]
        # validation_indices = validation_indices[:max_validation_samples]


        print("BASE TRAIN FRAMEWORK", self.framework_type)
        
        self.training_loader = DLDataLoader.create(
            framework=self.framework_type,
            DLDataReader=self.dl1dh_reader,
            indices=training_indices,
            tasks=self.reco_tasks,
            batch_size=self.batch_size,
            random_seed=self.random_seed,
            sort_by_intensity=self.sort_by_intensity,
            stack_telescope_images=self.stack_telescope_images,
            parameters=self.parameters,
            use_augmentation=True,
        )
        
        self.validation_loader = DLDataLoader.create(
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
        )


    def start(self):
        super().start()
        for task in self.tasks:

            # Create the experiment folder
            save_folder = create_experiment_folder(
                f"run_{task.name}_training_", next_number=self.experiment_number
            )

            # ------------------------------------------------------------------------------
            # Select the model and precision
            # ------------------------------------------------------------------------------

            if task == Task.type:
                precision = self.parameters["arch"]["precision_type"]
                model_net = create_model(self.parameters["model"]["model_type"])

            elif task == Task.energy:
                precision = self.parameters["arch"]["precision_energy"]
                model_net = create_model(self.parameters["model"]["model_energy"])
            
            elif task == Task.cameradirection or task == Task.skydirection:
                precision = self.parameters["arch"]["precision_direction"]
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

            elif task == Task.cameradirection or task == Task.skydirection:
                check_point_path = self.parameters["data"]["direction_checkpoint"]

            else:
                raise ValueError(
                    f"task:{task.name} is not supported. Task must be type, direction or energy"
                )
            # Load the checkpoint
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
            )
            
            if trainer_pl.is_global_zero:
                # Save configuration file.
                if not os.path.exists(trainer_pl.get_log_dir()):
                    os.makedirs(trainer_pl.get_log_dir())
                
                with open(os.path.join(trainer_pl.get_log_dir(),"parameters.json"), "w") as f:
                    json.dump(self.parameters, f, indent=4)
        
                print(f"Run tensorboard server: tensorboard --load_fast=false --host=0.0.0.0 --logdir={trainer_pl.get_log_dir()}/")

                print(f"Accelerator: {trainer_pl.accelerator}")   
                print(f"Num. Devices: {trainer_pl.num_devices}")  
                 
            trainer_pl.fit(
                model=lightning_model,
                train_dataloaders=self.training_loader,
                val_dataloaders=[self.validation_loader],
            )    

    def finish(self):
        super().finish()
        print("Pytorch finish")

    def show_version(self):
        print("Pytorch 2.3")
