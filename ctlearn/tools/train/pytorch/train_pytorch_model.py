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

try:
    import torch
except ImportError:
    raise ImportError("pytorch is not installed in your environment!")

try:
    import pytorch_lightning
except ImportError:
    raise ImportError("pytorch_lightning is not installed in your environment!")

from ctlearn.tools.train.base_train_model import TrainCTLearnModel
from ctlearn.core.ctlearn_enum import Task, Mode
from .utils import str_list_to_enum_list, sanity_check, read_configuration, create_experiment_folder, expected_structure

from ctlearn.core.pytorch.net_utils import create_model, ModelHelper

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
        "config_file":  "TrainPyTorchModel.config_file",
    }    

    def __init__(self, **kwargs):
        print("Pytorch init")      
        super().__init__(**kwargs)
        print("CONFIG VALUES PYTORCH:", self.config)    


    def setup(self):
        print("Pytorch setup")    
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

    def start(self):
        print("Pytorch start")      
        super().start()
        print("Pytorch start")       

        for task in self.tasks:

            # Create the experiment folder
            save_folder = create_experiment_folder(f"run_{task.name}_training_", next_number=self.experiment_number)

            # ------------------------------------------------------------------------------
            # Select the model and precision
            # ------------------------------------------------------------------------------
            if task == Task.direction:  
                precision = self.parameters["arch"]["precision_direction"]
                model_net = create_model(self.parameters["model"]["model_direction"])

            elif task == Task.type:  
                precision = self.parameters["arch"]["precision_type"]
                model_net = create_model(self.parameters["model"]["model_type"])

            elif task == Task.energy:  
                precision = self.parameters["arch"]["precision_energy"]
                model_net = create_model(self.parameters["model"]["model_energy"])

            else:
                raise ValueError(
                    f"task:{task.name} is not supported. Task must be type, direction or energy"
                )
            
            # ------------------------------------------------------------------------------
            # Load Checkpoints
            # ------------------------------------------------------------------------------
            if task == Task.type:   
                check_point_path = self.parameters["data"]["type_checkpoint"]

            if task == Task.energy:   
                check_point_path = self.parameters["data"]["energy_checkpoint"]

            if task == Task.direction:   
                check_point_path = self.parameters["data"]["direction_checkpoint"]


            # Load the checkpoint
            model_net = ModelHelper.loadModel(
                model_net, "", check_point_path, Mode.train, device_str=self.device_str
            )
            
    def finish(self):
        super().finish()
        print("Pytorch finish")      

    def show_version(self):
        print("Pytorch 2.3")