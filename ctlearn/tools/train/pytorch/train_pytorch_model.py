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

    pytorch_int = Int(
        allow_none=False,
        default_value=42
    ).tag(config=True)

    aliases = {
        **TrainCTLearnModel.aliases,  
        "pytorch_param": "TrainPyTorchModel.pytorch_int",
 
    }    

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("CONFIG VALUES PYTORCH:", self.config)    

    def setup(self):
        super().setup()
        print("Pytorch setup :)")
        print(f"DEBUG - framework_type raw: {self.reco_tasks} ({type(self.reco_tasks)})")
        

    def start(self):
        super().start()
        print("Pytorch start")       
        
    def finish(self):
        super().finish()
        print("Pytorch finish")      

    def show_version(self):
        print("Pytorch 2.3")