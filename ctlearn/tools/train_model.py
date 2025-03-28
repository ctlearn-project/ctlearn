"""
Tools to train a ``CTLearnModel` (in Keras or PyTorch) on R1/DL1a data using the ``DLDataReader`` and ``DLDataLoader``.
"""

import atexit
import pandas as pd
import numpy as np
import shutil
import tensorflow as tf
import sys
from ctapipe.core import Tool
from ctapipe.core.tool import ToolConfigurationError
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
from dl1_data_handler.reader import DLDataReader
from ctlearn.core.loader import DLDataLoader
from ctlearn.core.model import CTLearnModel
from ctlearn.utils import validate_trait_dict
from ctlearn.tools.pytorch.train_pytorch_model import TrainPyTorchModel
from ctlearn.tools.keras.train_keras_model import TrainKerasModel
# from ctlearn.tools.base_train_model import TrainCTLearnModel
import importlib.util

from enum import Enum
class FrameworkType(Enum):
    KERAS = 1
    PYTORCH = 2


class DLFrameWork(Tool):
    name = "dlframework"
    framework_type = CaselessStrEnum(
        ["pytorch", "keras"],
        default_value="pytorch",
        help="Framework to use",
    ).tag(config=True)

    aliases = {
        **TrainPyTorchModel.aliases,
        **TrainKerasModel.aliases,
        "framework": "DLFrameWork.framework_type",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("CONFIG VALUES:", self.config)        

    # def setup(self):
    #     pass  # Do Nothing

    def start(self):

        print(f"Selected Framework: {self.framework_type}")

        framework= self.string_to_type(self.framework_type)
        fw_obj = self.get_framework(framework)
        fw_obj.parse_command_line(argv=sys.argv[1:])  # parse reco correctly now
        fw_obj.run()

    def finish(self):
        pass

    @classmethod
    def string_to_type(self,str_type:str)->FrameworkType:

        type_=None
        str_type = str.upper(str_type)
        try:
            type_ = FrameworkType[str_type]
        except KeyError:
            print(f"'{str_type}' is not a valid enum type.")
        return type_
    
    def get_framework(self, framework_type: FrameworkType):
        if framework_type == FrameworkType.KERAS:
            if not self.is_package_available("tensorflow"):
               raise ImportError("TensorFlow is not installed. Cannot run Keras framework.")
            else: 
                fw = TrainKerasModel()

        elif framework_type == FrameworkType.PYTORCH:
            if not self.is_package_available("torch"):
                raise ImportError("PyTorch is not installed. Cannot run PyTorch framework.")
            else:            
                fw = TrainPyTorchModel()
        else:
            raise ValueError("Unknown Framework")

        return fw
    
    def is_package_available(self, package_name: str) -> bool:
        return importlib.util.find_spec(package_name) is not None

if __name__ == "__main__":

    DLFrameWork().launch_instance()
 