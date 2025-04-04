"""
Tools to train a ``CTLearnModel` (in Keras or PyTorch) on R1/DL1a data using the ``DLDataReader`` and ``DLDataLoader``.
"""

import sys
import argparse
from ctapipe.core import Tool
import warnings
from ctapipe.core.traits import (
    CaselessStrEnum,
)
from ctlearn import is_package_available
from ctlearn.tools.ctlearn_enum import FrameworkType

class DLFrameWork(Tool):
    name = "dlframework"
    framework_type = CaselessStrEnum(
        ["pytorch", "keras"],
        default_value="keras",
        help="Framework to use pytorch or keras",
    ).tag(config=True)

    aliases = {
        "framework": "DLFrameWork.framework_type",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def start(self):
        print(f"Selected Framework: {self.framework_type}")

        framework = self.string_to_type(self.framework_type)
        fw_obj = self.get_framework(framework)
        fw_obj.update_config(self.config)
        fw_obj.parse_command_line(argv=sys.argv[1:])
        fw_obj.run()

    @classmethod
    def string_to_type(self, str_type: str) -> FrameworkType:

        type_ = None
        str_type = str.upper(str_type)
        try:
            type_ = FrameworkType[str_type]
        except KeyError:
            print(f"'{str_type}' is not a valid enum type.")
        return type_

    @classmethod
    def get_framework(self, framework_type: FrameworkType):
        if framework_type == FrameworkType.KERAS:
            try:
                from ctlearn.tools.keras.train_keras_model import TrainKerasModel
            except ImportError:
                raise ImportError(f"Not possible to import TrainKerasModel")
            fw = TrainKerasModel()

        elif framework_type == FrameworkType.PYTORCH:
            try:
                from ctlearn.tools.pytorch.train_pytorch_model import (
                    TrainPyTorchModel,
                )
            except ImportError:
                raise ImportError(f"Not possible to import TrainPyTorchModel")

            fw = TrainPyTorchModel()
            
        else:
            raise ValueError(f"Unknown Framework: {framework_type.name}")
        # Update Aliases
        self.aliases.update(fw.aliases)
        DLFrameWork.aliases.update(fw.aliases)

        return fw


if __name__ == "__main__":

    # Parse the framework argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", default="keras")
    args, _ = parser.parse_known_args()

    # Get Framework type
    if args.framework:
        fw_type = DLFrameWork.string_to_type(args.framework)
    else:
        raise ValueError(
            f"Framework not defined, use : --framework keras or --framework pytorch"
        )

    DLFrameWork.get_framework(fw_type)

    # Launch the Framework
    DLFrameWork().launch_instance()
