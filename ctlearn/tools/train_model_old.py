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
from ctlearn.core.ctlearn_enum import FrameworkType

class DLFrameWork(Tool):
    name = "dlframework"

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
                from ctlearn.tools.train.keras.train_keras_model import TrainKerasModel
                fw = TrainKerasModel()

            except ImportError as e:
                raise ImportError(f"Not possible to import TrainKerasModel: {e}") from e
            

        elif framework_type == FrameworkType.PYTORCH:
            try:
                from ctlearn.tools.train.pytorch.train_pytorch_model import TrainPyTorchModel
                fw = TrainPyTorchModel()
                print("Pytorch")
            except ImportError as e:
                raise ImportError(f"Not possible to import TrainPyTorchModel: {e}") from e
            

            
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

    # DLFrameWork.get_framework(fw_type)
    fw = DLFrameWork.get_framework(fw_type)
    # fw = DLFrameWork()
    # print(sys.argv[1:])
    # fw.parse_command_line(argv=sys.argv[1:])
    fw.run()
    # Launch the Framework
    # DLFrameWork().launch_instance()
# Example: 
# python -m ctlearn.tools.train_model --framework pytorch --output ./output_dir2 --signal ./mc_tjark/ --pattern-signal gamma_*.dl1.h5 --reco energy --overwrite