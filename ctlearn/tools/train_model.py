import sys
from ctapipe.core import Tool
from ctapipe.core.traits import CaselessStrEnum
from ctlearn.core.ctlearn_enum import FrameworkType


class DLFrameWork(Tool):
    """
    Tool to select and run a specific deep learning training framework (Keras or PyTorch)
    for CTLearn model training. It dynamically loads the appropriate subclass based on
    the user-defined --framework argument.
    """
    name = "dlframework"

    framework_type = CaselessStrEnum(
        ["pytorch", "keras"],
        default_value="keras",
        help="Framework to use: pytorch or keras",
    ).tag(config=True)

    aliases = {
        "framework": "DLFrameWork.framework_type",
    }

    def __init__(self, **kwargs):
        """
        Initialize the DLFrameWork tool and prepare for framework injection.
        """
        super().__init__(**kwargs)
        self.framework_instance = None
        print("init")

    def setup(self):
        """
        Setup method called after basic trait parsing. 
        This dynamically loads and prepares the correct framework subclass 
        (TrainKerasModel or TrainPyTorchModel).
        """
        print("setup")
        framework_enum = self.string_to_type(self.framework_type)
        self.framework_instance = self.get_framework(framework_enum)

        # Inject aliases and shared config before full CLI parsing
        self.framework_instance.update_config(self.config)
        self.aliases.update(self.framework_instance.aliases)
        DLFrameWork.aliases.update(self.framework_instance.aliases)

    def start(self):
        """
        Start method called after setup. Executes the selected framework instance.
        """
        print("start")
        self.framework_instance.run()

    @classmethod
    def string_to_type(cls, str_type: str) -> FrameworkType:
        """
        Convert a string to a FrameworkType enum (case-insensitive).

        Parameters:
            str_type (str): The name of the framework (e.g., 'keras', 'pytorch').

        Returns:
            FrameworkType: Corresponding enum value.

        Raises:
            ValueError: If the provided string is not a valid framework type.
        """
        try:
            return FrameworkType[str_type.upper()]
        except KeyError:
            raise ValueError(f"'{str_type}' is not a valid framework type.")

    @classmethod
    def get_framework(cls, framework_type: FrameworkType):
        """
        Dynamically import and return the corresponding training class 
        based on the framework type.

        Parameters:
            framework_type (FrameworkType): Enum indicating which framework to use.

        Returns:
            Tool: An instance of the selected training framework (subclass of Tool).

        Raises:
            ImportError: If the training module could not be imported.
            ValueError: If the framework type is unknown.
        """
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

        return fw


if __name__ == "__main__":
    # Parse only --framework to determine which subclass to load
    minimal_args = [arg for arg in sys.argv[1:] if "--framework" in arg or arg in ["-h", "--help"]]
    tool = DLFrameWork()
    tool.initialize(argv=minimal_args)

    # Setup and inject the correct framework instance
    tool.setup()

    # Parse all CLI args with the selected framework subclass
    tool.framework_instance.initialize(argv=sys.argv[1:])
    tool.run()

# Example:
# python -m ctlearn.tools.train_model --framework=pytorch --output ./output_dir2 --signal ./mc_tjark/ --pattern-signal gamma_*.dl1.h5 --reco energy --overwrite