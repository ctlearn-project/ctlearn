"""
Base tool to train a ``CTLearnModel``on R1/DL1a data using the ``DLDataReader`` and ``DLDataLoader``.
"""

from abc import abstractmethod
import numpy as np


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
from ctlearn import __version__ as ctlearn_version
from ctlearn.core.model import CTLearnModel
from dl1_data_handler.reader import DLDataReader


class TrainCTLearnModel(Tool):
    """
    Base tool to train a ``~ctlearn.core.model.CTLearnModel`` on R1/DL1a data.

    The tool holds configurations and set up functions.
   
    """

    input_dir_signal = Path(
        help="Input directory for signal events",
        allow_none=False,
        exists=True,
        directory_ok=True,
        file_ok=False,
    ).tag(config=True)

    file_pattern_signal = List(
        trait=Unicode(),
        default_value=["*.h5"],
        help="List of specific file pattern for matching files in ``input_dir_signal``",
    ).tag(config=True)

    input_dir_background = Path(
        default_value=None,
        help="Input directory for background events",
        allow_none=True,
        exists=True,
        directory_ok=True,
        file_ok=False,
    ).tag(config=True)

    file_pattern_background = List(
        trait=Unicode(),
        default_value=["*.h5"],
        help="List of specific file pattern for matching files in ``input_dir_background``",
    ).tag(config=True)

    dl1dh_reader_type = ComponentName(DLDataReader, default_value="DLImageReader").tag(
        config=True
    )

    stack_telescope_images = Bool(
        default_value=False,
        allow_none=False,
        help=(
            "Set whether to stack the telescope images in the data loader. "
            "Requires DLDataReader mode to be ``stereo``."
        ),
    ).tag(config=True)

    sort_by_intensity = Bool(
        default_value=False,
        allow_none=True,
        help=(
            "Set whether to sort the telescope images by intensity in the data loader. "
            "Requires DLDataReader mode to be ``stereo``."
        ),
    ).tag(config=True)

    #model_type = ComponentName(CTLearnModel, default_value="KerasResNet").tag(config=True)

    output_dir = Path(
        exits=False,
        default_value=None,
        allow_none=False,
        directory_ok=True,
        file_ok=False,
        help="Output directory for the trained reconstructor.",
    ).tag(config=True)

    reco_tasks = List(
        trait=CaselessStrEnum(["type", "energy", "cameradirection", "skydirection"]),
        allow_none=False,
        help=(
            "List of reconstruction tasks to perform. "
            "'type': classification of the primary particle type; "
            "'energy': regression of the primary particle energy; "
            "'cameradirection': reconstruction of the primary particle arrival direction in camera coordinates; "
            "'skydirection': reconstruction of the primary particle arrival direction in sky coordinates."
        ),
    ).tag(config=True)

    n_epochs = Int(
        default_value=10,
        allow_none=False,
        help="Number of epochs to train the neural network.",
    ).tag(config=True)

    batch_size = Int(
        default_value=64,
        allow_none=False,
        help="Size of the batch to train the neural network.",
    ).tag(config=True)

    validation_split = Float(
        default_value=0.1,
        help="Fraction of the data to use for validation",
        min=0.01,
        max=0.99,
    ).tag(config=True)

    optimizer = Dict(
        default_value={
            "name": "Adam",
            "base_learning_rate": 0.0001,
            "adam_epsilon": 1.0e-8,
        },
        help=(
            "Optimizer to use for training. "
            "E.g. {'name': 'Adam', 'base_learning_rate': 0.0001, 'adam_epsilon': 1.0e-8}. "
        ),
    ).tag(config=True)

    random_seed = Int(
        default_value=0,
        help=(
            "Random seed for shuffling the data "
            "before the training/validation split "
            "and after the end of an epoch."
        ),
    ).tag(config=True)

    aliases = {
        "signal": "TrainCTLearnModel.input_dir_signal",
        "background": "TrainCTLearnModel.input_dir_background",
        "pattern-signal": "TrainCTLearnModel.file_pattern_signal",
        "pattern-background": "TrainCTLearnModel.file_pattern_background",
        "reco": "TrainCTLearnModel.reco_tasks",
        ("o", "output"): "TrainCTLearnModel.output_dir",
    }

    classes = classes_with_traits(CTLearnModel) + classes_with_traits(DLDataReader)

    def setup(self):
        self.log.info("ctlearn version %s", ctlearn_version)
        # Check if the output directory exists
        if self.output_dir.exists():
            raise ToolConfigurationError(
                f"Output directory {self.output_dir} already exists."
            )
        # Get signal input files
        self.input_url_signal = []
        for signal_pattern in self.file_pattern_signal:
            self.input_url_signal.extend(self.input_dir_signal.glob(signal_pattern))
        # Get bkg input files
        self.input_url_background = []
        if self.input_dir_background is not None:
            for background_pattern in self.file_pattern_background:
                self.input_url_background.extend(
                    self.input_dir_background.glob(background_pattern)
                )

        # Set up the data reader
        self.log.info("Loading data:")
        self.log.info("For a large dataset, this may take a while...")
        if self.dl1dh_reader_type == "DLFeatureVectorReader":
            raise NotImplementedError(
                "'DLFeatureVectorReader' is not supported in CTLearn yet. "
                "Missing stereo CTLearnModel implementation."
            )
        self.dl1dh_reader = DLDataReader.from_name(
            self.dl1dh_reader_type,
            input_url_signal=sorted(self.input_url_signal),
            input_url_background=sorted(self.input_url_background),
            parent=self,
        )
        self.log.info("Number of events loaded: %s", self.dl1dh_reader._get_n_events())
        if "type" in self.reco_tasks:
            self.log.info(
                "Number of signal events: %d", self.dl1dh_reader.n_signal_events
            )
            self.log.info(
                "Number of background events: %d", self.dl1dh_reader.n_bkg_events
            )
        # Check if the number of events is enough to form a batch
        if self.dl1dh_reader._get_n_events() < self.batch_size:
            raise ValueError(
                f"{self.dl1dh_reader._get_n_events()} events are not enough "
                f"to form a batch of size {self.batch_size}. Reduce the batch size."
            )
        # Check if there are at least two classes in the reader for the particle classification
        if self.dl1dh_reader.class_weight is None and "type" in self.reco_tasks:
            raise ValueError(
                "Classification task selected but less than two classes are present in the data."
            )
        # Check if stereo mode is selected for stacking telescope images
        if self.stack_telescope_images and self.dl1dh_reader.mode == "mono":
            raise ToolConfigurationError(
                f"Cannot stack telescope images in mono mode. Use stereo mode for stacking."
            )
        # Ckeck if only one telescope type is selected for stacking telescope images
        if (
            self.stack_telescope_images
            and len(list(self.dl1dh_reader.selected_telescopes)) > 1
        ):
            raise ToolConfigurationError(
                f"Cannot stack telescope images from multiple telescope types. Use only one telescope type."
            )
        # Check if sorting by intensity is disabled for stacking telescope images
        if self.stack_telescope_images and self.sort_by_intensity:
            raise ToolConfigurationError(
                f"Cannot stack telescope images when sorting by intensity. Disable sorting by intensity."
            )

        # Set up the data loaders for training and validation
        self.indices = list(range(self.dl1dh_reader._get_n_events()))
        # Shuffle the indices before the training/validation split
        np.random.seed(self.random_seed)
        np.random.shuffle(self.indices)
        self.n_validation_examples = int(
            self.validation_split * self.dl1dh_reader._get_n_events()
        )
        self.training_indices = self.indices[self.n_validation_examples:]
        self.validation_indices = self.indices[:self.n_validation_examples]

        # Set up framework-specific training tool
        self.setup_framework()
     
    @abstractmethod
    def setup_framework():
        """ This is an abstract method for the setup of the framework-specific training tool."""
        pass

    def finish(self):
        self.log.info("Tool is shutting down")
