

import numpy as np
import shutil
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
from ctlearn.core.data_loader.loader import DLDataLoader
from ctlearn.core.model import CTLearnModel


class TrainCTLearnModel(Tool):
    """
    Base class for training a ``CTLearnModel`` on R1/DL1a data using the ``DLDataReader`` and ``DLDataLoader``.

    The tool trains a CTLearn model on the input data (R1 calibrated waveforms or DL1a images) and
    saves the trained model in the output directory. The input data is loaded from the input directories
    for signal and background events using the ``DLDataReader`` and ``DLDataLoader``. The ``start`` method
    is implemented in the subclasses to train the model using the specified framework (Keras or PyTorch).
    The tool supports the following reconstruction tasks:
    - Classification of the primary particle type (gamma/proton)
    - Regression of the primary particle energy
    - Regression of the primary particle arrival direction based on the offsets in camera coordinates
    - Regression of the primary particle arrival direction based on the offsets in sky coordinates

    Attributes
    ----------
    input_dir_signal : Path
        Input directory for signal events.
    file_pattern_signal : List[Unicode]
        List of specific file pattern for matching files in ``input_dir_signal``.
    input_dir_background : Path
        Input directory for background events.
    file_pattern_background : List[Unicode]
        List of specific file pattern for matching files in ``input_dir_background``.
    dl1dh_reader_type : ComponentName
        Type of the DLDataReader to use for reading the input data.
    dl1dh_reader : DLDataReader
        DLDataReader instance for reading the input data.
    stack_telescope_images : Bool
        Set whether to stack the telescope images in the data loader.
    sort_by_intensity : Bool
        Set whether to sort the telescope images by intensity in the data loader.
    output_dir : Path
        Output directory for the trained reconstructor.
    reco_tasks : List[Unicode]
        List of reconstruction tasks to perform.
    n_epochs : Int
        Number of epochs to train the neural network.
    batch_size : Int
        Size of the batch to train the neural network.
    validation_split : Float
        Fraction of the data to use for validation.
    optimizer : Dict
        Optimizer to use for training.
    random_seed : Int
        Random seed for shuffling the data before the training/validation split and after the end of an epoch.
    save_onnx : Bool
        Set whether to save model in an ONNX file.
    overwrite : Bool
        Overwrite output dir if it exists.
    
    Methods
    -------
    setup()
        Set up the data reader and data loaders for training and validation.
    finish()
        Save the trained model in the output directory in ONNX if selected.
    """
    name = "ctlearn-train-model-base"

    framework_type = CaselessStrEnum(
        ["pytorch", "keras"],
        default_value="keras",
        help="Framework to use pytorch or keras"
    ).tag(config=True)

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

    dl1dh_reader_type = ComponentName(
        DLDataReader, default_value="DLImageReader"
    ).tag(config=True)

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
        default_value=None,
        help=(
            "List of reconstruction tasks to perform. "
            "'type': classification of the primary particle type "
            "'energy': regression of the primary particle energy "
            "'cameradirection': regression of the primary particle arrival direction in camera coordinates "
            "'skydirection': regression of the primary particle arrival direction in sky coordinates"
        )
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
        default_value={"name": "Adam", "base_learning_rate": 0.0001, "adam_epsilon": 1.0e-8},
	help=(
	    "Optimizer to use for training. "
	    "E.g. {'name': 'Adam', 'base_learning_rate': 0.0001, 'adam_epsilon': 1.0e-8}. "
	)
    ).tag(config=True)

    random_seed = Int(
        default_value=0,
        help=(
            "Random seed for shuffling the data "
            "before the training/validation split "
            "and after the end of an epoch."
        )
    ).tag(config=True)

    save_onnx = Bool(
        default_value=False,
        allow_none=False,
        help="Set whether to save model in an ONNX file.",
    ).tag(config=True)

    overwrite = Bool(help="Overwrite output dir if it exists").tag(config=True)

    aliases = {
        # "framework": "DLFrameWork.framework_type",
        "framework": "TrainCTLearnModel.framework_type",
        "signal": "TrainCTLearnModel.input_dir_signal",
        "background": "TrainCTLearnModel.input_dir_background",
        "pattern-signal": "TrainCTLearnModel.file_pattern_signal",
        "pattern-background": "TrainCTLearnModel.file_pattern_background",
        "reco": "TrainCTLearnModel.reco_tasks",
        ("o", "output"): "TrainCTLearnModel.output_dir",
    }

    flags = {
        "overwrite": (
            {"TrainCTLearnModel": {"overwrite": True}},
            "Overwrite existing files",
        ),
    }
    
    classes = (
        [
            CTLearnModel,
            DLDataReader,
        ]
        + classes_with_traits(CTLearnModel)
        + classes_with_traits(DLDataReader)
    )
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Common Init")

    def setup(self):
        print("Enter setup")
        # Check if the output directory exists and if it should be overwritten
        if self.output_dir.exists():
            if not self.overwrite:
                raise ToolConfigurationError(
                    f"Output directory {self.output_dir} already exists. Use --overwrite to overwrite."
                )
            else:
                # Remove the output directory if it exists
                self.log.info("Removing existing output directory %s", self.output_dir)
                shutil.rmtree(self.output_dir)

        # Must be moved to KERAS (It is moved already )
        # Create a MirroredStrategy.
        # self.strategy = tf.distribute.MirroredStrategy()
        # atexit.register(self.strategy._extended._collective_ops._lock.locked)  # type: ignore
        # self.log.info("Number of devices: %s", self.strategy.num_replicas_in_sync)
        
        # print(self.DLFrameWork.framework_type)
        # Get signal input files
        self.input_url_signal = []
        for signal_pattern in self.file_pattern_signal:
            self.input_url_signal.extend(self.input_dir_signal.glob(signal_pattern))
        
        # Get bkg input files
        self.input_url_background = []
        if self.input_dir_background is not None:
            for background_pattern in self.file_pattern_background:
                self.input_url_background.extend(self.input_dir_background.glob(background_pattern))

       
        print("DEBUG 1")
        # Set up the data reader
        self.log.info("Loading data:")
        self.log.info("For a large dataset, this may take a while...")
        if self.dl1dh_reader_type == "DLFeatureVectorReader":
            raise NotImplementedError(
                "'DLFeatureVectorReader' is not supported in CTLearn yet. "
                "Missing stereo CTLearnModel implementation."
            )
        print("DEBUG 2")
        print(f"self.dl1dh_reader_type: {self.dl1dh_reader_type}")
        self.dl1dh_reader = DLDataReader.from_name(
            self.dl1dh_reader_type,
            input_url_signal=sorted(self.input_url_signal),
            input_url_background=sorted(self.input_url_background),
            parent=self,
        )
        print("DEBUG 3")
        self.log.info("Number of events loaded: %s", self.dl1dh_reader._get_n_events())
        if "type" in self.reco_tasks:
            self.log.info("Number of signal events: %d", self.dl1dh_reader.n_signal_events)
            self.log.info("Number of background events: %d", self.dl1dh_reader.n_bkg_events)
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
        if self.stack_telescope_images and len(list(self.dl1dh_reader.selected_telescopes)) > 1:
            raise ToolConfigurationError(
                f"Cannot stack telescope images from multiple telescope types. Use only one telescope type."
            )
        # Check if sorting by intensity is disabled for stacking telescope images
        if self.stack_telescope_images and self.sort_by_intensity:
            raise ToolConfigurationError(
                f"Cannot stack telescope images when sorting by intensity. Disable sorting by intensity."
            )

        # Set up the data loaders for training and validation
        indices = list(range(self.dl1dh_reader._get_n_events()))
        # Shuffle the indices before the training/validation split
        np.random.seed(self.random_seed)
        np.random.shuffle(indices)
        n_validation_examples = int(self.validation_split * self.dl1dh_reader._get_n_events())
        training_indices = indices[n_validation_examples:]
        validation_indices = indices[:n_validation_examples]

        # Set self.strategy.num_replicas_in_sync to 1 in case that does not exist (Pytorch)
        if not hasattr(self, "strategy"):
            self.strategy = type("FakeStrategy", (), {"num_replicas_in_sync": 1})()
            print("num_replicas_in_sync:",self.strategy.num_replicas_in_sync)

        print(self.framework_type)

        self.training_loader = DLDataLoader.create(
            framework=self.framework_type,
            DLDataReader=self.dl1dh_reader,
            indices=training_indices,
            tasks=self.reco_tasks,
            batch_size=self.batch_size * self.strategy.num_replicas_in_sync,
            random_seed=self.random_seed,
            sort_by_intensity=self.sort_by_intensity,
            stack_telescope_images=self.stack_telescope_images,
        )

        self.validation_loader = DLDataLoader.create(
            framework=self.framework_type,
            DLDataReader=self.dl1dh_reader,
            indices=training_indices,
            tasks=self.reco_tasks,
            batch_size=self.batch_size * self.strategy.num_replicas_in_sync,
            random_seed=self.random_seed,
            sort_by_intensity=self.sort_by_intensity,
            stack_telescope_images=self.stack_telescope_images,
        )

    def finish(self):
        print("finish")
