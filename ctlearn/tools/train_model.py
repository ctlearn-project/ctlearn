"""
Tool to train a ``CTLearnModel`` on R1/DL1a data using the ``DLDataReader`` and ``DLDataLoader``.
"""

import atexit
import keras
import pandas as pd
import numpy as np
import shutil
import tensorflow as tf

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


class TrainCTLearnModel(Tool):
    """
    Tool to train a ``~ctlearn.core.model.CTLearnModel`` on R1/DL1a data.

    The tool trains a CTLearn model on the input data (R1 calibrated waveforms or DL1a images) and
    saves the trained model in the output directory. The input data is loaded from the input directories
    for signal and background events using the ``~dl1_data_handler.reader.DLDataReader`` and
    ``~dl1_data_handler.loader.DLDataLoader``. The tool supports the following reconstruction tasks:
    - Classification of the primary particle type (gamma/proton)
    - Regression of the primary particle energy
    - Regression of the primary particle arrival direction based on the offsets in camera coordinates
    - Regression of the primary particle arrival direction based on the offsets in sky coordinates
    """

    name = "ctlearn-train-model"
    description = __doc__

    examples = """
    To train a CTLearn model for the classification of the primary particle type:
    > ctlearn-train-model \\
        --signal /path/to/your/gammas_dl1_dir/ \\
        --pattern-signal "gamma_*_run1.dl1.h5" \\
        --pattern-signal "gamma_*_run10.dl1.h5" \\
        --background /path/to/your/protons_dl1_dir/ \\
        --pattern-background "proton_*_run1.dl1.h5" \\
        --pattern-background "proton_*_run10.dl1.h5" \\
        --output /path/to/your/type/ \\
        --reco type \\

    To train a CTLearn model for the regression of the primary particle energy:
    > ctlearn-train-model \\
        --signal /path/to/your/gammas_dl1_dir/ \\
        --pattern-signal "gamma_*_run1.dl1.h5" \\
        --pattern-signal "gamma_*_run10.dl1.h5" \\
        --output /path/to/your/energy/ \\
        --reco energy \\

    To train a CTLearn model for the regression of the primary particle
    arrival direction based on the offsets in camera coordinates:
    > ctlearn-train-model \\
        --signal /path/to/your/gammas_dl1_dir/ \\
        --pattern-signal "gamma_*_run1.dl1.h5" \\
        --pattern-signal "gamma_*_run10.dl1.h5" \\
        --output /path/to/your/direction/ \\
        --reco cameradirection \\

    To train a CTLearn model for the regression of the primary particle
    arrival direction based on the offsets in sky coordinates:
    > ctlearn-train-model \\
        --signal /path/to/your/gammas_dl1_dir/ \\
        --pattern-signal "gamma_*_run1.dl1.h5" \\
        --pattern-signal "gamma_*_run10.dl1.h5" \\
        --output /path/to/your/direction/ \\
        --reco skydirection \\
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

    model_type = ComponentName(
        CTLearnModel, default_value="ResNet"
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

    save_best_validation_only = Bool(
        default_value=True,
        allow_none=False,
        help="Set whether to save the best validation checkpoint only.",
    ).tag(config=True)

    optimizer = Dict(
        default_value={"name": "Adam", "base_learning_rate": 0.0001, "adam_epsilon": 1.0e-8},
	help=(
	    "Optimizer to use for training. "
	    "E.g. {'name': 'Adam', 'base_learning_rate': 0.0001, 'adam_epsilon': 1.0e-8}. "
	)
    ).tag(config=True)

    lr_reducing = Dict(
        default_value={"factor": 0.5, "patience": 5, "min_delta": 0.01, "min_lr": 0.000001},
        allow_none=True,
	help=(
	    "Learning rate reducing parameters for the Keras callback. "
	    "E.g. {'factor': 0.5, 'patience': 5, 'min_delta': 0.01, 'min_lr': 0.000001}. "
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
    
    early_stopping = Dict(
        default_value=None,
        allow_none=True,
	help=(
	    "Early stopping parameters for the Keras callback. "
	    "E.g. {'monitor': 'val_loss', 'patience': 4, 'verbose': 1, 'restore_best_weights': True}. "
	)
    ).tag(config=True)


    overwrite = Bool(help="Overwrite output dir if it exists").tag(config=True)

    aliases = {
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

    def setup(self):
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
        # Create a MirroredStrategy.
        self.strategy = tf.distribute.MirroredStrategy()
        atexit.register(self.strategy._extended._collective_ops._lock.locked)  # type: ignore
        self.log.info("Number of devices: %s", self.strategy.num_replicas_in_sync)
        # Get signal input files
        self.input_url_signal = []
        for signal_pattern in self.file_pattern_signal:
            self.input_url_signal.extend(self.input_dir_signal.glob(signal_pattern))
        # Get bkg input files
        self.input_url_background = []
        if self.input_dir_background is not None:
            for background_pattern in self.file_pattern_background:
                self.input_url_background.extend(self.input_dir_background.glob(background_pattern))

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
        self.training_loader = DLDataLoader(
            self.dl1dh_reader,
            training_indices,
            tasks=self.reco_tasks,
            batch_size=self.batch_size*self.strategy.num_replicas_in_sync,
            random_seed=self.random_seed,
            sort_by_intensity=self.sort_by_intensity,
            stack_telescope_images=self.stack_telescope_images,
        )
        self.validation_loader = DLDataLoader(
            self.dl1dh_reader,
            validation_indices,
            tasks=self.reco_tasks,
            batch_size=self.batch_size*self.strategy.num_replicas_in_sync,
            random_seed=self.random_seed,
            sort_by_intensity=self.sort_by_intensity,
            stack_telescope_images=self.stack_telescope_images,
        )

        # Set up the callbacks
        monitor = "val_loss"
        monitor_mode = "min"
        # Model checkpoint callback
        # Temp fix for supporting keras2 & keras3
        if int(keras.__version__.split(".")[0]) >= 3:
            model_path = f"{self.output_dir}/ctlearn_model.keras"
        else:
            model_path = f"{self.output_dir}/ctlearn_model.cpk"
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor=monitor,
            verbose=1,
            mode=monitor_mode,
            save_best_only=self.save_best_validation_only,
        )
        # Tensorboard callback
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=self.output_dir, histogram_freq=1
        )
        # CSV logger callback
        csv_logger_callback = keras.callbacks.CSVLogger(
            filename=f"{self.output_dir}/training_log.csv", append=True
        )
        self.callbacks = [model_checkpoint_callback, tensorboard_callback, csv_logger_callback]
	
        if self.early_stopping is not None:
            # EarlyStopping callback
            validate_trait_dict(self.early_stopping, ["monitor", "patience", "verbose", "restore_best_weights"])
            early_stopping_callback = keras.callbacks.EarlyStopping(
            	monitor=self.early_stopping["monitor"], 
		patience=self.early_stopping["patience"], 
		verbose=self.early_stopping["verbose"],
		restore_best_weights=self.early_stopping["restore_best_weights"]
            )
            self.callbacks.append(early_stopping_callback)

        # Learning rate reducing callback
        if self.lr_reducing is not None:
            # Validate the learning rate reducing parameters
            validate_trait_dict(self.lr_reducing, ["factor", "patience", "min_delta", "min_lr"])
            lr_reducing_callback = keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=self.lr_reducing["factor"],
                patience=self.lr_reducing["patience"],
                mode=monitor_mode,
                verbose=1,
                min_delta=self.lr_reducing["min_delta"],
                min_lr=self.lr_reducing["min_lr"],
            )
            self.callbacks.append(lr_reducing_callback)


	
    def start(self):
        
        # Open a strategy scope.
        with self.strategy.scope():
            # Construct the model
            self.log.info("Setting up the model.")
            self.model = CTLearnModel.from_name(
                self.model_type,
                input_shape=self.training_loader.input_shape,
                tasks=self.reco_tasks,
                parent=self,
            ).model
            # Validate the optimizer parameters
            validate_trait_dict(self.optimizer, ["name", "base_learning_rate"])
            # Set the learning rate for the optimizer
            learning_rate =  self.optimizer["base_learning_rate"]
            # Set the epsilon for the Adam optimizer
            adam_epsilon = None
            if self.optimizer["name"] == "Adam":
                # Validate the epsilon for the Adam optimizer
                validate_trait_dict(self.optimizer, ["adam_epsilon"])
                # Set the epsilon for the Adam optimizer
                adam_epsilon = self.optimizer["adam_epsilon"]
            # Select optimizer with appropriate arguments
            # Dict of optimizer_name: (optimizer_fn, optimizer_args)
            optimizers = {
                "Adadelta": (
                    keras.optimizers.Adadelta,
                    dict(learning_rate=learning_rate),
                ),
                "Adam": (
                    keras.optimizers.Adam,
                    dict(learning_rate=learning_rate, epsilon=adam_epsilon),
                ),
                "RMSProp": (keras.optimizers.RMSprop, dict(learning_rate=learning_rate)),
                "SGD": (keras.optimizers.SGD, dict(learning_rate=learning_rate)),
            }
            # Get the optimizer function and arguments
            optimizer_fn, optimizer_args = optimizers[self.optimizer["name"]]
            # Get the losses and metrics for the model
            losses, metrics = self._get_losses_and_mertics(self.reco_tasks)
            # Compile the model
            self.log.info("Compiling CTLearn model.")
            self.model.compile(optimizer=optimizer_fn(**optimizer_args), loss=losses, metrics=metrics)

        # Train and evaluate the model
        self.log.info("Training and evaluating...")
        self.model.fit(
            self.training_loader,
            validation_data=self.validation_loader,
            epochs=self.n_epochs,
            class_weight=self.dl1dh_reader.class_weight,
            callbacks=self.callbacks,
            verbose=2,
        )
        self.log.info("Training and evaluating finished succesfully!")


    def finish(self):

        # Saving model weights in onnx format
        if self.save_onnx:
            self.log.info("Converting Keras model into ONNX format...")
            self.log.info("Make sure tf2onnx is installed in your enviroment!")
            try:
                import tf2onnx
            except ImportError:
                raise ImportError("tf2onnx is not installed in your environment!")

            output_path = f"{self.output_dir}/ctlearn_model.onnx"
            tf2onnx.convert.from_keras(
                self.model, input_signature=self.model.input_layer.input._type_spec, output_path=output_path
            )
            self.log.info("ONNX model saved in %s", self.output_dir)

        self.log.info("Tool is shutting down")

    def _get_losses_and_mertics(self, tasks):
        """
        Build the fully connected head for the CTLearn model.

        Function to build the fully connected head of the CTLearn model using the specified parameters.

        Parameters
        ----------
        inputs : keras.layers.Layer
            Keras layer of the model.
        layers : dict
            Dictionary containing the number of neurons (as value) in the fully connected head for each task (as key).
        tasks : list
            List of tasks to build the head for.

        Returns
        -------
        logits : dict
            Dictionary containing the logits for each task.
        """
        losses, metrics = {}, {}
        if "type" in self.reco_tasks:
            losses["type"] = keras.losses.CategoricalCrossentropy(
                reduction="sum_over_batch_size"
            )
            metrics["type"] = [
                keras.metrics.CategoricalAccuracy(name="accuracy"),
                keras.metrics.AUC(name="auc"),
            ]
            # Temp fix till keras support class weights for multiple outputs or I wrote custom loss
            # https://github.com/keras-team/keras/issues/11735
            if len(tasks) == 1:
                losses = losses["type"]
                metrics = metrics["type"]
        if "energy" in self.reco_tasks:
            losses["energy"] = keras.losses.MeanAbsoluteError(
                reduction="sum_over_batch_size"
            )
            metrics["energy"] = keras.metrics.MeanAbsoluteError(name="mae_energy")
        if "cameradirection" in self.reco_tasks:
            losses["cameradirection"] = keras.losses.MeanAbsoluteError(
                reduction="sum_over_batch_size"
            )
            metrics["cameradirection"] = keras.metrics.MeanAbsoluteError(name="mae_cameradirection")
        if "skydirection" in self.reco_tasks:
            losses["skydirection"] = keras.losses.MeanAbsoluteError(
                reduction="sum_over_batch_size"
            )
            metrics["skydirection"] = keras.metrics.MeanAbsoluteError(name="mae_skydirection")
        return losses, metrics


def main():
    # Run the tool
    tool = TrainCTLearnModel()
    tool.run()


if __name__ == "main":
    main()
