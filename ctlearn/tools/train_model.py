"""
Perform camera calibration from pedestal and flatfield files
"""

import atexit
import pathlib
from argparse import ArgumentParser

import numpy as np
from astropy.table import Table
import keras
import tensorflow as tf
import matplotlib.pyplot as plt

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

import pandas as pd
import os

from dl1_data_handler.reader import DLDataReader
from dl1_data_handler.loader import DLDataLoader
from ctlearn.core.model import CTLearnModel


class TrainCTLearnModel(Tool):
    """
    Tool to train a `~ctlearn.core.model.CTLearnModel` on R1/DL1a data.

    The tool first performs a cross validation to give an initial estimate
    on the quality of the estimation and then finally trains one model
    per telescope type on the full dataset.
    """

    name = "ctlearn-train-model"
    description = __doc__

    examples = """
    To train a CTLearn model for the classification of the primary particle type:
    > ctlearn-train-model \\
        --signal_input_dir gammas.dl1.h5 \\
        --bkg_input_dir protons.dl1.h5 \\
        --reco type \\

    To train a CTLearn model for the regression of the primary particle energy:
    > ctlearn-train-model \\
        --signal_input_dir gammas.dl1.h5 \\
        --reco energy \\
    """

    input_dir_signal = Path(
        help="Input directory for signal events",
        allow_none=False,
        exists=True,
        directory_ok=True,
        file_ok=False,
    ).tag(config=True)

    file_pattern_signal = Unicode(
        default_value="*.h5",
        help="Give a specific file pattern for matching files in ``input_dir_signal``",
    ).tag(config=True)

    input_dir_background = Path(
        default_value=None,
        help="Input directory for background events",
        allow_none=True,
        exists=True,
        directory_ok=True,
        file_ok=False,
    ).tag(config=True)

    file_pattern_background = Unicode(
        default_value="*.h5",
        help="Give a specific file pattern for matching files in ``input_dir_background``",
    ).tag(config=True)

    dl1dh_reader_type = ComponentName(
        DLDataReader, default_value="DLImageReader"
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
        trait=CaselessStrEnum(["type", "energy", "direction"]),
        allow_none=False, 
        help=(
            "List of reconstruction tasks to perform. "
            "'type': classification of the primary particle type "
            "'energy': regression of the primary particle energy "
            "'direction': regression of the primary particle arrival direction "
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
        help="Optimizer to use for training.",
    ).tag(config=True)

    lr_reducing = Dict(
        default_value={"factor": 0.5, "patience": 5, "min_delta": 0.01, "min_lr": 0.000001},
        allow_none=True,
        help="Learning rate reducing parameters for the Keras callback.",
    ).tag(config=True)

    random_seed = Int(
        default_value=0,
        help="Random seed for sampling training events.",
    ).tag(config=True)

    save_onnx = Bool(
        default_value=False,
        allow_none=False,
        help="Set whether to save model in an ONNX file.",
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
        # Create a MirroredStrategy.
        self.strategy = tf.distribute.MirroredStrategy()
        atexit.register(self.strategy._extended._collective_ops._lock.locked)  # type: ignore
        self.log.info("Number of devices: {}".format(self.strategy.num_replicas_in_sync))
        # Get signal input Files
        self.input_url_signal = sorted(self.input_dir_signal.glob(self.file_pattern_signal))
        # Get bkg input Files
        self.input_url_background = []
        if self.input_dir_background is not None:
            self.input_url_background = sorted(self.input_dir_background.glob(self.file_pattern_background))

        # Set up the data reader
        self.log.info("Loading data:")
        self.log.info("  For a large dataset, this may take a while...")
        self.dl1dh_reader = DLDataReader.from_name(
            self.dl1dh_reader_type,
            input_url_signal=self.input_url_signal,
            input_url_background=self.input_url_background,
            parent=self,
        )

        self.log.info(f"  Number of events loaded: {self.dl1dh_reader._get_n_events()}")

        # Check if there are at least two classes in the reader for the particle classification
        if self.dl1dh_reader.class_weight is None and "type" in self.reco_tasks:
            raise ValueError(
                "Classification task selected but less than two classes are present in the data."
            )

        # Set up the data loaders for training and validation
        indices = list(range(self.dl1dh_reader._get_n_events()))
        n_validation_examples = int(self.validation_split * self.dl1dh_reader._get_n_events())
        training_indices = indices[n_validation_examples:]
        validation_indices = indices[:n_validation_examples]
        self.dl1dh_training_loader = DLDataLoader(
            self.dl1dh_reader,
            training_indices,
            tasks=self.reco_tasks,
            batch_size=self.batch_size*self.strategy.num_replicas_in_sync,
            random_seed=self.random_seed,
            #stack_telescope_images=stack_telescope_images,
        )
        self.dl1dh_validation_loader = DLDataLoader(
            self.dl1dh_reader,
            validation_indices,
            tasks=self.reco_tasks,
            batch_size=self.batch_size*self.strategy.num_replicas_in_sync,
            random_seed=self.random_seed,
            #stack_telescope_images=stack_telescope_images,
        )

        # Set up the callbacks
        monitor = "val_loss"
        monitor_mode = "min"
        # Model checkpoint callback
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=f"{self.output_dir}/ctlearn_model.ckp",
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
        # Learning rate reducing callback
        if self.lr_reducing is not None:
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
                input_shape=self.dl1dh_training_loader.input_shape,
                tasks=self.reco_tasks,
                parent=self,
            ).model
            # Select optimizer with appropriate arguments
            adam_epsilon = self.optimizer["adam_epsilon"] if "adam_epsilon" in self.optimizer else None
            learning_rate =  self.optimizer["base_learning_rate"]
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
            optimizer_fn, optimizer_args = optimizers[self.optimizer["name"]]
            losses, metrics = self._get_losses_and_mertics(self.reco_tasks)
            self.log.info("Compiling CTLearn model.")
            self.model.compile(optimizer=optimizer_fn(**optimizer_args), loss=losses, metrics=metrics)

        # Train and evaluate the model
        self.log.info("Training and evaluating...")
        self.model.fit(
            self.dl1dh_training_loader,
            validation_data=self.dl1dh_validation_loader,
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
            self.log.info("ONNX model saved in {}".format(output_dir))

        # Plotting training history
        self.log.info("Plotting training history...")
        training_log = pd.read_csv(self.output_dir + "/training_log.csv")
        for metric in training_log.columns:
            epochs = training_log["epoch"] + 1
            if metric != "epoch" and not metric.startswith("val_"):
                self.log.info("Plotting training history: {}".format(metric))
                fig, ax = plt.subplots()
                plt.plot(epochs, training_log[metric])
                legend = ["train"]
                if f"val_{metric}" in training_log:
                    plt.plot(epochs, training_log[f"val_{metric}"])
                    legend.append("val")
                plt.title(f"CTLearn training history - {metric}")
                plt.xlabel("epoch")
                plt.ylabel(metric)
                plt.legend(legend, loc="upper left")
                plt.savefig(f"{self.output_dir}/{metric}.png")

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
        if "energy" in self.reco_tasks:
            losses["energy"] = keras.losses.MeanAbsoluteError(
                reduction="sum_over_batch_size"
            )
            metrics["energy"] = keras.metrics.MeanAbsoluteError(name="mae_energy")
        if "direction" in self.reco_tasks:
            losses["direction"] = keras.losses.MeanAbsoluteError(
                reduction="sum_over_batch_size"
            )
            metrics["direction"] = keras.metrics.MeanAbsoluteError(name="mae_direction")
        # Temp fix till keras support class weights for multiple outputs or I wrote custom loss
        # https://github.com/keras-team/keras/issues/11735
        if len(tasks) == 1 and tasks[0] == "type":
            losses = losses[tasks[0]]
            metrics = metrics[tasks[0]]
        return losses, metrics



def main():
    # Run the tool
    tool = TrainCTLearnModel()
    tool.run()


if __name__ == "main":
    main()















