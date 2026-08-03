"""
Tool to train a Keras-based ``CTLearnModel``on R1/DL1a data using the ``DLDataReader`` and ``DLDataLoader``.
"""

import atexit
import tensorflow as tf
import keras

from ctlearn.core.keras.sequence import KerasSequence
from ctlearn.core.model import CTLearnModel
from ctlearn.tools.train_model import TrainCTLearnModel


class TrainCTLearnKerasModel(TrainCTLearnModel):
    """
    Tool to train a ``~ctlearn.core.model.CTLearnModel`` Keras-based model on R1/DL1a data.

    The tool trains a CTLearn Keras-based model on the input data (R1 calibrated waveforms or DL1a images) and
    saves the trained model in the output directory. The input data is loaded from the input directories
    for signal and background events using the ``~dl1_data_handler.reader.DLDataReader`` and
    ``~dl1_data_handler.loader.DLDataLoader``. The tool supports the following reconstruction tasks:
    - Classification of the primary particle type (gamma/proton)
    - Regression of the primary particle energy
    - Regression of the primary particle arrival direction based on the offsets in camera coordinates
    - Regression of the primary particle arrival direction based on the offsets in sky coordinates
    """

    name = "ctlearn-train-keras-model"
    description = __doc__

    examples = """
    To train a Keras-based CTLearn model for the classification of the primary particle type:
    > ctlearn-train-keras-model \\
        --signal /path/to/your/gammas_dl1_dir/ \\
        --pattern-signal "gamma_*_run1.dl1.h5" \\
        --pattern-signal "gamma_*_run10.dl1.h5" \\
        --background /path/to/your/protons_dl1_dir/ \\
        --pattern-background "proton_*_run1.dl1.h5" \\
        --pattern-background "proton_*_run10.dl1.h5" \\
        --output /path/to/your/type/ \\
        --reco type \\

    To train a Keras-based CTLearn model for the regression of the primary particle energy:
    > ctlearn-train-keras-model \\
        --signal /path/to/your/gammas_dl1_dir/ \\
        --pattern-signal "gamma_*_run1.dl1.h5" \\
        --pattern-signal "gamma_*_run10.dl1.h5" \\
        --output /path/to/your/energy/ \\
        --reco energy \\

    To train a Keras-based CTLearn model for the regression of the primary particle
    arrival direction based on the offsets in camera coordinates:
    > ctlearn-train-keras-model \\
        --signal /path/to/your/gammas_dl1_dir/ \\
        --pattern-signal "gamma_*_run1.dl1.h5" \\
        --pattern-signal "gamma_*_run10.dl1.h5" \\
        --output /path/to/your/direction/ \\
        --reco cameradirection \\

    To train a Keras-based CTLearn model for the regression of the primary particle
    arrival direction based on the offsets in sky coordinates:
    > ctlearn-train-keras-model \\
        --signal /path/to/your/gammas_dl1_dir/ \\
        --pattern-signal "gamma_*_run1.dl1.h5" \\
        --pattern-signal "gamma_*_run10.dl1.h5" \\
        --output /path/to/your/direction/ \\
        --reco skydirection \\
    """

    def setup_framework(self):
        # Create a MirroredStrategy.
        self.strategy = tf.distribute.MirroredStrategy()
        atexit.register(self.strategy._extended._collective_ops._lock.locked)  # type: ignore
        self.log.info("Number of devices: %s", self.strategy.num_replicas_in_sync)

        # Init the DLDataLoader for the 
        self.training_loader = KerasSequence(
            DLDataReader=self.dl1dh_reader,
            indices=self.training_indices,
            tasks=self.reco_tasks,
            batch_size=self.batch_size * self.strategy.num_replicas_in_sync,
            random_seed=self.random_seed,
            sort_by_intensity=self.sort_by_intensity,
            stack_telescope_images=self.stack_telescope_images,
        )
        self.validation_loader = KerasSequence(
            DLDataReader=self.dl1dh_reader,
            indices=self.validation_indices,
            tasks=self.reco_tasks,
            batch_size=self.batch_size * self.strategy.num_replicas_in_sync,
            random_seed=self.random_seed,
            sort_by_intensity=self.sort_by_intensity,
            stack_telescope_images=self.stack_telescope_images,
        )

        # Set up the callbacks
        monitor = "val_loss"
        monitor_mode = "min"
        # Model checkpoint callback
        model_path = f"{self.output_dir}/ctlearn_model.keras"
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
        self.callbacks = [
            model_checkpoint_callback,
            tensorboard_callback,
            csv_logger_callback,
        ]

        if self.early_stopping is not None:
            # EarlyStopping callback
            early_stopping_callback = keras.callbacks.EarlyStopping(
                monitor=self.early_stopping["monitor"],
                patience=self.early_stopping["patience"],
                verbose=self.early_stopping["verbose"],
                restore_best_weights=self.early_stopping["restore_best_weights"],
            )
            self.callbacks.append(early_stopping_callback)

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
            self.log.info("Setting up the Keras model.")
            self.model = CTLearnModel.from_name(
                f"Keras{self.model_type}",
                input_shape=self.training_loader.input_shape,
                tasks=self.reco_tasks,
                parent=self,
            ).model

            # Select optimizer with appropriate arguments
            optimizers = {
                "Adadelta": lambda: keras.optimizers.Adadelta(learning_rate=self.learning_rate),
                "Adam": lambda: keras.optimizers.Adam(
                    learning_rate=self.learning_rate, epsilon=self.adam_epsilon
                ),
                "RMSProp": lambda: keras.optimizers.RMSprop(learning_rate=self.learning_rate),
                "SGD": lambda: keras.optimizers.SGD(learning_rate=self.learning_rate),
            }
            self.opt = optimizers[self.optimizer["name"]]()
           
            # Get the losses and metrics for the model
            losses, metrics = self._get_losses_and_mertics(self.reco_tasks)
            # Compile the model
            self.log.info("Compiling CTLearn model.")
            self.model.compile(
                optimizer=self.opt, loss=losses, metrics=metrics
            )

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
            metrics["cameradirection"] = keras.metrics.MeanAbsoluteError(
                name="mae_cameradirection"
            )
        if "skydirection" in self.reco_tasks:
            losses["skydirection"] = keras.losses.MeanAbsoluteError(
                reduction="sum_over_batch_size"
            )
            metrics["skydirection"] = keras.metrics.MeanAbsoluteError(
                name="mae_skydirection"
            )
        return losses, metrics


def main():
    # Run the tool
    tool = TrainCTLearnKerasModel()
    tool.run()


if __name__ == "main":
    main()