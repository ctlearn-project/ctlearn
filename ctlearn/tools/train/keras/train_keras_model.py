import atexit
import pandas as pd
import numpy as np
import shutil
import tensorflow as tf
from tqdm import tqdm
from time import time
from keras.callbacks import Callback

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
from ctlearn.core.data_loader.loader import DLDataLoader
from ctlearn.tools.train.base_train_model import TrainCTLearnModel
from ctlearn.core.keras.model import CTLearnModel
from ctlearn.utils import validate_trait_dict

try:
    import keras
except ImportError:
    raise ImportError("keras is not installed in your environment!")

class TqdmProgressBar(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time()
        self.progress_bar = tqdm(total=self.params['steps'], desc=f'Epoc {epoch + 1}/{self.params["epochs"]}', unit='batch')

    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time() - self.epoch_start_time
        print(f'\nEpoch Time {epoch + 1}: {epoch_duration:.2f} seconds')
        self.progress_bar.close()

    def on_batch_begin(self, batch, logs=None):
        self.batch_start_time = time()

    def on_batch_end(self, batch, logs=None):
        # batch_duration = time() - self.batch_start_time
        self.progress_bar.set_postfix(loss=logs.get('loss'), val_loss=logs.get('val_loss'))
        self.progress_bar.update(1)

class TrainKerasModel(TrainCTLearnModel):
    """
    Tool to train a ``~ctlearn.core.model.CTLearnModel`` on R1/DL1a data using keras.

    The tool sets up the keras model using the specified optimizer and callbacks. The keras model is trained
    on the input data (R1 calibrated waveforms or DL1a images) and saved in the output directory.
    """

    name = "ctlearn-train-keras-model"
    description = __doc__

    examples = """
    To train a CTLearn model for the classification of the primary particle type:
    > ctlearn-train-keras-model \\
        --signal /path/to/your/gammas_dl1_dir/ \\
        --pattern-signal "gamma_*_run1.dl1.h5" \\
        --pattern-signal "gamma_*_run10.dl1.h5" \\
        --background /path/to/your/protons_dl1_dir/ \\
        --pattern-background "proton_*_run1.dl1.h5" \\
        --pattern-background "proton_*_run10.dl1.h5" \\
        --output /path/to/your/type/ \\
        --reco type \\

    To train a CTLearn model for the regression of the primary particle energy:
    > ctlearn-train-keras-model \\
        --signal /path/to/your/gammas_dl1_dir/ \\
        --pattern-signal "gamma_*_run1.dl1.h5" \\
        --pattern-signal "gamma_*_run10.dl1.h5" \\
        --output /path/to/your/energy/ \\
        --reco energy \\

    To train a CTLearn model for the regression of the primary particle
    arrival direction based on the offsets in camera coordinates:
    > ctlearn-train-keras-model \\
        --signal /path/to/your/gammas_dl1_dir/ \\
        --pattern-signal "gamma_*_run1.dl1.h5" \\
        --pattern-signal "gamma_*_run10.dl1.h5" \\
        --output /path/to/your/direction/ \\
        --reco cameradirection \\

    To train a CTLearn model for the regression of the primary particle
    arrival direction based on the offsets in sky coordinates:
    > ctlearn-train-keras-model \\
        --signal /path/to/your/gammas_dl1_dir/ \\
        --pattern-signal "gamma_*_run1.dl1.h5" \\
        --pattern-signal "gamma_*_run10.dl1.h5" \\
        --output /path/to/your/direction/ \\
        --reco skydirection \\
    """

    model_type = ComponentName(
        CTLearnModel, default_value="ResNet"
    ).tag(config=True)

    save_best_validation_only = Bool(
        default_value=True,
        allow_none=False,
        help="Set whether to save the best validation checkpoint only.",
    ).tag(config=True)

    lr_reducing = Dict(
        default_value={"factor": 0.5, "patience": 5, "min_delta": 0.01, "min_lr": 0.000001},
        allow_none=True,
	help=(
	    "Learning rate reducing parameters for the Keras callback. "
	    "E.g. {'factor': 0.5, 'patience': 5, 'min_delta': 0.01, 'min_lr': 0.000001}. "
	)
    ).tag(config=True)

    early_stopping = Dict(
        default_value=None,
        allow_none=True,
	help=(
	    "Early stopping parameters for the Keras callback. "
	    "E.g. {'monitor': 'val_loss', 'patience': 4, 'verbose': 1, 'restore_best_weights': True}. "
	)
    ).tag(config=True)

    aliases = {
        **TrainCTLearnModel.aliases,  
    }    
	
    def setup(self):
        
        print(tf.config.list_physical_devices('GPU'))
        # Create a MirroredStrategy.
        self.strategy = tf.distribute.MirroredStrategy()
        atexit.register(self.strategy._extended._collective_ops._lock.locked)  # type: ignore
        self.log.info("Number of devices: %s", self.strategy.num_replicas_in_sync)
        # print(self.framework_type)
        super().setup()

        # Set up the data loaders for training and validation
        indices = list(range(self.dl1dh_reader._get_n_events()))
        # Shuffle the indices before the training/validation split
        np.random.seed(self.random_seed)
        np.random.shuffle(indices)
        n_validation_examples = int(
            self.validation_split * self.dl1dh_reader._get_n_events()
        )
        training_indices = indices[n_validation_examples:]
        validation_indices = indices[:n_validation_examples]

        # Set self.strategy.num_replicas_in_sync to 1 in case that does not exist (Pytorch)
        if not hasattr(self, "strategy"):
            self.strategy = type("FakeStrategy", (), {"num_replicas_in_sync": 1})()
            print("num_replicas_in_sync:", self.strategy.num_replicas_in_sync)

        print("BASE TRAIN FRAMEWORK", self.framework_type)
        
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
            indices=validation_indices,
            tasks=self.reco_tasks,
            batch_size=self.batch_size * self.strategy.num_replicas_in_sync,
            random_seed=self.random_seed,
            sort_by_intensity=self.sort_by_intensity,
            stack_telescope_images=self.stack_telescope_images,
        )
               
    def start(self):
        print("Start KERAS")
        print("EPOCHS:",self.n_epochs)
        print("save_onnx:",self.save_onnx)
        # Set up the keras callbacks
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

        tqdm_callback = TqdmProgressBar()
        self.callbacks.append(tqdm_callback)

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