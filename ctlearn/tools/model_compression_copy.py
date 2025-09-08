import tensorflow as tf
import pathlib
from ctapipe.core.traits import (Path, Bool, ComponentName, Dict, Unicode, List, CaselessStrEnum, Int, Float, classes_with_traits)
import keras
import atexit
import tensorflow_model_optimization as tfmot
from ctapipe.core import Tool
from ctapipe.core.tool import ToolConfigurationError
import shutil
import numpy as np

from ctlearn.tools.train_model import TrainCTLearnModel
from ctlearn.core.model import CTLearnModel 
from dl1_data_handler.reader import DLDataReader
import os


#from tensorflow_model_optimization.python.core.sparsity.keras import prune, prune_low_magnitude
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
import tensorflow_model_optimization.python.core.sparsity.keras as sparse_keras


__all__ = [
    "CompressCTLearnModel"
]

class CompressCTLearnModel(Tool):
    """Tool to compress a Keras model using different techniques.
    Current techniques supported include:
    - Post-training quantization (TFLite conversion is required)
    - Quantization-aware training
    - Pruning
    These techniques can be called in a user specified order with 
    limitation that, if post-training quantization is used it must
    be the last step in the compression process.
    This tool can also be called by the TrainCTLearn tool.
    """

    name = "ctlearn-compress-model"
    description = __doc__

    
    compression_techniques = List(
    trait=Dict(
        key_trait=Unicode(help="Compression technique name. " \
        "Valid names are 'qat' for Quantization-Aware training, 'pruning' and 'ptq' for post-training quantization."),
        value_trait=Dict(help="Parameters for the compression technique"),
    ),
    help=(
        "Specify the compression techniques to apply to the model as a list of tuples. "
        "The order specified will be used, except if post-training quantization (ptq) is not called last."
        "In that case, the order of the other techniques will be preserved, but ptq will be applied last. "
        "Each tuple contains the technique name and its parameters. "
        "For example: [{'qat': {'epochs': 10}}, {'pruning': {'initial_sparsity': 0.2, 'final_sparsity': 0.8}}, "
        "{'ptq': {'quantization_type': 'float16'}}]."
    ),
    default_value=[],
    ).tag(config=True)
    
    model_type = ComponentName(
        CTLearnModel, default_value="ResNet", allow_none=True,
    ).tag(config=True)

    load_type_model_from = Path(
        default_value=None,
        help=(
            "Path to a Keras model file (Keras3) or directory (Keras2) for the classification "
            "of the primary particle type."
        ),
        allow_none=True,
        exists=True,
        directory_ok=True,
        file_ok=True,
    ).tag(config=True)

    load_energy_model_from = Path(
        default_value=None,
        help=(
            "Path to a Keras model file (Keras3) or directory (Keras2) for the regression "
            "of the primary particle energy."
        ),
        allow_none=True,
        exists=True,
        directory_ok=True,
        file_ok=True,
    ).tag(config=True)

    load_model_from = Path(
        default_value=None,
        help=(
            "Path to a Keras model file (Keras3) or directory (Keras2) for the regression "
            "of the primary particle arrival direction based on camera coordinate offsets."
        ),
        allow_none=True,
        exists=True,
        directory_ok=True,
        file_ok=True,
    ).tag(config=True)

    load_cameradirection_model_from = Path(
        default_value=None,
        help=(
            "Path to a Keras model file (Keras3) or directory (Keras2) for the regression "
            "of the primary particle arrival direction based on camera coordinate offsets."
        ),
        allow_none=True,
        exists=True,
        directory_ok=True,
        file_ok=True,
    ).tag(config=True)

    load_skydirection_model_from = Path(
        default_value=None,
        help=(
            "Path to a Keras model file (Keras3) or directory (Keras2) for the regression "
            "of the primary particle arrival direction based on spherical coordinate offsets."
        ),
        allow_none=True,
        exists=True,
        directory_ok=True,
        file_ok=True,
    ).tag(config=True)

    output_dir = Path(
        exits=False,
        default_value=None,
        allow_none=False,
        directory_ok=True,
        file_ok=False,
        help="Output directory for the trained reconstructor.",
    ).tag(config=True)

    summaries_dir = Path(
        default_value=None,
        allow_none=True,
        directory_ok=True,
        file_ok = False,
        help="Output directory for the pruning summaries.",
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

    model_type = ComponentName(
        CTLearnModel, default_value="ResNet"
    ).tag(config=True)

    reco_tasks = CaselessStrEnum(["type", "energy", "cameradirection", "skydirection"],
        allow_none=False, 
        help=(
            "List of reconstruction tasks to perform. "
            "'type': classification of the primary particle type "
            "'energy': regression of the primary particle energy "
            "'cameradirection': regression of the primary particle arrival direction in camera coordinates "
            "'skydirection': regression of the primary particle arrival direction in sky coordinates"
        ),
    ).tag(config=True)

    n_epochs = Int(
        allow_none=True,
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
        "signal": "CompressCTLearnModel.input_dir_signal",
        "background": "CompressCTLearnModel.input_dir_background",
        "pattern_signal": "CompressCTLearnModel.file_pattern_signal",
        "pattern_background": "CompressCTLearnModel.file_pattern_background",
        "model_type": "CompressCTLearnModel.model_type",
        #("t", "type_model"): "CompressPredictCTLearnModel.load_type_model_from",
        #("e", "energy_model"): "CompressPredictCTLearnModel.load_energy_model_from",
        #("d", "cameradirection_model", ): "CompressCTLearnModel.load_cameradirection_model_from",
        #("s", "skydirection_model"): "CompressCTLearnModel.load_skydirection_model_from",
        ("o", "output_dir"): "CompressCTLearnModel.output_dir",
        "sumdir": "CompressCTLearnModel.summaries_dir",
        "reco": "CompressCTLearnModel.reco_tasks",
        ("mcomp", "model_compression"): "CompressCTLearnModel.compression_techniques",
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
        print("hiyaaa")
        print(f"mcOutput directory: {self.output_dir}")
        # Check if the output directory exists and if it should be overwritten
        if self.output_dir.exists():
            if not self.overwrite:
                print("mcoverwriting...")
                self.log.warning(
                    f"mcOutput directory {self.output_dir} already exists. Proceeding without overwriting."
                    )
                raise ToolConfigurationError(
                    f"mcOutput directory {self.output_dir} already exists. Use --overwrite to overwrite."
                    )
                
            else:
                # Remove the output directory if it exists
                print("mcremoving...")
                self.log.info("mcRemoving existing output directory %s", self.output_dir)
                shutil.rmtree(self.output_dir)
        self.log.info("mcOutput directory removed. Proceeding with compression...")
        print("mcOutput directory removed. Proceeding with compression...")
        """try:
            cwd = os.getcwd()
        except FileNotFoundError:
            cwd = "/tmp"  # Fallback to a default directory
            os.chdir(cwd)
            self.log.warning(f"Working directory not found. Falling back to {cwd}.")
        """
        # Ensure the parent directory for the summaries file exists
        
        if self.summaries_dir:
            if not self.summaries_dir.exists():
                self.log.info(f"mcCreating pruning summaries directory: {self.summaries_dir}")
                self.summaries_dir.mkdir(parents=True, exist_ok=True)

        """if self.summaries_dir is not None:
            if not self.summaries_dir.exists():
                self.log.info(f"Creating pruning summaries directory: {self.summaries_dir}")
                self.summaries_dir.mkdir(parents=True, exist_ok=True)"""
            
            # Create a MirroredStrategy.
        self.strategy = tf.distribute.MirroredStrategy()
        atexit.register(self.strategy._extended._collective_ops._lock.locked)  # type: ignore
        self.log.info("mcNumber of devices: %s", self.strategy.num_replicas_in_sync)
        print("mcNumber of devices: %s", self.strategy.num_replicas_in_sync)

        if int(keras.__version__.split(".")[0]) >= 3:
            self.output_dir = f"{self.output_dir}/ctlearn_model.keras"
        else:
            self.output_dir = f"{self.output_dir}/ctlearn_model.cpk"

        print("mcMirrored strategy ok")
        
        
    def start(self):
        """
        Start the compression process.
        Here the model is loaded and the compression techniques are applied.
        """
        print("mcStarting")
        # Instantiate TrainCTLearnModel
        train_tool = TrainCTLearnModel()
        
        # Pass variables from CompressCTLearnModel to TrainCTLearnModel
        train_tool.input_dir_signal = self.input_dir_signal
        train_tool.file_pattern_signal = self.file_pattern_signal
        train_tool.input_dir_background = self.input_dir_background
        train_tool.file_pattern_background = self.file_pattern_background
        train_tool.output_dir = self.output_dir
        train_tool.reco_tasks = self.reco_tasks
        #train_tool.n_epochs = self.n_epochs
        train_tool.n_epochs = self.compression_techniques["pruning"].get("epochs")
        train_tool.batch_size = self.batch_size
        train_tool.validation_split = self.validation_split
        train_tool.optimizer = self.optimizer
        train_tool.lr_reducing = self.lr_reducing
        train_tool.early_stopping = self.early_stopping
        train_tool.log_file = self.log_file
        

        print(f"mcSignal directory: {train_tool.input_dir_signal}")
        print(f"mcSignal file pattern: {train_tool.file_pattern_signal}")
        print(f"mcself Signal directory: {self.input_dir_signal}")
        print(f"mcself Signal file pattern: {self.file_pattern_signal}")



        self.log.info("mcLoading Keras model from %s", self.load_model_from)
        
        self.model = self.load_model2()
        
        print("Organizing compression techniques...")
        self.log.info("Organizing model compression techniques...")
        counter = 0
        for i, technique_dict in enumerate(self.compression_techniques):
            if "ptq" in technique_dict and i!= len(self.compression_techniques) - 1:
                self.compression_techniques.append(self.compression_techniques.pop(i))
                counter+=1
            if counter > 1:
                self.log.warning("Multiple post-training quantization techniques found. " \
                "Only the first found is applied. Remember post-training quantization must be the last step in the compression process.")

        self.log.debug("mcCompression techniques to be applied: %s", self.compression_techniques)
        print("Compression techniques to be applied: %s", self.compression_techniques)

        for technique in self.compression_techniques:
            for key, hyperparams in technique.items():
                print("Technique: ", key)
                if key == "ptq":
                    self.log.info("Applying post-training quantization...")
                ###### Needs to be filled up later
                elif key == "qat":
                    self.log.info("Applying Quantization-Aware Training...")
                elif key == "pruning":
                    self.log.info("Building pruning schedule...")
                    pruning_params = self.build_pruning_schedule(hyperparams)
                    
                    self.n_epochs = hyperparams.get("n_epochs", 3)
                    self.batch_size = hyperparams.get("batch_size", 64)
                    self.log.info("Calling pruning...")
                    self.compressed_model = self.pruning(train_tool)
                    self.log.info("Applying pruning...")
                elif technique == "None":
                    self.log.info("No compression technique applied.")




    def post_training_quantization(self, quantization_type="default"):
        """
        Apply post-training quantization to the model.
        Parameters:
            quantization_type (str): Type of quantization ('default', 'float16', 'int8').
        Returns:
            str: Path to the compressed model file.
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        if quantization_type == "default":
            # Default quantization applies dynamic range quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif quantization_type == "float16":
            # IEEE standard for 16-bit floating point numbers
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif quantization_type == "smooth_int8":
            # int8 quantization for weights but float operators when they 
            # don't have an integer implementation 
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self._representative_dataset_gen()
        elif quantization_type == "full_int8":
            # Full int8 quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self._representative_dataset_gen()
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8
            converter.inference_output_type = tf.int8  # or tf.uint8
      
        quant_model = converter.convert()
        ## needs to be checked from here on
        model_path = Path(f"{self.output_dir}/compressed_model_{quantization_type}.tflite")
        model_path.write_bytes(quant_model)
        self.log.info("Successfully applied post-training quantization with TFLite conversion.")
        
        

    def quantization_aware_training(self, train_dataset, epochs=10):
        """
        Apply Quantization-Aware Training (QAT) to the model.
        Args:
            train_dataset: Dataset for training the model with QAT.
            epochs (int): Number of epochs for QAT.
        Returns:
            tf.keras.Model: Quantized model after training.
        """
        quantize_model = tfmot.quantization.keras.quantize_model(self.model)
        #if self.load_m
        quantize_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        quantize_model.fit(train_dataset, epochs=epochs)

        return quantize_model

    def pruning(self, train_tool, pruning_params, train_dataset=None):
        """
        Apply pruning to the model.
        Args:
            train_dataset: Dataset for training the model with pruning (optional).
            epochs (int): Number of epochs for pruning (if training is required).
        Returns:
            tf.keras.Model: Pruned model.
        """
        print("hello")
        #prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        ##end_step = np.ceil(num_images / batch_size).astype(np.int32) * pruning_epochs
        
        """pruning_params = {'pruning_schedule': sparse_keras.pruning_schedule.ConstantSparsity(
            target_sparsity =0.50,
            begin_step = 0,
            end_step = -1,
            frequency = 100
            )}"""
        

        self.pruned_model_stage1 = prune_low_magnitude(self.model, **pruning_params)

        # It would be a good idea to load log_dir as Unicode and avoid Path and str conversion later.
        pruning_callbacks = [tfmot.sparsity.keras.UpdatePruningStep(), tfmot.sparsity.keras.PruningSummaries(log_dir=str(self.summaries_dir))]
        print("callbacks added")
        #train_tool = TrainCTLearnModel()
        #train_tool.setup()  # Call setup to initialize callbacks and other attributes
        print("extending next")
        #train_tool.callbacks = train_tool.callbacks.extend(callbacks)
        train_tool.callbacks.extend(pruning_callbacks)
        print("Callbacks in train_tool:")
        for callback in train_tool.callbacks:
            print(type(callback), callback)
        print("mmmmmccccccno, but in all seriousness")
        train_tool.run()
        """
        pruned_model = prune_low_magnitude(self.model, **pruning_params)
        # check

        train_tool.input_dir_signal = self.input_dir_signal

        model_for_pruning.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adam(),
              metrics=['accuracy'])

        # Model callbacks
        callbacks = [tfmot.keras.UpdatePruningStep(), tfmot.keras.PruningSummaries()]

        # Fitting data
        model_for_pruning.fit(input_train, target_train,
                      batch_size=batch_size,
                      epochs=pruning_epochs,
                      verbose=verbosity,
                      callbacks=callbacks,
                      validation_split=validation_split)
        
        #callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
        
        
        pruning_params = {
            "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.2, final_sparsity=0.8, begin_step=0, end_step=1000
            )
        }
        pruned_model = prune_low_magnitude(self.model, **pruning_params)

        train_tool = TrainCTLearnModel()
        train_tool.n_epochs = self.compression_techniques["pruning"].get("epochs")
        train_tool.batch_size = self.compression_techniques["pruning"].get("batch_size", 1)
        train_tool.callbacks = train_tool.callbacks.extend(self.compression_techniques["pruning"].get("callbacks", []))
        """


    def _representative_dataset_gen(self):
        for _ in range(100):
            yield [tf.random.normal([1, 224, 224, 3])]


    def load_model(self):
        if self.load_model_from:
            self.log.info("Loading model from %s", self.load_model_from)
            self.model = keras.models.load_model(self.load_model_from)
            return self.model
        else:
            self.log.info("Creating new model of type %s", self.model_type)

            

        model_pretrained = [select_model for select_model in trained_model if select_model is not None]
        
        if len(model_pretrained) > 1 or (len(model_pretrained) >= 1 and self.model_type is not None):
            raise ToolConfigurationError(
                "Multiple model paths provided. Please specify only one model path."
            )
        elif len(model_pretrained) == 1 and self.model_type is None:
            chosen_model = keras.models.load_model(model_pretrained[0])
        elif len(model_pretrained) == 0 and self.model_type is not None:
            chosen_model = CTLearnModel.from_name(
                self.model_type,
                input_shape=self.training_loader.input_shape,
                tasks=self.reco_tasks,
                parent=self,
            ).model
        elif len(model_pretrained) == 0 and self.model_type is None:
            raise ToolConfigurationError(
                "No model path provided. Please specify a model path or a model type."
            )
        return chosen_model
         #train_model in pretrained_models is None

    def load_model2(self):
        if self.model_type == "LoadedModel":
                if self.load_model_from == None:
                    raise ToolConfigurationError(
                        "No model path provided. Please specify a model path."
                    )
                if self.overwrite_head or self.trainable.backbone is None:
                    raise Warning("Backbone is trainable but head will not be overwritten")
                
                #model_reco = self.load_{self.reco}_model_from
                print("specified model will be loaded")
        elif self.model_type != "LoadedModel" and self.model_type is not None:
            print("Untrained model will be used")
        else:
            print("Default ResNet model will be used")




    def loadmodel3(self):
        model_reco = (getattr(self, f"load_{self.reco_tasks}_model_from"))
        ###try? self.load_{self.reco_tasks}_model_from = (getattr(self, f"load_{self.reco_tasks}_model_from"))
        #model_reco2 = (self, f"load_{self.reco_tasks}_model_from")
        print(f"Loading model from {model_reco}")
        #self.model2 = keras.models.load_model(pathlib.Path(model_reco2))
        print(f"load_{self.reco_tasks}_model_from", type(model_reco))
        self.model = keras.models.load_model(model_reco)
        print("ok")
                
        if (self.load_type_model_from or self.load_energy_model_from or self.load_cameradirection_model_from or self.load_skydirection_model_from) is None:
            print("Loading new model")
            with self.strategy.scope():
            # Construct the model
                self.log.info("Setting up the model.")
                self.model = CTLearnModel.from_name(
                self.model_type,
                input_shape=self.training_loader.input_shape,
                tasks=self.reco_tasks,
                parent=self,
            ).model
                
        return 
        #if self.load_model_from is None:
        #    chosen_model = self.model_type
        #elif self.load_model_from is not None and self.load_model_from is in self.reco:
        #    model_reco = "self.load_{self.reco}_model_from"
        #    chosen_model = 

    def build_pruning_schedule(self, hyperparams):
        """
        Build the pruning schedule based on user-defined parameters.
        Args:
            hyperparams (dict): Dictionary containing pruning parameters.
        Returns:
            tfmot.sparsity.keras.PruningSchedule: Pruning schedule object.
        """
        schedule_type = hyperparams.get("schedule_type", "polynomial_decay")  # Default to polynomial decay
        if schedule_type == "polynomial_decay":
            schedule_values = tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=hyperparams.get("initial sparsity", 0.2),
                final_sparsity=hyperparams.get("final sparsity", 0.6),
                begin_step=hyperparams.get("begin step", 0),
                end_step=hyperparams.get("end step", -1),
            )
        elif schedule_type == "constant_sparsity":
            schedule_values = tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=hyperparams.get("target sparsity", 0.5),
                begin_step=hyperparams.get("begin step", 0),
                end_step=hyperparams.get("end step", -1),
                frequency=hyperparams.get("frequency", 100),
            )
        else:
            raise ValueError(f"Unsupported pruning schedule type: {schedule_type}")
        
        pruning_params = {'pruning_schedule': schedule_values}
        
        return pruning_params
            
                

def main():
    # Run the tool
    tool = CompressCTLearnModel()
    tool.run()


if __name__ == "main":
    main()