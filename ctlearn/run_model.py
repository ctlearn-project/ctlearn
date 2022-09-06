import atexit
import argparse
import importlib
import logging
import math
from random import randint
import os
import glob
from pprint import pformat
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import tf2onnx

from dl1_data_handler.reader import DL1DataReaderSTAGE1, DL1DataReaderDL1DH
from ctlearn.data_loader import KerasBatchGenerator
from ctlearn.output_handler import *
from ctlearn.utils import *


def run_model(config, mode="train", debug=False, log_to_file=False):

    # Load options relating to logging and checkpointing
    root_model_dir = model_dir = config["Logging"]["model_directory"]

    random_seed = None
    if config["Logging"].get("add_seed", False):
        random_seed = config["Data"]["seed"]
        model_dir += f"/seed_{random_seed}"
        if not os.path.exists(model_dir):
            if mode == "predict":
                raise ValueError(
                    f"Invalid output directory '{model_dir}'. "
                    "Must be a path to an existing directory in the predict mode."
                )
            os.makedirs(model_dir)

    # Read class names from the example identifiers file
    class_names = None
    if mode == "predict" and os.path.isfile(model_dir + "/example_identifiers_file.h5"):
        example_identifiers_file = pd.HDFStore(
            model_dir + "/example_identifiers_file.h5"
        )
        if "/class_names" in list(example_identifiers_file.keys()):
            class_names = pd.read_hdf(
                example_identifiers_file, key="/class_names"
            ).to_dict("records")
            class_names = [name[0] for name in class_names]

    # Set up logging, saving the config and optionally logging to a file
    logger = setup_logging(config, model_dir, debug, log_to_file)

    # Log the loaded configuration
    logger.debug(pformat(config))

    logger.info("Logging has been correctly set up")

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    atexit.register(strategy._extended._collective_ops._pool.close)  # type: ignore
    logger.info("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # Set up the DL1DataReader
    config["Data"], data_format = setup_DL1DataReader(config, mode)
    # Create data reader
    logger.info("Loading data:")
    logger.info("  For a large dataset, this may take a while...")
    if data_format == "stage1":
        reader = DL1DataReaderSTAGE1(**config["Data"])
    elif data_format == "dl1dh":
        reader = DL1DataReaderDL1DH(**config["Data"])

    logger.info("  Number of events loaded: {}".format(len(reader)))

    # Set up the KerasBatchGenerator
    indices = list(range(len(reader)))

    if "Input" not in config:
        config["Input"] = {}
    batch_size_per_worker = config["Input"].get("batch_size_per_worker", 64)
    batch_size = batch_size_per_worker * strategy.num_replicas_in_sync
    concat_telescopes = config["Input"].get("concat_telescopes", False)

    if mode == "train":
        if "Training" not in config:
            config["Training"] = {}
        validation_split = np.float(config["Training"].get("validation_split", 0.1))
        if not 0.0 < validation_split < 1.0:
            raise ValueError(
                "Invalid validation split: {}. "
                "Must be between 0.0 and 1.0".format(validation_split)
            )
        num_training_examples = math.floor((1 - validation_split) * len(reader))
        training_indices = indices[:num_training_examples]
        validation_indices = indices[num_training_examples:]

        data = KerasBatchGenerator(
            reader,
            training_indices,
            batch_size=batch_size,
            mode=mode,
            class_names=class_names,
            concat_telescopes=concat_telescopes,
        )
        validation_data = KerasBatchGenerator(
            reader,
            validation_indices,
            batch_size=batch_size,
            mode=mode,
            class_names=class_names,
            concat_telescopes=concat_telescopes,
        )
    elif mode == "predict":
        logger.info(
            "  Simulation info for pyirf.simulations.SimulatedEventsInfo: {}".format(
                reader.simulation_info
            )
        )
        data = KerasBatchGenerator(
            reader,
            indices,
            batch_size=batch_size,
            mode=mode,
            class_names=class_names,
            shuffle=False,
            concat_telescopes=concat_telescopes,
        )

        # Keras is only considering the last complete batch.
        # In prediction mode we don't want to loose the last
        # uncomplete batch, so we are creating an adiitional
        # batch generator for the remaining events.
        rest_data = None
        rest = len(indices) % batch_size
        if rest > 0:
            rest_indices = indices[-rest:]
            rest_data = KerasBatchGenerator(
                reader,
                rest_indices,
                batch_size=rest,
                mode=mode,
                class_names=class_names,
                shuffle=False,
                concat_telescopes=concat_telescopes,
            )

    # Construct the model
    model_file = config["Model"].get("model_file", None)
    logger.info("Setting up model:")

    model_directory = config["Model"].get(
        "model_directory",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "default_models/")),
    )

    sys.path.append(model_directory)
    logger.info("  Constructing model from config.")
    # Write the model parameters in the params dictionary
    model_params = {**config["Model"], **config.get("Model Parameters", {})}
    model_params["model_directory"] = model_directory
    model_params["num_classes"] = reader.num_classes

    # Open a strategy scope.
    with strategy.scope():

        # Backbone model
        backbone_module = importlib.import_module(config["Model"]["backbone"]["module"])
        backbone_model = getattr(
            backbone_module, config["Model"]["backbone"]["function"]
        )
        backbone, backbone_inputs = backbone_model(data, model_params)
        backbone_output = backbone(backbone_inputs)

        # Head model
        head_module = importlib.import_module(config["Model"]["head"]["module"])
        head_model = getattr(head_module, config["Model"]["head"]["function"])
        logits, losses, loss_weights, metrics = head_model(
            inputs=backbone_output, tasks=config["Reco"], params=model_params
        )

        if "saved_model.pb" in np.array([os.listdir(model_dir)]):
            logger.info("  Loading weights from '{}'.".format(model_dir))
            model = tf.keras.models.load_model(model_dir)
        else:
            model = tf.keras.Model(backbone_inputs, logits, name="CTLearn_model")

        if config["Model"].get("plot_model", False) and mode == "train":
            logger.info(
                "  Saving the backbone architecture in '{}/backbone.png'.".format(
                    model_dir
                )
            )
            tf.keras.utils.plot_model(
                backbone,
                to_file=model_dir + "/backbone.png",
                show_shapes=True,
                show_layer_names=True,
            )
            logger.info(
                "  Saving the model architecture in '{}/model.png'.".format(model_dir)
            )
            tf.keras.utils.plot_model(
                model,
                to_file=model_dir + "/model.png",
                show_shapes=True,
                show_layer_names=True,
            )

        logger.info("  Model has been correctly set up from config.")

        optimizer_name = config["Training"].get("optimizer", "Adam")
        adam_epsilon = float(config["Training"].get("adam_epsilon", 1.0e-8))
        learning_rate = float(config["Training"].get("base_learning_rate", 0.0001))

        # Select optimizer with appropriate arguments
        # Dict of optimizer_name: (optimizer_fn, optimizer_args)
        optimizers = {
            "Adadelta": (
                tf.keras.optimizers.Adadelta,
                dict(learning_rate=learning_rate),
            ),
            "Adam": (
                tf.keras.optimizers.Adam,
                dict(learning_rate=learning_rate, epsilon=adam_epsilon),
            ),
            "RMSProp": (tf.keras.optimizers.RMSprop, dict(learning_rate=learning_rate)),
            "SGD": (tf.keras.optimizers.SGD, dict(learning_rate=learning_rate)),
        }
        optimizer_fn, optimizer_args = optimizers[optimizer_name]
        optimizer = optimizer_fn(**optimizer_args)
        logger.info("  Compiling model.")
        model.compile(optimizer=optimizer, loss=losses, metrics=metrics)

    if mode == "train":
        logger.info("Setting up training:")
        logger.info("  Validation split: {}".format(validation_split))

        if not 0.0 < validation_split < 1.0:
            raise ValueError(
                "Invalid validation split: {}. "
                "Must be between 0.0 and 1.0".format(validation_split)
            )
        num_epochs = int(config["Training"].get("num_epochs", 10))
        logger.info("  Number of epochs: {}".format(num_epochs))
        logger.info(
            "  Size of the batches per worker: {}".format(batch_size_per_worker)
        )
        logger.info("  Size of the batches: {}".format(batch_size))
        num_training_examples = math.floor((1 - validation_split) * len(reader))
        logger.info(
            "  Number of training steps per epoch: {}".format(
                int(num_training_examples / batch_size)
            )
        )
        logger.info("  Optimizer: {}".format(optimizer_name))
        logger.info("  Learning rate: {}".format(learning_rate))
        lr_reducing_patience = int(config["Training"].get("lr_reducing_patience", 5))
        logger.info(
            "  Learning rate reducing patience: {}".format(lr_reducing_patience)
        )
        lr_reducing_factor = config["Training"].get("lr_reducing_factor", 0.5)
        logger.info("  Learning rate reducing factor: {}".format(lr_reducing_factor))
        lr_reducing_mindelta = config["Training"].get("lr_reducing_mindelta", 0.01)
        logger.info(
            "  Learning rate reducing min delta: {}".format(lr_reducing_mindelta)
        )
        lr_reducing_minlr = config["Training"].get(
            "lr_reducing_minlr", 0.1 * learning_rate
        )
        logger.info("  Learning rate reducing min lr: {}".format(lr_reducing_minlr))
        verbose = int(config["Training"].get("verbose", 2))
        logger.info("  Verbosity mode: {}".format(verbose))
        workers = int(config["Training"].get("workers", 1))
        # ToDo: Fix multiprocessing issue
        workers = 1
        logger.info("  Number of workers: {}".format(workers))
        use_multiprocessing = True if workers > 1 else False
        logger.info("  Use of multiprocessing: {}".format(use_multiprocessing))

        # Set up the callbacks
        monitor = "val_loss" if reader.num_classes < 3 else "loss"
        monitor_mode = "min"

        # Model checkpoint callback
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_dir,
            monitor=monitor,
            verbose=1,
            mode=monitor_mode,
            save_best_only=True,
        )
        # Tensorboard callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=model_dir, histogram_freq=1
        )
        # CSV logger callback
        csv_logger_callback = tf.keras.callbacks.CSVLogger(
            filename=model_dir + "/training_log.csv", append=True
        )
        # Learning rate reducing callback
        lr_reducing_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=lr_reducing_factor,
            patience=lr_reducing_patience,
            mode=monitor_mode,
            verbose=1,
            min_delta=lr_reducing_mindelta,
            min_lr=lr_reducing_minlr,
        )

        callbacks = [
            model_checkpoint_callback,
            tensorboard_callback,
            csv_logger_callback,
            lr_reducing_callback,
        ]

        # Class weights calculation
        # Scaling by total/2 helps keep the loss to a similar magnitude.
        # The sum of the weights of all examples stays the same.
        class_weight = None
        if reader.num_classes >= 2:
            logger.info("  Apply class weights:")
            total = reader.simulated_particles["total"]
            logger.info("    Total number: {}".format(total))
            for particle_id, num_particles in reader.simulated_particles.items():
                if particle_id != "total":
                    logger.info(
                        "    Breakdown by '{}' ({}) with original particle id '{}': {}".format(
                            reader.shower_primary_id_to_name[particle_id],
                            reader.shower_primary_id_to_class[particle_id],
                            particle_id,
                            num_particles,
                        )
                    )
            logger.info("    Class weights: {}".format(reader.class_weight))

        initial_epoch = 0
        if "training_log.csv" in os.listdir(model_dir):
            initial_epoch = (
                pd.read_csv(model_dir + "/training_log.csv")["epoch"].iloc[-1] + 1
            )

        # Train and evaluate the model
        logger.info("Training and evaluating...")
        history = model.fit(
            x=data,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=num_epochs,
            initial_epoch=initial_epoch,
            class_weight=reader.class_weight,
            callbacks=callbacks,
            verbose=verbose,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
        )
        logger.info("Training and evaluating finished succesfully!")
        model.save(model_dir)
        logger.info("Keras model saved in {}saved_model.pb".format(model_dir))
        logger.info("Converting Keras model into ONNX format...")
        input_type_spec = [input._type_spec for input in backbone_inputs]
        output_path = model_dir + model.name + ".onnx"
        tf2onnx.convert.from_keras(
            model, input_signature=input_type_spec, output_path=output_path
        )
        logger.info("ONNX model saved in {}".format(output_path))

        # Plotting training history
        training_log = pd.read_csv(model_dir + "/training_log.csv")
        for metric in training_log.columns:
            epochs = training_log["epoch"] + 1
            if metric != "epoch" and not metric.startswith("val_"):
                logger.info("Plotting training history: {}".format(metric))
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
                plt.savefig(f"{model_dir}/{metric}.png")

    elif mode == "predict":
        # Generate predictions and add to output
        logger.info("Predicting...")
        predictions = model.predict(data)
        if rest_data:
            predictions = np.concatenate((predictions, model.predict(rest_data)), axis=0)

        prediction_file = config["Prediction"]["prediction_file"].replace(".h5", "")
        if random_seed:
            prediction_file += f"_{random_seed}"
        prediction_file = f"{prediction_file}.h5"
        write_output(
            prediction_file,
            data,
            rest_data,
            reader,
            predictions,
            config["Reco"],
        )

    # clear the handlers, shutdown the logging and delete the logger
    logger.handlers.clear()
    logging.shutdown()
    del logger
    return


def main():

    parser = argparse.ArgumentParser(
        description=("Train/Predict with a CTLearn model.")
    )
    parser.add_argument(
        "--config_file",
        "-c",
        help="Path to YAML configuration file with training options",
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Input directories (not required when file_list is set in the config file)",
        nargs="+",
    )
    parser.add_argument(
        "--pattern",
        "-p",
        help="Pattern to mask unwanted files from the data input directory",
        default=["*.h5"],
        nargs="+",
    )
    parser.add_argument(
        "--mode",
        "-m",
        default="train",
        help="Mode to run CTLearn; valid options: train, predict, or train_and_predict",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory, where the logging, model weights and plots are stored",
    )
    parser.add_argument(
        "--reco",
        "-r",
        help="Reconstruction task to perform; valid options: particletype, energy, and/or direction",
        nargs="+",
    )
    parser.add_argument(
        "--default_model",
        "-d",
        help="Default CTLearn Model; valid options: TRN (mono), mergedTRN (stereo), and CNNRNN (stereo)",
    )
    parser.add_argument(
        "--cleaned_images",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Flag, if the network should be trained with cleaned images",
    )
    parser.add_argument(
        "--pretrained_weights", "-w", help="Path to the pretrained weights"
    )
    parser.add_argument(
        "--prediction_directory",
        "-y",
        help="Path to store the CTLearn predictions (optional)",
    )
    parser.add_argument(
        "--tel_types",
        "-t",
        help="Selection of telescope types; valid option: LST_LST_LSTCam, LST_MAGIC_MAGICCam, MST_MST_FlashCam, MST_MST_NectarCam, SST_SCT_SCTCam, and/or SST_ASTRI_ASTRICam",
        nargs="+",
    )
    parser.add_argument(
        "--allowed_tels",
        "-a",
        type=int,
        help="List of allowed tel_ids, others will be ignored. Selected tel_ids will be ignored, when their telescope type is not selected",
        nargs="+",
    )
    parser.add_argument(
        "--size_cut", "-z", type=float, help="Hillas intensity cut to perform"
    )
    parser.add_argument(
        "--leakage_cut", "-l", type=float, help="Leakage intensity cut to perform"
    )
    parser.add_argument(
        "--multiplicity_cut", "-u", type=int, help="Multiplicity cut to perform"
    )
    parser.add_argument(
        "--num_epochs", "-e", type=int, help="Number of epochs to train"
    )
    parser.add_argument("--batch_size", "-b", type=int, help="Batch size per worker")
    parser.add_argument(
        "--random_seed", "-s", type=int, help="Selection of random seed (4 digits)"
    )
    parser.add_argument(
        "--log_to_file",
        action="store_true",
        help="Log to a file in model directory instead of terminal",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print debug/logger messages"
    )

    args = parser.parse_args()

    # Use the default CTLearn config file if no config file is provided
    # and a default CTLearn model is selected.
    if args.default_model and not args.config_file:
        default_config_files = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "default_config_files/")
        )
        args.config_file = f"{default_config_files}/{args.default_model}.yml"

    with open(args.config_file, "r") as config_file:
        config = yaml.safe_load(config_file)

    if args.reco:
        config["Reco"] = args.reco

    if args.cleaned_images:
        config["Data"]["image_channels"] = [
            "cleaned_" + channel for channel in config["Data"]["image_channels"]
        ]

    if args.tel_types:
        config["Data"]["selected_telescope_types"] = args.tel_types

    if args.allowed_tels:
        config["Data"]["selected_telescope_ids"] = args.allowed_tels

    parameter_selection = []
    if args.size_cut:
        parameter_selection.append(
            {"col_name": "hillas_intensity", "min_value": args.size_cut}
        )

    if args.leakage_cut:
        parameter_selection.append(
            {"col_name": "leakage_intensity_width_2", "max_value": args.leakage_cut}
        )

    for parameter in config["Data"].get("parameter_selection", []):
        if parameter["col_name"] == "hillas_intensity" and args.size_cut:
            continue
        if parameter["col_name"] == "leakage_intensity_width_2" and args.leakage_cut:
            continue
        parameter_selection.append(parameter)

    if parameter_selection:
        config["Data"]["parameter_selection"] = parameter_selection

    if args.multiplicity_cut:
        config["Data"]["multiplicity_selection"] = {"Subarray": args.multiplicity_cut}

    if args.output:
        config["Logging"] = {}
        config["Logging"]["model_directory"] = args.output
    # Create output directory if it doesn't exist already
    if not os.path.exists(config["Logging"]["model_directory"]):
        if "predict" in args.mode:
            raise ValueError(
                f"Invalid output directory '{config['Logging']['model_directory']}'. "
                "Must be a path to an existing directory in the predict mode."
            )
        os.makedirs(config["Logging"]["model_directory"])

    # Set the path to pretrained weights from the command line
    if args.pretrained_weights:
        config["Model"]["pretrained_weights"] = args.pretrained_weights
        config["Model"]["trainable_backbone"] = False

    # Overwrite the number of epochs, batch size and random seed in the config file
    if args.num_epochs:
        if "Training" not in config:
            config["Training"] = {}
        config["Training"]["num_epochs"] = args.num_epochs
    if args.batch_size:
        if "Input" not in config:
            config["Input"] = {}
        config["Input"]["batch_size_per_worker"] = args.batch_size
    if args.random_seed:
        if 1000 <= args.random_seed <= 9999:
            config["Data"]["seed"] = args.random_seed
            config["Logging"]["add_seed"] = True
        else:
            raise ValueError(
                "Random seed: '{}'. "
                "Must be 4 digit integer!".format(args.random_seed)
            )
    random_seed = config["Data"].get("seed", 1234)

    if "train" in args.mode:

        # Shuffle the data in train mode as default
        if "shuffle" not in config["Data"]:
            config["Data"]["shuffle"] = True

        # Training file handling
        training_file_list = (
            f"{config['Logging']['model_directory']}/training_file_list.txt"
        )
        if args.input:
            for input in args.input:
                abs_file_dir = os.path.abspath(input)
                with open(training_file_list, "a") as file_list:
                    for pattern in args.pattern:
                        files = glob.glob(os.path.join(abs_file_dir, pattern))
                        if not files:
                            continue
                        for file in np.sort(files):
                            file_list.write(f"{file}\n")
            config["Data"]["file_list"] = training_file_list

        if "training_file_list.txt" in os.listdir(config["Logging"]["model_directory"]):
            config["Data"]["file_list"] = training_file_list

        config["Data"][
            "example_identifiers_file"
        ] = f"{config['Logging']['model_directory']}/example_identifiers_file.h5"

        run_model(config, mode="train", debug=args.debug, log_to_file=args.log_to_file)

    if "predict" in args.mode:
        if args.input:
            for input in args.input:
                abs_file_dir = os.path.abspath(input)
                for pattern in args.pattern:
                    files = glob.glob(os.path.join(abs_file_dir, pattern))
                    if not files:
                        continue
                    for file in files:

                        with open(args.config_file, "r") as config_file:
                            config = yaml.safe_load(config_file)

                        if args.reco:
                            config["Reco"] = args.reco
                        if args.cleaned_images:
                            config["Data"]["image_channels"] = [
                                "cleaned_" + channel
                                for channel in config["Data"]["image_channels"]
                            ]
                        if args.tel_types:
                            config["Data"]["selected_telescope_types"] = args.tel_types
                        if args.allowed_tels:
                            config["Data"]["selected_telescope_ids"] = args.allowed_tels
                        if parameter_selection:
                            config["Data"]["parameter_selection"] = parameter_selection
                        if args.multiplicity_cut:
                            config["Data"]["multiplicity_selection"] = {
                                "Subarray": args.multiplicity_cut
                            }
                        if args.batch_size:
                            if "Input" not in config:
                                config["Input"] = {}
                            config["Input"]["batch_size_per_worker"] = args.batch_size
                        if args.output:
                            config["Logging"] = {}
                            config["Logging"]["model_directory"] = args.output
                        if args.pretrained_weights:
                            config["Model"][
                                "pretrained_weights"
                            ] = args.pretrained_weights
                            config["Model"]["trainable_backbone"] = False

                        config["Data"]["shuffle"] = False
                        config["Data"]["seed"] = random_seed
                        if args.random_seed:
                            config["Logging"]["add_seed"] = True

                        if "Prediction" not in config:
                            config["Prediction"] = {}

                        prediction_file = (
                            file.split("/")[-1]
                            .replace("_S_", "_E_")
                            .replace("dl1", "dl2")
                            .replace("DL1", "DL2")
                        )
                        prediction_path = file.replace(f'{file.split("/")[-1]}', "")
                        if args.prediction_directory:
                            prediction_path = args.prediction_directory
                        prediction_file = f"{prediction_path}/{prediction_file}"
                        config["Prediction"]["prediction_file_lists"] = {
                            prediction_file: file
                        }
                        config["Prediction"]["prediction_file"] = prediction_file
                        run_model(
                            config,
                            mode="predict",
                            debug=args.debug,
                            log_to_file=args.log_to_file,
                        )
        else:
            for key in config["Prediction"]["prediction_file_lists"]:
                with open(args.config_file, "r") as config_file:
                    config = yaml.safe_load(config_file)
                if args.reco:
                    config["Reco"] = args.reco
                if args.cleaned_images:
                    config["Data"]["image_channels"] = [
                        "cleaned_" + channel
                        for channel in config["Data"]["image_channels"]
                    ]
                if args.tel_types:
                    config["Data"]["selected_telescope_types"] = args.tel_types
                if args.allowed_tels:
                    config["Data"]["selected_telescope_ids"] = args.allowed_tels
                if parameter_selection:
                    config["Data"]["parameter_selection"] = parameter_selection
                if args.multiplicity_cut:
                    config["Data"]["multiplicity_selection"] = {
                        "Subarray": args.multiplicity_cut
                    }
                if args.batch_size:
                    if "Input" not in config:
                        config["Input"] = {}
                    config["Input"]["batch_size_per_worker"] = args.batch_size
                if args.output:
                    config["Logging"] = {}
                    config["Logging"]["model_directory"] = args.output
                if args.pretrained_weights:
                    config["Model"]["pretrained_weights"] = args.pretrained_weights
                    config["Model"]["trainable_backbone"] = False

                config["Data"]["shuffle"] = False
                config["Data"]["seed"] = random_seed
                if args.random_seed:
                    config["Logging"]["add_seed"] = True
                if "Prediction" not in config:
                    config["Prediction"] = {}
                config["Prediction"]["prediction_file"] = key
                run_model(
                    config,
                    mode="predict",
                    debug=args.debug,
                    log_to_file=args.log_to_file,
                )


if __name__ == "__main__":
    main()
