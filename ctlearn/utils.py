import importlib
import logging
import os
import sys
import time

import numpy as np
import tables
import yaml

from importlib.metadata import version


def setup_logging(config, log_dir, debug, log_to_file):
    # Log configuration to a text file in the log dir
    time_str = time.strftime("%Y%m%d_%H%M%S")
    config_filename = os.path.join(log_dir, time_str + "_config.yml")
    with open(config_filename, "w") as outfile:
        ctlearn_version = version("ctlearn")
        tensorflow_version = version("tensorflow")
        outfile.write(
            "# Training performed with "
            "CTLearn version {} and TensorFlow version {}.\n".format(
                ctlearn_version, tensorflow_version
            )
        )
        yaml.dump(config, outfile, default_flow_style=False)

    # Set up logger
    logger = logging.getLogger()

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.handlers = []  # remove existing handlers from any previous runs
    if not log_to_file:
        handler = logging.StreamHandler()
    else:
        logging_filename = os.path.join(log_dir, time_str + "_logfile.log")
        handler = logging.FileHandler(logging_filename)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
    logger.addHandler(handler)

    return logger


def setup_DL1DataReader(config, mode):
    # Parse file list or prediction file list
    if mode in ["train", "load_only"]:
        if isinstance(config["Data"]["file_list"], str):
            data_files = []
            with open(config["Data"]["file_list"]) as f:
                for line in f:
                    line = line.strip()
                    if line and line[0] != "#":
                        data_files.append(line)
            config["Data"]["file_list"] = data_files
        if not isinstance(config["Data"]["file_list"], list):
            raise ValueError(
                "Invalid file list '{}'. "
                "Must be list or path to file or directory".format(
                    config["Data"]["file_list"]
                )
            )
    else:
        file_list = config["Prediction"]["prediction_file_lists"][
            config["Prediction"]["prediction_file"]
        ]
        if file_list.endswith(".txt"):
            data_files = []
            with open(file_list) as f:
                for line in f:
                    line = line.strip()
                    if line and line[0] != "#":
                        data_files.append(line)
            config["Data"]["file_list"] = data_files
        elif file_list.endswith(".h5"):
            config["Data"]["file_list"] = [file_list]

        if os.path.isdir(file_list):
            config["Data"]["file_list"] = np.sort(
                np.array(
                    [file_list + x for x in os.listdir(file_list) if x.endswith(".h5")]
                )
            ).tolist()

        if not isinstance(config["Data"]["file_list"], list):
            raise ValueError(
                "Invalid prediction file list '{}'. "
                "Must be list or path to file or directory".format(file_list)
            )

    dl1bparameter_names = None
    with tables.open_file(config["Data"]["file_list"][0], mode="r") as f:
        # Retrieve the name convention for the dl1b parameters
        first_tablename = next(
            f.root.dl1.event.telescope.parameters._f_iter_nodes()
        ).name
        dl1bparameter_names = f.root.dl1.event.telescope.parameters._f_get_child(
            f"{first_tablename}"
        ).colnames

    allow_overwrite = config["Data"].get("allow_overwrite", True)
    if "allow_overwrite" in config["Data"]:
        del config["Data"]["allow_overwrite"]

    selected_telescope_types = config["Data"]["selected_telescope_types"]
    camera_types = [tel_type.split("_")[-1] for tel_type in selected_telescope_types]

    if (
        "parameter_settings" not in config["Data"]
        and dl1bparameter_names is not None
        and mode == "predict"
    ):
        config["Data"]["parameter_settings"] = {"parameter_list": dl1bparameter_names}
   
    '''
    stack_telescope_images = config["Input"].get("stack_telescope_images", False)
    if config["Data"]["mode"] == "stereo" and not stack_telescope_images:
        for tel_desc in selected_telescope_types:
            transformations.append(
                {
                    "name": "SortTelescopes",
                    "args": {"sorting": "size", "tel_desc": f"{tel_desc}"},
                }
            )
    '''
    # Convert interpolation image shapes from lists to tuples, if present
    if "interpolation_image_shape" in config["Data"].get("mapping_settings", {}):
        config["Data"]["mapping_settings"]["interpolation_image_shape"] = {
            k: tuple(l)
            for k, l in config["Data"]["mapping_settings"][
                "interpolation_image_shape"
            ].items()
        }

    if allow_overwrite:
        config["Data"]["mapping_settings"]["camera_types"] = camera_types

    # Possibly add additional info to load if predicting to write later
    if mode == "predict":
        if "Prediction" not in config:
            config["Prediction"] = {}

    return config["Data"]

