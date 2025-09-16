from ctlearn.core.ctlearn_enum import Task
from typing import List
import os
import yaml

expected_structure = {
    "data": {
        "train_gamma_proton": None,
        "validation_gamma_proton": None,
        "train_gamma": None,
        "validation_gamma": None,
        "test_gamma": None,
        "test_proton": None,
        "test_electron": None,
        "test_validation_gamma": None,
        "test_validation_gamma_proton": None,
    },
    "run_details": {
        "mode": None,
        "task": None,
        "test_type": None,
        "experiment_number": None,
    },
    "cut-off": {"leakage_intensity": None, "intensity": None},
    "model": {
        "model_type": {
            "model_name": None,
            "parameters": None,
        },
        "model_energy": {
            "model_name": None,
            "parameters": None,
        },
        "model_direction": {
            "model_name": None,
            "parameters": None,
        },
    },
    "hyp": {
        "epochs": None,
        "batches": None,
        "dynamic_batches": None,
        "optimizer": None,
        "momentum": None,
        "weight_decay": None,
        "learning_rate": None,
        "lrf": None,
        "start_epoch": None,
        "steps_epoch": None,
        "l2_lambda": None,
        "adam_epsilon": None,
        "gradient_clip_val": None,
        "save_k": None,
    },
    "augmentation": {
        "use_augmentation": None,
        "aug_prob": None,
        "rot_prob": None,
        "trans_prob": None,
        "flip_hor_prob": None,
        "flip_ver_prob": None,
        "mask_prob": None,
        "noise_prob": None,
        "max_rot": None,
        "max_trans": None,
    },
    "normalization": {
        "use_clean": None,
        "type_mu": None,
        "type_sigma": None,
        "dir_mu": None,
        "dir_sigma": None,
        "energy_mu": None,
        "energy_sigma": None,
    },
    "dataset": {"num_workers": None, "pin_memory": None, "persistent_workers": None},
    "arch": {
        "device": None,
        "precision_type": None,
        "precision_energy": None,
        "precision_direction": None,
        "devices": None,
        "strategy": None,
    },
}


# -------------------------------------------------------------------------------------------------------------------
# Sanity check function
def sanity_check(config, expected_structure):
    """
    Recursively checks if all required keys in the expected_structure exist in the config data.
    Raises a KeyError if a key is missing.
    """
    for key, substructure in expected_structure.items():
        if key not in config:
            raise KeyError(f"Missing key: {key}")

        # If the substructure is a dictionary, recursively check the subkeys
        if isinstance(substructure, dict):
            if not isinstance(config[key], dict):
                raise KeyError(
                    f"Expected a dictionary for key: {key}, but got: {type(config[key])}"
                )
            sanity_check(config[key], substructure)


# -------------------------------------------------------------------------------------------------------------------
def read_configuration(config_file_str="./config/training_config.yml"):
    parameters = None

    if os.path.exists(config_file_str):
        with open(config_file_str, "r") as config_file:
            parameters = yaml.safe_load(config_file)

    else:
        print("Configuration file not found.")

    return parameters


# -------------------------------------------------------------------------------------------------------------------
def create_experiment_folder(prefix="run_", next_number=None):
    """
    Create the next folder within the specified directory with a given prefix.
    If next_number is not specified, automatically find the next available number.

    :param run_directory: The directory where folders are managed.
    :param prefix: Prefix used for folders.
    :param next_number: Optional. Specify the number to be used for the new folder.
    """
    run_directory = "./run"

    # Ensure the 'run' directory exists
    if not os.path.exists(run_directory):
        os.makedirs(run_directory)
        print(f"Directory '{run_directory}' created.")

    if next_number is None:
        # List all subdirectories in the 'run' directory
        folders = [
            f
            for f in os.listdir(run_directory)
            if os.path.isdir(os.path.join(run_directory, f))
        ]
        # Filter folders that match the prefix pattern and end with a digit
        matching_folders = [
            f for f in folders if f.startswith(prefix) and f[len(prefix) :].isdigit()
        ]

        # Find the highest number and calculate the next one
        if matching_folders:
            highest_number = max(
                int(folder[len(prefix) :]) for folder in matching_folders
            )
            next_number = highest_number + 1
        else:
            next_number = 0

    # Create the new folder with the next number
    new_folder_name = f"{prefix}{next_number}"
    new_folder_path = os.path.join(run_directory, new_folder_name)
    new_folder_path += "/"
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        print(f"New folder created: {new_folder_path}")
    return new_folder_path


# -------------------------------------------------------------------------------------------------------------------
def str_list_to_enum_list(reco_tasks: List) -> List[Task]:

    tasks = []
    for task_str in reco_tasks:
        try:
            tasks.append(Task[task_str])
        except KeyError:
            print(f"'{task_str}' is not a valid enum type.")
    return tasks


# -------------------------------------------------------------------------------------------------------------------
def get_absolute_config_path(config_file_str="./ctlearn/tools/train/pytorch/config/training_config_iaa_neutron_training.yml") -> str:
    """
    Returns the absolute path of the configuration file.

    :param config_file_str: Relative or absolute path to the configuration file.
    :return: Absolute path of the configuration file.
    """
    return os.path.abspath(config_file_str)