import importlib
import logging
import os
import pkg_resources
import sys
import time
import pickle
import numpy as np
import pandas as pd
import tables
import yaml
import zlib
import math
import re
import matplotlib.pyplot as plt
import torch
from astropy.coordinates import SkyCoord, AltAz
import astropy.units as u
from ctapipe_io_lst.constants import LST1_LOCATION
from ctlearn.core.ctlearn_enum import Task, Mode
import time as time_
from ctapipe.io import read_table, write_table
from astropy.table import Table
crab_nebula_config = {
  "observation_mode": "wobble",
  "n_off_wobble": 1,
  "source_name": "Crab Nebula",
  "source_ra": 83.63308333,
  "source_dec": 22.0145
}
from astropy.time import Time
from ctapipe.coordinates import CameraFrame

def clip_alt(alt):
    """
    Make sure altitude is not larger than 90 deg (it happens in some MC files for zenith=0),
    to keep astropy happy
    """
    return np.clip(alt, -90.0 * u.deg, 90.0 * u.deg)

def radec_to_camera(sky_coordinate, obstime, pointing_alt, pointing_az, focal):
    """
    Coordinate transform from sky coordinate to camera coordinates (x, y) in distance

    Parameters
    ----------
    sky_coordinate: astropy.coordinates.sky_coordinate.SkyCoord
    obstime: astropy.time.Time
    pointing_alt: pointing altitude in angle unit
    pointing_az: pointing altitude in angle unit
    focal: astropy Quantity

    Returns
    -------
    camera frame: `astropy.coordinates.sky_coordinate.SkyCoord`
    """

    horizon_frame = AltAz(location=LST1_LOCATION, obstime=obstime)

    pointing_direction = SkyCoord(
        alt=clip_alt(pointing_alt), az=pointing_az, frame=horizon_frame,unit="deg"
    )

    camera_frame = CameraFrame(
        focal_length=focal,
        telescope_pointing=pointing_direction,
        obstime=obstime,
        location=LST1_LOCATION,
    )

    camera_pos = sky_coordinate.transform_to(camera_frame)

    return camera_pos

def get_expected_source_pos(data, data_type, config, effective_focal_length=29.30565 * u.m):

    # For real data
    if data_type == 'real_data':
        # source is always at the ceter of camera for ON mode
        if config.get('observation_mode') == 'on':
            expected_src_pos_x_m = np.zeros(len(data))
            expected_src_pos_y_m = np.zeros(len(data))

        # compute source position in camera coordinate event by event for wobble mode
        elif config.get('observation_mode') == 'wobble':

            if 'source_name' in config:
                source_coord = SkyCoord.from_name(config.get('source_name'))
            elif 'source_ra' and 'source_dec' in config:
                source_coord = SkyCoord(config.get('source_ra'), config.get('source_dec'), frame="icrs", unit="deg")
            else:
                raise KeyError(
                    'source position (`source_name` or `source_ra` & `source_dec`) is not defined in a config file for source-dependent analysis.'
                )

            time = data['dragon_time']
            obstime = Time(time, scale='utc', format='unix')
            # pointing_alt = u.Quantity(data['alt_tel'], u.rad, copy=False)
            # pointing_az = u.Quantity(data['az_tel'], u.rad, copy=False)
            pointing_alt_deg = data['pointing_alt']*u.deg 
            pointing_az_deg = data['pointing_az']*u.deg

            pointing_alt_rad = pointing_alt_deg.to(u.rad)
            pointing_az_rad = pointing_az_deg.to(u.rad)

            source_pos = radec_to_camera(source_coord, obstime, pointing_alt_deg, pointing_az_deg, effective_focal_length)

            expected_src_pos_x_m = source_pos.x.to_value(u.m)
            expected_src_pos_y_m = source_pos.y.to_value(u.m)
            
        else:
            raise KeyError(
                '`observation_mode` is not defined in a config file for source-dependent analysis. It should be `on` or `wobble`'
            )

    return expected_src_pos_x_m, expected_src_pos_y_m

def sky_to_camera(alt, az, focal, pointing_alt, pointing_az, horizon_frame):
    """
    Coordinate transform from aky position (alt, az) (in angles)
    to camera coordinates (x, y) in distance.

    Parameters
    ----------
    alt: astropy Quantity
    az: astropy Quantity
    focal: astropy Quantity
    pointing_alt: pointing altitude in angle unit
    pointing_az: pointing altitude in angle unit

    Returns
    -------
    camera frame: `astropy.coordinates.sky_coordinate.SkyCoord`
    """
    pointing_direction = SkyCoord(
        alt=clip_alt(pointing_alt), az=pointing_az, frame=horizon_frame
    )

    camera_frame = CameraFrame(
        focal_length=focal, telescope_pointing=pointing_direction
    )

    event_direction = SkyCoord(alt=clip_alt(alt), az=az, frame=horizon_frame)

    camera_pos = event_direction.transform_to(camera_frame)

    return camera_pos
#-------------------------------------------------------------------------------------------------------------------
def reco_src_sky_to_camera(data, effective_focal_length=29.30565 * u.m):

    time = data['dragon_time']
    obstime = Time(time, scale='utc', format='unix')
    # horizon_frame = AltAz(location=LST1_LOCATION, obstime=obstime)


    # time = data['utc_time']
    # time_utc = Time(time, format="mjd", scale="tai")
    # obstime = time_utc.utc

    alt_deg = u.Quantity(data['alt'], u.deg, copy=False)
    az_deg = u.Quantity(data['az'], u.deg, copy=False)
    
    horizon_frame = AltAz(location=LST1_LOCATION, obstime=obstime)

    tel_alt_deg = u.Quantity(data['tel_pointing_alt'], u.deg, copy=False)
    tel_az_deg = u.Quantity(data['tel_pointing_az'], u.deg, copy=False)

    source_pos_in_camera = sky_to_camera(
        alt_deg,
        az_deg,
        effective_focal_length,
        tel_alt_deg,
        tel_az_deg,
        horizon_frame=horizon_frame,
    )
    expected_src_pos_x_m = source_pos_in_camera.x.to_value(u.m)
    expected_src_pos_y_m = source_pos_in_camera.y.to_value(u.m)
    
    # -----------------------------------------------------------------
    # TODO: Remove this. It is the equivalent to the code above 
    # tel_alt_rad = tel_alt_deg.to(u.rad)
    # tel_az_rad = tel_az_deg.to(u.rad)
    # time = data['utc_time']
    # time_utc = Time(time, format="mjd", scale="tai")
    # time_utc = time_utc.utc
    # telescope_pointing = SkyCoord(alt=tel_alt_rad, az=tel_az_rad,
    #                     frame=AltAz(obstime=time_utc,
    #                                 location=LST1_LOCATION))
    
    # source_pos = SkyCoord(
    #     alt=clip_alt(alt_deg), az=az_deg, frame=horizon_frame
    # )
 
    # # CameraFrame is terribly slow without the erfa interpolator below...
    # # with erfa_astrom.set(ErfaAstromInterpolator(5 * u.min)):
    # camera_frame = CameraFrame(focal_length=effective_focal_length,
    #                         telescope_pointing=telescope_pointing,
    #                         location=LST1_LOCATION, obstime=time_utc)
        
    # source_pos_camera = source_pos.transform_to(camera_frame)    

    return expected_src_pos_x_m, expected_src_pos_y_m
#-------------------------------------------------------------------------------------------------------------------
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
        "test_validation_gamma_proton": None
    },
    "run_details": {
        "mode": None,
        "task": None,
        "test_type": None,
        "experiment_number": None
    },
    "cut-off": {
        "leakage_intensity": None,
        "intensity": None
    },
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
        }
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
        "save_k": None
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
        "max_trans": None
    },
    "normalization": {
        "use_clean": None,
        "type_mu": None,
        "type_sigma": None,
        "dir_mu": None,
        "dir_sigma": None,
        "energy_mu": None,
        "energy_sigma": None
    },
    "dataset": {
        "num_workers": None,
        "pin_memory": None,
        "persistent_workers": None
    },
    "arch": {
        "device": None,
        "precision_type": None,
        "precision_energy": None,
        "precision_direction": None,
        "devices": None,
        "strategy": None
    },
}
#-------------------------------------------------------------------------------------------------------------------
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
                raise KeyError(f"Expected a dictionary for key: {key}, but got: {type(config[key])}")
            sanity_check(config[key], substructure)

#-------------------------------------------------------------------------------------------------------------------
def extract_loss_value(file_path):
    # Use regular expression to find the pattern
    match = re.search(r'_([0-9]+\.[0-9]+)\.pth$', file_path)
    if match:
        return match.group(1)  # Return the whole match
    else:
        return None
#-------------------------------------------------------------------------------------------------------------------
def convert_ascii_list_to_string(ascii_values, padding_value=0):
    # Filter out padding values and convert each ASCII value to its corresponding character
    characters = [chr(ascii_val)
                    for ascii_val in ascii_values if ascii_val != padding_value]

    # Join all characters to form the string
    return ''.join(characters)
#-------------------------------------------------------------------------------------------------------------------
def extract_accuracy_from_filename(filename):
    # Use a regular expression to find numbers that might be in the format of floating point numbers
    match = re.search(r"(\d+\.\d+)", filename)
    if match:
        # Return the first occurrence of a floating point number as a float
        return float(match.group(1))
    else:
        # If no matching number format is found, handle it appropriately
        return None
#-------------------------------------------------------------------------------------------------------------------
def find_bin_index(value, bins):
    index = np.digitize([value], bins) - 1
    index = max(0, min(index[0], len(bins) - 1))  # Ensure the index is within the valid range
    bin_value = bins[index]
    return index, bin_value
#-------------------------------------------------------------------------------------------------------------------
def get_bin_value(index:int, bins):
    # Ensure the index is within the valid range
    index = max(0, min(index, len(bins) - 1))
    # Return the bin value associated with the index
    bin_value = bins[index]
    return bin_value
#-------------------------------------------------------------------------------------------------------------------
def get_bin_value(indices:np.array, bins):
    # Ensure indices are within the valid range
    indices = np.clip(indices, 0, len(bins) - 1)
    # Return the bin values associated with the indices
    return np.array(bins)[indices]
#-------------------------------------------------------------------------------------------------------------------
def get_bin_value(indices:torch.tensor, bins):
    # Convert bins to a tensor if they aren't already
    # bins_tensor = torch.tensor(bins)
    # Ensure indices are within the valid range
    indices = torch.clamp(indices, 0, len(bins) - 1)
    
    # Return the bin values associated with the indices
    return bins[indices]

#-------------------------------------------------------------------------------------------------------------------
def decompress_data(compressed_data, data_shape, data_type=np.float16):

    decompressed_data = zlib.decompress(compressed_data)
    restored_array = np.frombuffer(decompressed_data, dtype=data_type)
    restored_array_reshaped = np.reshape(restored_array, data_shape)

    return restored_array_reshaped
#-------------------------------------------------------------------------------------------------------------------
def compress_data(data):
    # Convert to bytes
    bytes_data = data.tobytes()

    # Array Compression
    compressed_data = zlib.compress(bytes_data)
    return compressed_data
#-------------------------------------------------------------------------------------------------------------------
def create_experiment_folder(prefix="run_", next_number=None):

    
    """
    Create the next folder within the specified directory with a given prefix.
    If next_number is not specified, automatically find the next available number.

    :param run_directory: The directory where folders are managed.
    :param prefix: Prefix used for folders.
    :param next_number: Optional. Specify the number to be used for the new folder.
    """
    run_directory="./run"

    # Ensure the 'run' directory exists
    if not os.path.exists(run_directory):
        os.makedirs(run_directory)
        print(f"Directory '{run_directory}' created.")

    if next_number is None:
        # List all subdirectories in the 'run' directory
        folders = [f for f in os.listdir(run_directory) if os.path.isdir(os.path.join(run_directory, f))]
        # Filter folders that match the prefix pattern and end with a digit
        matching_folders = [f for f in folders if f.startswith(prefix) and f[len(prefix):].isdigit()]

        # Find the highest number and calculate the next one
        if matching_folders:
            highest_number = max(int(folder[len(prefix):]) for folder in matching_folders)
            next_number = highest_number + 1
        else:
            next_number = 0

    # Create the new folder with the next number
    new_folder_name = f"{prefix}{next_number}"
    new_folder_path = os.path.join(run_directory, new_folder_name)
    new_folder_path+="/"
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        print(f"New folder created: {new_folder_path}")
    return new_folder_path
#------------------------------------------------------------------------------------------------------------
def cartesian_to_alt_az(cartesian):
    x = cartesian[0]
    y = cartesian[1]
    z = cartesian[2]
    # Calculate the radius
    r = math.sqrt(x**2 + y**2 + z**2)

    # Calculate altitude in radians
    if r == 0:  # Avoid division by zero
        altitude_rad = 0
    else:
        altitude_rad = math.asin(z / r)

    # Calculate azimuth in radians
    azimuth_rad = math.atan2(y, x)

    # Adjust azimuth to be within the range [0, 2*pi)
    # if azimuth_rad < 0:
    #     azimuth_rad += 2 * math.pi

    return np.array([altitude_rad, azimuth_rad, r])
#------------------------------------------------------------------------------------------------------------
def alt_az_to_cartesian(altitude_rad, azimuth_rad, r=1):

    # Estimate Cartesian coordinates
    x = r * math.cos(altitude_rad) * math.cos(azimuth_rad)
    y = r * math.cos(altitude_rad) * math.sin(azimuth_rad)
    z = r * math.sin(altitude_rad)

    return np.array([x, y, z])
#------------------------------------------------------------------------------------------------------------
def load_pickle(pickle_file):
    with open(pickle_file, 'rb') as file:
        print(f"Loading pickle file :{pickle_file}")
        return pickle.load(file)
    
#------------------------------------------------------------------------------------------------------------
def create_key_value_array(data_dict, id):
  """
  Creates an array of key-value pairs from a dictionary where each key has a list of values.

  Args:
      data_dict (dict): The dictionary containing keys with lists of values.

  Returns:
      list: An array of key-value pairs, where each pair is a tuple containing the key and one value.
  """

  key_value_array = []
  for key, value_list in data_dict.items():
    # Handle cases where a key has an empty list:
    if value_list.size==0:
      continue  # Skip keys with empty lists

    # Choose the first value from the list:
    value = value_list[id]

    # Create a tuple (key, value) and append it to the array
    key_value_array.append(value)

  return key_value_array

#------------------------------------------------------------------------------------------------------------
def decompress_data(compressed_data, data_shape, data_type=np.float16):

    decompressed_data = zlib.decompress(compressed_data)
    restored_array = np.frombuffer(decompressed_data, dtype=data_type)
    restored_array_reshaped = np.reshape(restored_array, data_shape)

    return restored_array_reshaped
#------------------------------------------------------------------------------------------------------------
def setup_logging(config, log_dir, debug, log_to_file):

    # Log configuration to a text file in the log dir
    time_str = time.strftime("%Y%m%d_%H%M%S")
    config_filename = os.path.join(log_dir, time_str + "_config.yml")
    with open(config_filename, "w") as outfile:
        ctlearn_version = pkg_resources.get_distribution("ctlearn").version
        tensorflow_version = pkg_resources.get_distribution("tensorflow").version
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
#------------------------------------------------------------------------------------------------------------
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
                "Must be list or path to file".format(config["Data"]["file_list"])
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
        if not isinstance(config["Data"]["file_list"], list):
            raise ValueError(
                "Invalid prediction file list '{}'. "
                "Must be list or path to file".format(file_list)
            )

    mc_file = True
    with tables.open_file(config["Data"]["file_list"][0], mode="r") as f:
        if "CTA PRODUCT DATA MODEL NAME" in f.root._v_attrs:
            data_format = "stage1"
        elif "dl1_data_handler_version" in f.root._v_attrs:
            data_format = "dl1dh"
        else:
            raise ValueError(
                "Data format is not implemented in the DL1DH reader. Available data formats are 'stage1' and 'dl1dh'."
            )
        if data_format == "dl1dh" and "source_name" in f.root._v_attrs:
            mc_file = False

    allow_overwrite = config["Data"].get("allow_overwrite", True)
    if "allow_overwrite" in config["Data"]:
        del config["Data"]["allow_overwrite"]

    selected_telescope_types = config["Data"]["selected_telescope_types"]
    camera_types = [tel_type.split("_")[-1] for tel_type in selected_telescope_types]

    tasks = config["Reco"]
    transformations = []
    event_info = []
    if data_format == "dl1dh":
        if "parameter_list" not in config["Data"] and mode == "predict":
            config["Data"]["parameter_list"] = [
                "hillas_intensity",
                "log_hillas_intensity",
                "hillas_x",
                "hillas_y",
                "hillas_r",
                "hillas_phi",
                "hillas_length",
                "hillas_width",
                "hillas_psi",
                "hillas_skewness",
                "leakage_intensity_width_1",
                "leakage_intensity_width_2",
                "morphology_num_islands",
                "impact",
                "log_impact",
                "maxheight",
                "log_maxheight",
                "cherenkovdensity",
                "log_hillasintensity_over_cherenkovdensity",
                "cherenkovradius",
                "impact_over_cherenkovradius",
                "p1grad",
                "sqrt_p1grad_p1grad",
            ]
        # Parse list of event selection filters
        event_selection = {}
        for s in config["Data"].get("event_selection", {}):
            s = {"module": "dl1_data_handler.filters", **s}
            filter_fn, filter_params = load_from_module(**s)
            event_selection[filter_fn] = filter_params
        config["Data"]["event_selection"] = event_selection

        # Parse list of image selection filters
        image_selection = {}
        for s in config["Data"].get("image_selection", {}):
            s = {"module": "dl1_data_handler.filters", **s}
            filter_fn, filter_params = load_from_module(**s)
            image_selection[filter_fn] = filter_params
        config["Data"]["image_selection"] = image_selection

        if "direction" in tasks:
            event_info.append("src_pos_cam_x")
            event_info.append("src_pos_cam_y")
            transformations.append(
                {
                    "name": "AltAz",
                    "args": {
                        "alt_col_name": "src_pos_cam_x",
                        "az_col_name": "src_pos_cam_y",
                        "deg2rad": False,
                    },
                }
            )
    else:
        if "parameter_list" not in config["Data"] and mode == "predict":
            config["Data"]["parameter_list"] = [
                "hillas_intensity",
                "hillas_fov_lon",
                "hillas_fov_lat",
                "hillas_r",
                "hillas_phi",
                "hillas_length",
                "hillas_length_uncertainty",
                "hillas_width",
                "hillas_width_uncertainty",
                "hillas_psi",
                "hillas_skewness",
                "hillas_kurtosis",
                "timing_slope",
                "timing_intercept",
                "timing_deviation",
                "leakage_pixels_width_1",
                "leakage_pixels_width_2",
                "leakage_intensity_width_1",
                "leakage_intensity_width_2",
                "concentration_cog",
                "concentration_core",
                "concentration_pixel",
                "morphology_n_pixels",
                "morphology_n_islands",
                "morphology_n_medium_islands",
                "morphology_n_large_islands",
                "intensity_max",
                "intensity_min",
                "intensity_mean",
                "intensity_std",
                "intensity_skewness",
                "intensity_kurtosis",
                "peak_time_max",
                "peak_time_min",
                "peak_time_mean",
                "peak_time_std",
                "peak_time_skewness",
                "peak_time_kurtosis",
            ]
        if "direction" in tasks:
            event_info.append("true_alt")
            event_info.append("true_az")
            transformations.append({"name": "DeltaAltAz_fix_subarray"})

    if "particletype" in tasks:
        event_info.append("true_shower_primary_id")

    if "energy" in tasks:
        if mc_file:
            event_info.append("true_energy")
        transformations.append({"name": "MCEnergy"})

    concat_telescopes = config["Input"].get("concat_telescopes", False)
    if config["Data"]["mode"] == "stereo" and not concat_telescopes:
        for tel_desc in selected_telescope_types:
            transformations.append(
                {
                    "name": "SortTelescopes",
                    "args": {"sorting": "size", "tel_desc": f"{tel_desc}"},
                }
            )

    # Convert interpolation image shapes from lists to tuples, if present
    if "interpolation_image_shape" in config["Data"].get("mapping_settings", {}):
        config["Data"]["mapping_settings"]["interpolation_image_shape"] = {
            k: tuple(l)
            for k, l in config["Data"]["mapping_settings"][
                "interpolation_image_shape"
            ].items()
        }

    if allow_overwrite:
        config["Data"]["event_info"] = event_info
        config["Data"]["mapping_settings"]["camera_types"] = camera_types
    else:
        transformations = config["Data"].get("transforms", {})

    transforms = []
    # Parse list of Transforms
    for t in transformations:
        t = {"module": "dl1_data_handler.transforms", **t}
        transform, args = load_from_module(**t)
        transforms.append(transform(**args))
    config["Data"]["transforms"] = transforms

    # Possibly add additional info to load if predicting to write later
    if mode == "predict":

        if "Prediction" not in config:
            config["Prediction"] = {}
        if "event_info" not in config["Data"]:
            config["Data"]["event_info"] = []
        config["Data"]["event_info"].extend(["event_id", "obs_id"])
        if data_format == "dl1dh" and not mc_file:
            config["Data"]["event_info"].extend(["mjd", "milli_sec", "nano_sec"])

    return config["Data"], data_format
#------------------------------------------------------------------------------------------------------------
def load_from_module(name, module, path=None, args=None):
    if path is not None and path not in sys.path:
        sys.path.append(path)
    mod = importlib.import_module(module)
    fn = getattr(mod, name)
    params = args if args is not None else {}
    return fn, params
#------------------------------------------------------------------------------------------------------------
def recover_alt_az(fix_pointing,alt_off, az_off):
    
    az_off_deg = u.Quantity(az_off, unit=u.rad).to(u.deg).value
    alt_off_deg = u.Quantity(alt_off, unit=u.rad).to(u.deg).value
    
    reco_direction = fix_pointing.spherical_offsets_by(
                             [az_off_deg] * u.deg,
                             [alt_off_deg] * u.deg
                             )
    
    # Clamp altitude offset to avoid exceeding valid range
    # alt_off_deg = np.clip(alt_off_deg, -90, 90)
    
    return reco_direction
#------------------------------------------------------------------------------------------------------------
def write_output(h5file, data, predictions, labels, task:Task, mode:Mode):
    prediction_dir = h5file.replace(f'{h5file.split("/")[-1]}', "")
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    # Store dl2 data
    reco = {}
    if os.path.isfile(h5file):
        with pd.HDFStore(h5file, mode="r") as file:
            h5file_keys = list(file.keys())
            if f"/dl2/reco" in h5file_keys:
                reco = pd.read_hdf(file, key=f"/dl2/reco")


    reco["event_id"] = np.array(data["event_id"])
    reco["obs_id"] = np.array(data["obs_id"])



    if task ==Task.type:

        for n, name in enumerate(data["class_names"]):
            reco[name + "ness"] = np.array(predictions["type"][:, n])
        # reco["type_feature_vector"] =  np.array(predictions["type_feature_vector"]) 
        reco["reco_type"]=  np.array(predictions["type_class"])*101
        
    if mode != Mode.observation:#"observation":
        if data["energy_unit"] == "log(TeV)":
            reco["true_energy"] = np.power(10, labels["true_energy"])
        else:
            reco["true_energy"] = labels["true_energy"]


    if task == Task.energy:        
        # if data["energy_unit"] == "log(TeV)" or np.min(predictions["energy"]) < 0.0:
        #     reco["reco_energy"] = np.power(10, predictions["energy"][:, 0])
        #     reco["log_reco_energy"] =predictions["energy"][:, 0]
        # else:
        reco["reco_energy"] = np.power(10, predictions["energy"][:, 0])
        reco["log_reco_energy"] =predictions["energy"][:, 0]


    if mode != Mode.observation:#"observation":
        reco["true_alt"] = np.float32(np.rad2deg(labels["true_alt_az"][:,0]))
        reco["true_az"] =  np.float32(np.rad2deg(labels["true_alt_az"][:,1]))

    if task==Task.direction:             
        # reco["reco_alt"] = np.array(predictions["direction"][:, 1])  
        # reco["reco_az"] = np.array(predictions["direction"][:, 0])  

        reco_az, reco_alt = [], []
        reco_az_off, reco_alt_off = [], []
        reco_src_x, reco_src_y = [], []
        dragon_time = []
        utc_time = []

        data_type = 'real_data'
        if mode == Mode.observation: 
            # reco["reco_src_x"] = []
            # reco["reco_src_y"] = []
            reco["src_x"] = data["pointing"]["src_x"]
            reco["src_y"] = data["pointing"]["src_y"]
            
        reco_data = {}
        if mode != Mode.observation:#"observation":
            pointing_alt = np.full(len(data["obs_id"]), data["pointing"]["pointing_alt"])
            pointing_az = np.full(len(data["obs_id"]), data["pointing"]["pointing_az"])

            reco["alt_tel"] = pointing_alt
            reco["az_tel"] = pointing_az
        else:
            pointing_alt = data["pointing"]["pointing_alt"]
            pointing_az =  data["pointing"]["pointing_az"]

            reco["alt_tel"] = pointing_alt
            reco["az_tel"] = pointing_az

        # fix_pointing_t = SkyCoord(
        #     pointing_az * u.deg,
        #     pointing_alt * u.deg,
        #     frame="altaz",
        #     unit="deg",
        # )

        horizon_frame="altaz"

        if mode== Mode.observation:
            time = data["pointing"]["dragon_time"]
            obstime = Time(time, scale='utc', format='unix')
            horizon_frame = AltAz(location=LST1_LOCATION, obstime=obstime)   
                     
            # obstime_lst = Time("2018-11-01T02:00")
            # horizon_frame_lst = AltAz(location=LST1_LOCATION, obstime=obstime_lst)

            # fix_pointing_lst = SkyCoord(
            #     az=pointing_az * u.deg,
            #     alt=pointing_alt * u.deg,
            #     frame=horizon_frame_lst,
            #     unit="deg",
            # )

        fix_pointing = SkyCoord(
            az=pointing_az * u.deg,
            alt=pointing_alt * u.deg,
            frame=horizon_frame,
            unit="deg",
        )
        # ----------------------------------------------------------------------------------------------------------------------------------------
        # start = time_.time()
        sigma_az=[]
        sigma_alt=[]
        if len(predictions["direction"])>0:
            az_off_ = predictions["direction"][:, 0]
            alt_off_ = predictions["direction"][:, 1]
            
        else:
            az_off_ = predictions["direction_mu"][:, 0]
            alt_off_ = predictions["direction_mu"][:, 1]           
            sigma_az = predictions["direction_sigma"][:, 0]
            sigma_alt = predictions["direction_sigma"][:, 1]

        reco_direction_ = recover_alt_az(fix_pointing,alt_off_,az_off_)
        reco_az_item_=reco_direction_.az.to_value(u.deg)[0]
        reco_alt_item_=reco_direction_.alt.to_value(u.deg)[0]

        reco_az=reco_az_item_
        reco_alt = reco_alt_item_
        reco_az_off = az_off_
        reco_alt_off = alt_off_

        reco_data_ = {}
        if mode == Mode.observation:
            reco_data_["alt"] = reco_alt
            reco_data_["az"] = reco_az
            reco_data_["tel_pointing_alt"] = fix_pointing.alt.value 
            reco_data_["tel_pointing_az"] = fix_pointing.az.value 

            reco_data_["dragon_time"] = data["pointing"]["dragon_time"]
            reco_data_["utc_time"] = data["pointing"]["utc_time"]

            reco_src_ = reco_src_sky_to_camera(reco_data_,effective_focal_length=data["effective_focal_length"])

            reco_src_x = reco_src_[0]
            reco_src_y = reco_src_[1]
            dragon_time = data["pointing"]["dragon_time"]
            # New fix
            # dragon_time = Time(dragon_time, format='unix_tai')

            utc_time = data["pointing"]["utc_time"]

        # end = time_.time()
        # print("Optimized: ", end - start)
        # ----------------------------------------------------------------------------------------------------------------------------------------
        if mode == Mode.observation:
            reco["reco_src_x"] = reco_src_x
            reco["reco_src_y"] = reco_src_y
            reco["dragon_time"] = dragon_time
            reco["utc_time"] = utc_time

        reco["reco_alt"] = reco_alt
        reco["reco_az"] = reco_az
        reco["reco_alt_off"] = reco_alt_off
        reco["reco_az_off"] = reco_az_off
        reco["sigma_alt"]= sigma_alt
        reco["sigma_az"]= sigma_az
        
    del predictions
    import gc
    gc.collect()
    
    # Convertir reco a Astropy Table
    reco_table = Table()
    for key, val in reco.items():
        reco_table[key] = np.array(val)

    if data["include_nsb_patches"] is None:
        pd.DataFrame(data=reco).to_hdf(h5file, key=f"/dl2/reco", mode="a", format="table")
        # write_table(
        #     reco_table,
        #     h5file,
        #     f"/dl2/reco",
        #     overwrite=True,
        # )
    else:
        # write_table(
        #     reco_table,
        #     h5file,
        #     f"/trigger/reco",
        #     overwrite=True,
        # )
        pd.DataFrame(data=reco).to_hdf(h5file, key=f"/trigger/reco", mode="a", format="table")



    # Guardar usando ctapipe.io.write_table



    # Store the simulation information for pyirf
    if data["simulation_info"] and data["include_nsb_patches"] != "all":
        pd.DataFrame(data=data["simulation_info"] , index=[0]).to_hdf(
            h5file, key=f"/info/mc_header", mode="a", format="table")


    # Store the selected Hillas parameters (dl1b)
    if data["parameter_names"] and data["include_nsb_patches"] != "all":
        tel_counter = 0
        if data["mode"]== "mono":
            tel_type = list(data["selected_telescopes"].keys())[0]
            tel_ids = "tel"
            for tel_id in data["selected_telescopes"][tel_type]:
                tel_ids += f"_{tel_id}"
            parameters = {}
            for p, parameter in enumerate(data["parameter_names"]):
                parameter_list = np.array(data["parameter_data"])[:,p]

                parameters[parameter] = parameter_list
            pd.DataFrame(data=parameters).to_hdf(
                h5file, key=f"/dl1b/{tel_type}/{tel_ids}", mode="a", format="table")
        else:
            for tel_type in data["selected_telescopes"]:
                for t, tel_id in enumerate(data["selected_telescopes"][tel_type]):
                    parameters = {}
                    for p, parameter in enumerate(data["parameter_list"]):
                        parameter_list = np.array(data["parameter_list"])[:, tel_counter + t, p]
       
                        parameters[parameter] = parameter_list
                    pd.DataFrame(data=parameters).to_hdf(
                        h5file, key=f"/dl1b/{tel_type}/tel_{tel_id}", mode="a", format="table")
                tel_counter += len(data["selected_telescopes"][tel_type])

    print(f"File saved: {h5file}")