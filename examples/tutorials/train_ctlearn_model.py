"""
Tutorial 2: Train CTLearn models
================================

Train models for CTAO image and waveform data with CTLearn. The image
examples use only the gamma-ray and proton ``00`` files; the ``01`` files
remain untouched for future evaluation tutorials.
The tutorial is intenionally kept simple and uses only a small fraction
of the available statistics to keep the training time manageable.
The resulting performance metrics are not representative of a production-level
analysis nor should they be compared directly with published results. The main
goal is to illustrate the analysis workflow and to become familiar with the
command-line tools and the DL2 data format. This tutorial can be modified from
the users to use larger datasets and more statistics to obtain meaningful results.

Each section explains a training task followed immediately by its literal
command-line example. Running this script executes every example in order.
"""

######################################################################
# Inspect the training environment and cached input data
# ------------------------------------------------------
#
# Run ``explore_public_data.py`` first to download and convert the shared
# tutorial data. The converted directory is keyed by the ctapipe version and
# the optional event limit, matching the cache layout used by that tutorial.

import importlib.metadata
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from ctapipe import __version__ as ctapipe_version

MAX_EVENTS = os.environ.get("CTLEARN_TUTORIAL_MAX_EVENTS")
if MAX_EVENTS is not None and int(MAX_EVENTS) < 1:
    raise ValueError("CTLEARN_TUTORIAL_MAX_EVENTS must be a positive integer")
MAX_EVENTS_SUFFIX = f"-max{MAX_EVENTS}" if MAX_EVENTS is not None else ""

TESTDATA_DIR = (
    Path(
        os.environ.get(
            "CTLEARN_TUTORIAL_DATA_DIR",
            str(Path.home() / ".cache" / "ctlearn" / "tutorial-data"),
        )
    )
    .expanduser()
    .resolve()
)
CONVERTED_DIR = TESTDATA_DIR / f"ctapipe-{ctapipe_version}{MAX_EVENTS_SUFFIX}"
GAMMA_FILE = CONVERTED_DIR / "gamma-diffuse_with_images_00.dl2.h5"
PROTON_FILE = CONVERTED_DIR / "proton_with_images_00.dl2.h5"
WAVEFORM_FILE = TESTDATA_DIR / f"gamma_prod5{MAX_EVENTS_SUFFIX}.r1.dl1.h5"

print("Training environment:")
for distribution in ("ctapipe", "ctlearn", "dl1-data-handler", "keras", "tensorflow"):
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = "not installed"
    print(f"  {distribution}: {version}")

missing_files = [path for path in (GAMMA_FILE, PROTON_FILE) if not path.is_file()]
if missing_files:
    missing = "\n".join(f"  - {path}" for path in missing_files)
    raise FileNotFoundError(
        "Cached stereo training data is missing. Run "
        "examples/tutorials/explore_public_data.py first. Missing files:\n"
        f"{missing}"
    )

print("\nCached input files:")
for path in (GAMMA_FILE, PROTON_FILE):
    print(f"  {path.name}: {path.stat().st_size / 1024**2:.1f} MiB")
print("Only the _00 image files are selected; the _01 files are reserved.")

######################################################################
# Training commands, configs, and outputs
# ---------------------------------------
#
# CTLearn provides ``ctlearn-train-keras-model`` and
# ``ctlearn-train-pytorch-model`` command-line tools. Use ``--help`` or
# ``--help-all`` to inspect their options. Each example has a Keras command
# followed immediately by the equivalent PyTorch command. In a notebook,
# prefix either command with ``!``.
#
# The three JSON configuration files shipped beside this tutorial are copies
# of the workshop examples:
#
# - ``docs/source/config/dl1dh_example_config.json`` selects LST/LSTCam images,
#   the image and relative peak-time channels, bilinear mapping, and an
#   example 50,000 p.e. intensity cut. It also specifies NectarCam waveform
#   settings.
# - ``docs/source/config/ctlearn_train_model_example_config.json`` sets the
#   reader, epoch and batch defaults, validation split, Adam optimizer and
#   learning rate, checkpointing, learning-rate reduction, random seed, and
#   early stopping.
# - ``docs/source/config/ctlearn_models_example_config.json`` defines the
#   prediction heads, attention settings, custom ``SingleCNN`` and ``ResNet``
#   architectures, and ``LoadedModel`` transfer-learning options.


def find_config_dir():
    roots = []
    script_path = globals().get("__file__")
    if script_path is not None:
        roots.append(Path(script_path).resolve().parents[2])
    roots.extend((Path.cwd(), *Path.cwd().parents))

    config_names = (
        "dl1dh_example_config.json",
        "ctlearn_train_model_example_config.json",
        "ctlearn_models_example_config.json",
    )
    for root in roots:
        config_dir = root / "docs" / "source" / "config"
        if all((config_dir / name).is_file() for name in config_names):
            return config_dir

    raise FileNotFoundError(
        "Could not locate tutorial configs under docs/source/config. "
        "Run this tutorial from the CTLearn repository or build it with "
        "Sphinx-Gallery."
    )


CONFIG_DIR = find_config_dir()
DL1DH_CONFIG = CONFIG_DIR / "dl1dh_example_config.json"
TRAINING_CONFIG = CONFIG_DIR / "ctlearn_train_model_example_config.json"
MODEL_CONFIG = CONFIG_DIR / "ctlearn_models_example_config.json"
for config_path in (DL1DH_CONFIG, TRAINING_CONFIG, MODEL_CONFIG):
    if not config_path.is_file():
        raise FileNotFoundError(f"Tutorial configuration is missing: {config_path}")

OUTPUT_DIR = (
    Path(
        os.environ.get(
            "CTLEARN_TUTORIAL_OUTPUT_DIR",
            str(Path.home() / ".cache" / "ctlearn" / "tutorial-training"),
        )
    )
    .expanduser()
    .resolve()
)
TRAINING_OUTPUTS = (
    "keras-energy",
    "pytorch-energy",
    "keras-cameradirection",
    "pytorch-cameradirection",
    "keras-skydirection",
    "pytorch-skydirection",
    "keras-classification",
    "pytorch-classification",
    "keras-singlecnn-energy",
    "pytorch-singlecnn-energy",
    "keras-resnet-energy",
    "pytorch-resnet-energy",
    "keras-transfer-learning-energy",
    "pytorch-transfer-learning-energy",
    "keras-stereo-energy",
    "pytorch-stereo-energy",
    "keras-waveform-energy",
    "pytorch-waveform-energy",
)
existing_outputs = [
    OUTPUT_DIR / name for name in TRAINING_OUTPUTS if (OUTPUT_DIR / name).exists()
]
if existing_outputs:
    existing = "\n".join(f"  - {path}" for path in existing_outputs)
    raise FileExistsError(
        "Training output directories already exist. Set "
        "CTLEARN_TUTORIAL_OUTPUT_DIR to a new directory before running all "
        f"examples:\n{existing}"
    )

os.environ["CTLEARN_TUTORIAL_DATA_DIR"] = str(TESTDATA_DIR)
os.environ["CTLEARN_TUTORIAL_CONVERTED_DIR"] = str(CONVERTED_DIR)
os.environ["CTLEARN_TUTORIAL_CONFIG_DIR"] = str(CONFIG_DIR.resolve())
os.environ["CTLEARN_TUTORIAL_OUTPUT_DIR"] = str(OUTPUT_DIR)
os.environ["CTLEARN_TUTORIAL_MAX_EVENTS_SUFFIX"] = MAX_EVENTS_SUFFIX

print(f"\nTraining outputs: {OUTPUT_DIR}")


def run_command_line(command):
    print(f"Running command: $ {command.strip()}")
    result = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(result.stdout, end="")
    result.check_returncode()


######################################################################
# Part A2: Monoscopic CNN-based model training
# ===========================================
#
# We first train a basic monoscopic CNN-based model using the default CTLearn
# model configuration. The model uses a **ThinResNet** backbone for feature
# extraction and a standard MLP head for the task-specific prediction. We
# consider three different reconstruction tasks, each trained separately:

######################################################################
# 1. Primary particle energy regression
# -------------------------------------
#
# The model reconstructs the energy of the primary particle as a continuous
# value. Internally, CTLearn performs the regression on the energy in TeV and
# on a logarithmic scale, since the primary-particle energy spans a very wide
# range. Working in logarithmic space has been found to be more efficient for
# the regression and is therefore a fixed, non-configurable aspect of the
# energy reconstruction. During inference, the prediction tool converts the
# predicted values back from logarithmic space to the original linear energy
# scale in TeV.

run_command_line(
    """ctlearn-train-keras-model --verbose \\
        --signal "$CTLEARN_TUTORIAL_CONVERTED_DIR" \\
        --pattern-signal "gamma-diffuse_with_images_00.dl2.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/keras-energy" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/ctlearn_train_model_example_config.json" \\
        --reco energy \\
        --DLImageReader.focal_length_choice EQUIVALENT \\
        --TrainCTLearnModel.n_epochs 5 \\
        --TrainCTLearnModel.batch_size 4""",
)

run_command_line(
    """ctlearn-train-pytorch-model --verbose \\
        --signal "$CTLEARN_TUTORIAL_CONVERTED_DIR" \\
        --pattern-signal "gamma-diffuse_with_images_00.dl2.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/pytorch-energy" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/ctlearn_train_model_example_config.json" \\
        --reco energy \\
        --DLImageReader.focal_length_choice EQUIVALENT \\
        --TrainCTLearnModel.n_epochs 5 \\
        --TrainCTLearnModel.batch_size 4""",
)

######################################################################
# 2. Arrival direction regression
# -------------------------------
#
# The model reconstructs the arrival direction of the primary particle.
# This can be done in two ways: either using the **cameradirection** task,
# which transforms the particle origin into the camera frame and performs
# the regression in camera coordinates, with distances expressed in meters;
# or using the **skydirection** task, which performs the regression in the
# sky frame, with field-of-view offsets expressed in degrees. For monoscopic
# CNN-based models, we generally recommend using the **cameradirection**
# task. During inference, the prediction tool converts the reconstructed
# camera coordinates back to the original Alt/Az representation, with the
# final direction expressed in degrees.

# Camera coordinates
run_command_line(
    """ctlearn-train-keras-model --verbose \\
        --signal "$CTLEARN_TUTORIAL_CONVERTED_DIR" \\
        --pattern-signal "gamma-diffuse_with_images_00.dl2.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/keras-cameradirection" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/ctlearn_train_model_example_config.json" \\
        --reco cameradirection \\
        --DLImageReader.focal_length_choice EQUIVALENT \\
        --TrainCTLearnModel.n_epochs 5 \\
        --TrainCTLearnModel.batch_size 4""",
)

run_command_line(
    """ctlearn-train-pytorch-model --verbose \\
        --signal "$CTLEARN_TUTORIAL_CONVERTED_DIR" \\
        --pattern-signal "gamma-diffuse_with_images_00.dl2.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/pytorch-cameradirection" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/ctlearn_train_model_example_config.json" \\
        --reco cameradirection \\
        --DLImageReader.focal_length_choice EQUIVALENT \\
        --TrainCTLearnModel.n_epochs 5 \\
        --TrainCTLearnModel.batch_size 4""",
)

# Sky coordinates
run_command_line(
    """ctlearn-train-keras-model --verbose \\
        --signal "$CTLEARN_TUTORIAL_CONVERTED_DIR" \\
        --pattern-signal "gamma-diffuse_with_images_00.dl2.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/keras-skydirection" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/ctlearn_train_model_example_config.json" \\
        --reco skydirection \\
        --DLImageReader.focal_length_choice EQUIVALENT \\
        --TrainCTLearnModel.n_epochs 5 \\
        --TrainCTLearnModel.batch_size 4""",
)

run_command_line(
    """ctlearn-train-pytorch-model --verbose \\
        --signal "$CTLEARN_TUTORIAL_CONVERTED_DIR" \\
        --pattern-signal "gamma-diffuse_with_images_00.dl2.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/pytorch-skydirection" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/ctlearn_train_model_example_config.json" \\
        --reco skydirection \\
        --DLImageReader.focal_length_choice EQUIVALENT \\
        --TrainCTLearnModel.n_epochs 5 \\
        --TrainCTLearnModel.batch_size 4""",
)

######################################################################
# 3. Primary particle type classification
# ---------------------------------------
#
# The model performs binary classification of the primary particle type,
# learning to distinguish between the two particle classes: signal
# (gamma-ray events) and background (proton events), based on the information
# contained in the camera image. The input directory containing the signal
# events can be specified using ``--TrainCTLearnModel.input_dir_signal`` or the
# shorthand ``--signal``. One or more file patterns can be provided using
# ``--TrainCTLearnModel.file_pattern_signal`` or ``--pattern-signal``. The
# corresponding options for the background events are
# ``--TrainCTLearnModel.input_dir_background`` / ``--background`` and
# ``--TrainCTLearnModel.file_pattern_background`` / ``--pattern-background``.
#
# It is recommended to approximately balance the training dataset before
# training. The training log reports the number of signal and background
# events remaining after the selection cuts, which can be used to check the
# class balance. If a significant imbalance is present, the input file
# patterns should be adjusted to include a more balanced selection of
# gamma-ray and proton events. CTLearn includes a standard, fixed mechanism
# that computes class weights and incorporates them into the loss
# calculation to compensate for differences in the number of signal and
# background events. However, class weighting is only effective up to a
# certain degree of class imbalance. It is therefore important to monitor
# the training carefully, particularly when the input samples are not well
# balanced. The training logs and learning curves should be inspected
# throughout the training. In particular, the evolution of metrics such as
# loss, accuracy, and AUC on a per-epoch basis can help identify potential
# issues arising from class imbalance or other training problems.

run_command_line(
    """ctlearn-train-keras-model --verbose \\
        --signal "$CTLEARN_TUTORIAL_CONVERTED_DIR" \\
        --pattern-signal "gamma-diffuse_with_images_00.dl2.h*" \\
        --background "$CTLEARN_TUTORIAL_CONVERTED_DIR" \\
        --pattern-background "proton_with_images_00.dl2.h*" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/keras-classification" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/ctlearn_train_model_example_config.json" \\
        --reco type \\
        --DLImageReader.focal_length_choice EQUIVALENT \\
        --TrainCTLearnModel.n_epochs 3 \\
        --TrainCTLearnModel.batch_size 4""",
)

run_command_line(
    """ctlearn-train-pytorch-model --verbose \\
        --signal "$CTLEARN_TUTORIAL_CONVERTED_DIR" \\
        --pattern-signal "gamma-diffuse_with_images_00.dl2.h*" \\
        --background "$CTLEARN_TUTORIAL_CONVERTED_DIR" \\
        --pattern-background "proton_with_images_00.dl2.h*" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/pytorch-classification" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/ctlearn_train_model_example_config.json" \\
        --reco type \\
        --DLImageReader.focal_length_choice EQUIVALENT \\
        --TrainCTLearnModel.n_epochs 3 \\
        --TrainCTLearnModel.batch_size 4""",
)

######################################################################
# Part B2: Customizing the CTLearn model architecture
# ==================================================
#
# CTLearn provides a flexible configuration system that allows users to
# customize the neural network architecture.
#
# Two main CNN-based architectures are currently implemented: **SingleCNN**
# and **ResNet**, which can be set by ``--TrainCTLearnModel.model_type``.

######################################################################
# 1. SingleCNN model
# ------------------
#
# The **SingleCNN** component implements a plain convolutional neural
# network consisting of configurable convolutional layers. It can be used
# to construct very simple and lightweight models, but it can also be
# configured to create deeper architectures, such as VGG-like networks.
# This architecture represents the more legacy model structure in CTLearn
# and was the starting point for early proof-of-concept studies exploring
# the applicability of CNNs to the problem. For applications closer to the
# camera hardware and its readout (for the trigger system), particularly
# where predictions need to be made at very high rates and with very low
# latency, computational efficiency becomes increasingly important.

run_command_line(
    """ctlearn-train-keras-model --verbose \\
        --signal "$CTLEARN_TUTORIAL_CONVERTED_DIR" \\
        --pattern-signal "gamma-diffuse_with_images_00.dl2.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/keras-singlecnn-energy" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/ctlearn_train_model_example_config.json" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/ctlearn_models_example_config.json" \\
        --reco energy \\
        --DLImageReader.focal_length_choice EQUIVALENT \\
        --TrainCTLearnModel.model_type SingleCNN \\
        --TrainCTLearnModel.n_epochs 2 \\
        --TrainCTLearnModel.batch_size 4""",
)

run_command_line(
    """ctlearn-train-pytorch-model --verbose \\
        --signal "$CTLEARN_TUTORIAL_CONVERTED_DIR" \\
        --pattern-signal "gamma-diffuse_with_images_00.dl2.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/pytorch-singlecnn-energy" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/ctlearn_train_model_example_config.json" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/ctlearn_models_example_config.json" \\
        --reco energy \\
        --DLImageReader.focal_length_choice EQUIVALENT \\
        --TrainCTLearnModel.model_type SingleCNN \\
        --TrainCTLearnModel.n_epochs 2 \\
        --TrainCTLearnModel.batch_size 4""",
)

######################################################################
# 2. ResNet model
# ---------------
#
# The **ResNet** component provides the currently preferred architecture for
# constructing deeper CNN models. Residual networks allow substantially
# deeper architectures while maintaining efficient training and inference,
# and they generally provide improved performance compared with the simpler
# plain CNN architecture. The architecture, number of layers or blocks,
# number of filters, and other relevant parameters can therefore be
# configured directly through the CTLearn configuration system, allowing
# the model complexity to be adapted to the requirements of a particular
# application. For ResNet architectures, CTLearn currently supports two
# types of residual blocks:
#
# - Basic blocks, which use the simpler residual block structure and are
#   suitable for relatively lightweight networks.
# - Bottleneck blocks, which use a more compact sequence of convolutions to
#   enable deeper networks while controlling the computational cost.

run_command_line(
    """ctlearn-train-keras-model --verbose \\
        --signal "$CTLEARN_TUTORIAL_CONVERTED_DIR" \\
        --pattern-signal "gamma-diffuse_with_images_00.dl2.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/keras-resnet-energy" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/ctlearn_train_model_example_config.json" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/ctlearn_models_example_config.json" \\
        --reco energy \\
        --DLImageReader.focal_length_choice EQUIVALENT \\
        --TrainCTLearnModel.model_type ResNet \\
        --TrainCTLearnModel.n_epochs 2 \\
        --TrainCTLearnModel.batch_size 4""",
)

run_command_line(
    """ctlearn-train-pytorch-model --verbose \\
        --signal "$CTLEARN_TUTORIAL_CONVERTED_DIR" \\
        --pattern-signal "gamma-diffuse_with_images_00.dl2.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/pytorch-resnet-energy" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/ctlearn_train_model_example_config.json" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/ctlearn_models_example_config.json" \\
        --reco energy \\
        --DLImageReader.focal_length_choice EQUIVALENT \\
        --TrainCTLearnModel.model_type ResNet \\
        --TrainCTLearnModel.n_epochs 2 \\
        --TrainCTLearnModel.batch_size 4""",
)

######################################################################
# Part C2: Using a pre-trained model and transfer learning
# =======================================================
#
# The LoadedModel component allows a previously trained model following
# the CTLearnModel structure to be loaded for either resuming training or
# transfer learning. Transfer learning reuses the learned features of an
# existing model and adapts them to new, but related, data or tasks. This
# can reduce training time compared with training a model from scratch.
#
# Two parameters are particularly important:
#
# - ``--LoadedModel.trainable_backbone`` controls whether the pre-trained
#   backbone is frozen (false) or trainable (true).
# - ``--LoadedModel.overwrite_head`` controls whether the structure of the
#   original prediction head is reused (false) or replaced with a new MLP
#   tailored to the new task or data (true).
#
# The preceding SingleCNN command trains the model loaded below. The copied
# configuration keeps its prediction head and leaves the backbone trainable.

run_command_line(
    """ctlearn-train-keras-model --verbose \\
        --signal "$CTLEARN_TUTORIAL_CONVERTED_DIR" \\
        --pattern-signal "gamma-diffuse_with_images_00.dl2.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/keras-transfer-learning-energy" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/ctlearn_train_model_example_config.json" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/ctlearn_models_example_config.json" \\
        --reco energy \\
        --DLImageReader.focal_length_choice EQUIVALENT \\
        --TrainCTLearnModel.model_type LoadedModel \\
        --LoadedModel.load_model_from "$CTLEARN_TUTORIAL_OUTPUT_DIR/keras-singlecnn-energy/ctlearn_model.keras" \\
        --TrainCTLearnModel.n_epochs 2 \\
        --TrainCTLearnModel.batch_size 4""",
)

run_command_line(
    """ctlearn-train-pytorch-model --verbose \\
        --signal "$CTLEARN_TUTORIAL_CONVERTED_DIR" \\
        --pattern-signal "gamma-diffuse_with_images_00.dl2.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/pytorch-transfer-learning-energy" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/ctlearn_train_model_example_config.json" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/ctlearn_models_example_config.json" \\
        --reco energy \\
        --DLImageReader.focal_length_choice EQUIVALENT \\
        --TrainCTLearnModel.model_type LoadedModel \\
        --PyTorchLoadedModel.load_model_from "$CTLEARN_TUTORIAL_OUTPUT_DIR/pytorch-singlecnn-energy/ctlearn_model.pth" \\
        --TrainCTLearnModel.n_epochs 2 \\
        --TrainCTLearnModel.batch_size 4""",
)

######################################################################
# Part D2: Stereoscopic CNN-based model training
# ---------------------------------------------
#
# CTLearn currently provides a basic approach for stereoscopic CNN-based
# model training. The images from the different telescopes are concatenated
# channel-wise and then passed to a standard, monoscopic CNN backbone as
# above. In this way, the network can process the combined input while
# retaining stereoscopic information. An earlier version of CTLearn also
# included a CNN-RNN architecture, using recurrent neural networks to
# explicitly handle information from multiple telescopes dynamically. This
# model has not yet been migrated to the latest CTLearn version based on
# TensorFlow 2 and Keras 3, so this tutorial focuses on the channel-wise
# concatenation approach.
#
# The main configuration parameters for stereoscopic training are:
#
# - ``--DLImageReader.mode``: set to "stereo" to enable stereoscopic image
#   reading
# - ``--DLImageReader.min_telescopes``: defines the minimum number of
#   telescopes that must participate in an event after the quality cuts.
# - ``--TrainCTLearnModel.stack_telescope_images``: set to true to
#   stack/concatenate the images from the telescopes.
#
# Since the resulting input contains information from multiple telescopes
# and is therefore larger than the corresponding monoscopic input, it is
# recommended to reduce the batch size using
# ``--TrainCTLearnModel.batch_size`` to account for the increased memory
# consumption.

run_command_line(
    """ctlearn-train-keras-model --verbose \\
        --signal "$CTLEARN_TUTORIAL_CONVERTED_DIR" \\
        --pattern-signal "gamma-diffuse_with_images_00.dl2.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/keras-stereo-energy" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/ctlearn_train_model_example_config.json" \\
        --reco energy \\
        --DLImageReader.mode stereo \\
        --DLImageReader.min_telescopes 2 \\
        --TrainCTLearnModel.stack_telescope_images True \\
        --TrainCTLearnModel.n_epochs 5 \\
        --TrainCTLearnModel.batch_size 2 \\
        --TableQualityQuery.quality_criteria []""",
)

run_command_line(
    """ctlearn-train-pytorch-model --verbose \\
        --signal "$CTLEARN_TUTORIAL_CONVERTED_DIR" \\
        --pattern-signal "gamma-diffuse_with_images_00.dl2.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/pytorch-stereo-energy" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/ctlearn_train_model_example_config.json" \\
        --reco energy \\
        --DLImageReader.mode stereo \\
        --DLImageReader.min_telescopes 2 \\
        --TrainCTLearnModel.stack_telescope_images True \\
        --TrainCTLearnModel.n_epochs 5 \\
        --TrainCTLearnModel.batch_size 2 \\
        --TableQualityQuery.quality_criteria []""",
)

######################################################################
# Part E2: CNN-based model training using waveform data
# ====================================================
#
# The final example demonstrates training directly on waveform data. The
# example dataset used in this tutorial is intentionally very small, so
# only a few events are available for training. In practical applications,
# waveform datasets can be very large, often reaching several terabytes,
# and processing the full waveforms also requires substantial RAM and
# computational resources. Because the example dataset is so small, the
# training configuration needs to be adjusted accordingly. We use a very
# small batch size and increase the validation split to ensure that at
# least one batch is available for validation. For this example, we select
# the MST equipped with the NectarCam camera and disable the quality
# selection, leaving only 10 events. These are divided into 5 batches of 2
# events, with 20% of the data reserved for validation, corresponding to
# one validation batch.
#
# An important configuration change is required to read waveform data: the
# ``--TrainCTLearnModel.dl1dh_reader_type`` option must be set to the
# appropriate DL1 data reader component for waveform reading
# (**DLWaveformReader**) rather than the default image reader.

if not WAVEFORM_FILE.is_file():
    raise FileNotFoundError(
        f"Cached waveform data is missing: {WAVEFORM_FILE}. "
        "Run examples/tutorials/explore_public_data.py first."
    )
run_command_line(
    """ctlearn-train-keras-model --verbose \\
        --signal "$CTLEARN_TUTORIAL_DATA_DIR" \\
        --pattern-signal "gamma_prod5${CTLEARN_TUTORIAL_MAX_EVENTS_SUFFIX}.r1.dl1.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/keras-waveform-energy" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/ctlearn_train_model_example_config.json" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --reco energy \\
        --TrainCTLearnModel.n_epochs 2 \\
        --TrainCTLearnModel.batch_size 2 \\
        --TrainCTLearnModel.validation_split 0.2 \\
        --TrainCTLearnModel.dl1dh_reader_type DLWaveformReader \\
        --DLWaveformReader.focal_length_choice EQUIVALENT \\
        --TableQualityQuery.quality_criteria []""",
)

run_command_line(
    """ctlearn-train-pytorch-model --verbose \\
        --signal "$CTLEARN_TUTORIAL_DATA_DIR" \\
        --pattern-signal "gamma_prod5${CTLEARN_TUTORIAL_MAX_EVENTS_SUFFIX}.r1.dl1.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/pytorch-waveform-energy" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/ctlearn_train_model_example_config.json" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --reco energy \\
        --TrainCTLearnModel.n_epochs 2 \\
        --TrainCTLearnModel.batch_size 2 \\
        --TrainCTLearnModel.validation_split 0.2 \\
        --TrainCTLearnModel.dl1dh_reader_type DLWaveformReader \\
        --DLWaveformReader.focal_length_choice EQUIVALENT \\
        --TableQualityQuery.quality_criteria []""",
)

######################################################################
# Part F2: Comparing training curves
# =================================
#
# Each training run writes its metrics to ``training_log.csv`` together
# with the TensorBoard model checkpoints for training and validation,
# the final model, the configuration and provenance files.
#
# See Part G for instructions on how to monitor the training runs live using
# TensorBoard. Here, we read the training logs and plot the evolution of the
# metrics over the training epochs. The plots show the mean absolute error (MAE) for
# the regression tasks and the accuracy and AUC for the classification task.
#
# Note: The statistics of the training runs are intentionally very limited, so the
# training curves are not expected to be smooth or to converge to optimal values.
# The purpose of this tutorial is to demonstrate the training workflow and the use
# of CTLearn, rather than to achieve optimal performance on the example dataset.

training_logs = {}
for run_name in (
    "keras-energy",
    "pytorch-energy",
    "keras-cameradirection",
    "pytorch-cameradirection",
    "keras-skydirection",
    "pytorch-skydirection",
    "keras-classification",
    "pytorch-classification",
):
    log_path = OUTPUT_DIR / run_name / "training_log.csv"
    if not log_path.is_file():
        raise FileNotFoundError(f"Training log is missing: {log_path}")
    training_logs[run_name] = pd.read_csv(log_path)

frameworks = (
    ("Keras", "keras", "tab:blue"),
    ("PyTorch", "pytorch", "tab:orange"),
)
splits = (("", "train", "-"), ("val_", "validation", "--"))


def plot_metric(ax, run_name, metric, label, color, **kwargs):
    log = training_logs[run_name]
    for prefix, split_name, split_style in splits:
        column = f"{prefix}{metric}"
        if column not in log:
            raise ValueError(
                f"Column {column!r} is missing from {run_name}/training_log.csv"
            )
        ax.plot(
            log["epoch"] + 1,
            log[column],
            color=color,
            linestyle=split_style,
            label=f"{label} - {split_name}",
            **kwargs,
        )


fig, (energy_ax, camera_ax, sky_ax, type_ax) = plt.subplots(
    1, 4, figsize=(24, 6), constrained_layout=True
)
fig.suptitle("CTLearn training curves", fontsize=16)

for display_name, framework, color in frameworks:
    plot_metric(
        energy_ax,
        f"{framework}-energy",
        "mae_energy",
        display_name,
        color,
        marker="o",
    )
energy_ax.set_title("Energy")
energy_ax.set_xlabel("Epoch")
energy_ax.set_ylabel("Mean absolute error")
energy_ax.grid(True, alpha=0.3)
energy_ax.legend(fontsize=8)

for display_name, framework, color in frameworks:
    plot_metric(
        camera_ax,
        f"{framework}-cameradirection",
        "mae_cameradirection",
        display_name,
        color,
        marker="o",
    )
    plot_metric(
        sky_ax,
        f"{framework}-skydirection",
        "mae_skydirection",
        display_name,
        color,
        marker="s",
    )
camera_ax.set_title("Camera direction")
camera_ax.set_xlabel("Epoch")
camera_ax.set_ylabel("Mean absolute error [m]")
camera_ax.grid(True, alpha=0.3)
camera_ax.legend(fontsize=8)
sky_ax.set_title("Sky direction")
sky_ax.set_xlabel("Epoch")
sky_ax.set_ylabel("Sky direction MAE [deg]")
sky_ax.grid(True, alpha=0.3)
sky_ax.legend(fontsize=8)

for display_name, framework, color in frameworks:
    for metric, marker in (("accuracy", "o"), ("auc", "s")):
        plot_metric(
            type_ax,
            f"{framework}-classification",
            metric,
            f"{display_name} {metric}",
            color,
            marker=marker,
        )
type_ax.set_title("Particle type")
type_ax.set_xlabel("Epoch")
type_ax.set_ylabel("Score")
type_ax.set_ylim(0.4, 0.7)
type_ax.grid(True, alpha=0.3)
type_ax.legend(fontsize=8, ncol=2, loc="best")

training_curves_path = OUTPUT_DIR / "training-curves.png"
fig.savefig(training_curves_path, dpi=350)
print(f"\nTraining curves saved to {training_curves_path}")
plt.show()

######################################################################
# Part G2: Monitoring training with TensorBoard
# ============================================
#
# CTLearn also writes TensorBoard event files, which are useful for monitoring
# metrics live or exploring additional details after training. The tutorial
# does not launch TensorBoard; to monitor a run while it is training, start it
# in a separate terminal and point it at the tutorial output directory:
#
# .. code-block:: console
#
#    tensorboard --logdir "$HOME/.cache/ctlearn/tutorial-training/"
#
# Open the URL printed by TensorBoard in a browser. If you set
# ``CTLEARN_TUTORIAL_OUTPUT_DIR`` to a different location, use that directory
# in the command instead. On a remote or cluster environment, use the
# appropriate port forwarding to access the TensorBoard web interface.
