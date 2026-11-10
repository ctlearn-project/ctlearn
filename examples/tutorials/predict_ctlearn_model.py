"""
**Tutorial 3**: Perform inference with CTlearn models
=====================================================

Predict the CTAO image and waveform data using the CTLearn models
from the second tutorial ``train_ctlearn_model.py``. The predictions
are stored in DL2 files following the CTAO reference data model.
The tutorial also shows how to visualize the classifier performance,
energy resolution, and angular resolution of the predicted events.
The tutorial is intenionally kept simple and uses only a small fraction
of the available statistics to keep the prediction time manageable.
The resulting performance metrics are not representative of a production-level
analysis nor should they be compared directly with published results. The main
goal is to illustrate the analysis workflow and to become familiar with the
command-line tools and the DL2 data format. This tutorial can be modified from
the users to use larger datasets and more statistics to obtain meaningful results.

Each section explains a prediction task followed immediately by its literal
command-line example. Running this script executes every example in order.
"""

######################################################################
# Check paths, environments, CTAO test data, and trained models
# -------------------------------------------------------------
#
# The executable tutorial uses the shared downloaded and processed data
# from ``explore_public_data.py`` and the Keras and PyTorch models produced
# by ``train_ctlearn_model.py``. In a notebook, prefix the displayed
# command-line examples with ``!``.

import os
import subprocess
from pathlib import Path

import astropy.units as u
import ctaplot
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Column, Table
from ctapipe import __version__ as ctapipe_version
from ctapipe.io import TableLoader, write_table
from sklearn.metrics import auc, roc_curve

from ctlearn.tools.utils import get_lst1_subarray_description

MAX_EVENTS = os.environ.get("CTLEARN_TUTORIAL_MAX_EVENTS")
if MAX_EVENTS is not None and int(MAX_EVENTS) < 1:
    raise ValueError("CTLEARN_TUTORIAL_MAX_EVENTS must be a positive integer")
MAX_EVENTS_SUFFIX = f"-max{MAX_EVENTS}" if MAX_EVENTS is not None else ""

DATA_DIR = (
    Path(
        os.environ.get(
            "CTLEARN_TUTORIAL_DATA_DIR",
            str(Path.home() / ".cache" / "ctlearn" / "tutorial-data"),
        )
    )
    .expanduser()
    .resolve()
)
CONVERTED_DIR = (
    Path(
        os.environ.get(
            "CTLEARN_TUTORIAL_CONVERTED_DIR",
            str(DATA_DIR / f"ctapipe-{ctapipe_version}{MAX_EVENTS_SUFFIX}"),
        )
    )
    .expanduser()
    .resolve()
)
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
PREDICTION_DIR = OUTPUT_DIR / "predictions"
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


def find_config_dir():
    roots = []
    script_path = globals().get("__file__")
    if script_path is not None:
        roots.append(Path(script_path).resolve().parents[2])
    roots.extend((Path.cwd(), *Path.cwd().parents))

    for root in roots:
        config_dir = root / "docs" / "source" / "config"
        if (config_dir / "dl1dh_example_config.json").is_file():
            return config_dir

    raise FileNotFoundError(
        "Could not locate tutorial configs under docs/source/config. "
        "Run this tutorial from the CTLearn repository or build it with "
        "Sphinx-Gallery."
    )


CONFIG_DIR = find_config_dir()
DL1DH_CONFIG = CONFIG_DIR / "dl1dh_example_config.json"
for config_path in (DL1DH_CONFIG,):
    if not config_path.is_file():
        raise FileNotFoundError(f"Tutorial configuration is missing: {config_path}")

GAMMA_FILE = CONVERTED_DIR / "gamma-diffuse_with_images_01.dl2.h5"
PROTON_FILE = CONVERTED_DIR / "proton_with_images_01.dl2.h5"
missing_files = [path for path in (GAMMA_FILE, PROTON_FILE) if not path.is_file()]
if missing_files:
    missing = "\n".join(f"  - {path}" for path in missing_files)
    raise FileNotFoundError(
        "Held-out tutorial data is missing. Run "
        "examples/tutorials/explore_public_data.py first. Missing files:\n"
        f"{missing}"
    )

os.environ["CTLEARN_TUTORIAL_DATA_DIR"] = str(DATA_DIR)
os.environ["CTLEARN_TUTORIAL_CONVERTED_DIR"] = str(CONVERTED_DIR)
os.environ["CTLEARN_TUTORIAL_OUTPUT_DIR"] = str(OUTPUT_DIR)
os.environ["CTLEARN_TUTORIAL_CONFIG_DIR"] = str(CONFIG_DIR.resolve())
os.environ["CTLEARN_TUTORIAL_MAX_EVENTS_SUFFIX"] = MAX_EVENTS_SUFFIX

print(f"Converted held-out data: {CONVERTED_DIR}")
print(f"Trained model outputs: {OUTPUT_DIR}")
print(f"Prediction outputs: {PREDICTION_DIR}")
print(f"Reader configuration: {DL1DH_CONFIG}")


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
# Explore the CTLearn prediction tools
# ------------------------------------
#
# Installing CTLearn provides several command-line tools for running and
# managing different parts of the workflow. For model inference, we use the
# ``ctlearn-predict-{mono,stereo}-model`` command. The ``-h`` option displays a
# short version of the help message, showing the main command-line options and
# their usage. For a complete description of all available options, including
# the configuration parameters of the different components, use ``--help-all``.
# Although the command is executed from this Jupyter notebook for convenience
# and documentation, ``ctlearn-predict-{mono,stereo}-model`` is a
# command-line tool. In a typical workflow, the prediction command would be
# executed directly from a terminal or submitted as part of a cluster job,
# for example through sbatch on an HPC system.

run_command_line("ctlearn-predict-mono-model -h")
run_command_line("ctlearn-predict-stereo-model -h")

######################################################################
# *Part A3*: Monoscopic CNN-based model inference
# ===============================================
#
# We perform inference using the three previously trained CTLearn models from
# ``train_ctlearn_model.py``: the primary particle type (classification),
# primary particle energy (regression), and arrival direction (regression)
# models. The models are applied to both gamma-ray (signal) and proton
# (background) input data. The three inference tasks are executed sequentially
# within a single tool invocation. The prediction tool provides several
# options for selecting which data levels from the input file are copied to
# the output file. By default, the goal is to progress to the next data level
# while keeping the output file as compact as possible. However, retaining
# additional data levels can be useful for debugging and for inspecting
# individual events in more detail. The resulting file contains the model
# predictions together with the selected input data levels and follows the
# CTAO reference data model.
#
# **Important**: The same configuration settings used during training should
# be used for inference whenever applicable. In particular, the data
# preprocessing and data reader configuration must be consistent with those
# used to train the models. The ``01`` files are used here as held-out input
# data, and the paths to each model are shown directly in the commands.

######################################################################
# 1. Keras monoscopic model prediction
# ------------------------------------

run_command_line(
    """ctlearn-predict-mono-model --verbose \\
        --input_url "$CTLEARN_TUTORIAL_CONVERTED_DIR/gamma-diffuse_with_images_01.dl2.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/predictions/gamma-diffuse_with_images_01.ctlearn.dl2.h5" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --type_model "$CTLEARN_TUTORIAL_OUTPUT_DIR/keras-classification/ctlearn_model.keras" \\
        --energy_model "$CTLEARN_TUTORIAL_OUTPUT_DIR/keras-energy/ctlearn_model.keras" \\
        --cameradirection_model "$CTLEARN_TUTORIAL_OUTPUT_DIR/keras-cameradirection/ctlearn_model.keras" \\
        --PredictCTLearnModel.batch_size 16 \\
        --overwrite"""
)
run_command_line(
    """ctlearn-predict-mono-model --verbose \\
        --input_url "$CTLEARN_TUTORIAL_CONVERTED_DIR/proton_with_images_01.dl2.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/predictions/proton_with_images_01.ctlearn.dl2.h5" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --type_model "$CTLEARN_TUTORIAL_OUTPUT_DIR/keras-classification/ctlearn_model.keras" \\
        --energy_model "$CTLEARN_TUTORIAL_OUTPUT_DIR/keras-energy/ctlearn_model.keras" \\
        --cameradirection_model "$CTLEARN_TUTORIAL_OUTPUT_DIR/keras-cameradirection/ctlearn_model.keras" \\
        --PredictCTLearnModel.batch_size 16 \\
        --overwrite"""
)

######################################################################
# 2. PyTorch monoscopic model prediction
# --------------------------------------
#
# The same prediction tasks can be run with the PyTorch models. The model
# paths below point to the corresponding PyTorch outputs from
# ``train_ctlearn_model.py``.

run_command_line(
    """ctlearn-predict-mono-model --verbose \\
        --input_url "$CTLEARN_TUTORIAL_CONVERTED_DIR/gamma-diffuse_with_images_01.dl2.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/predictions/gamma-diffuse_with_images_01.pytorch.ctlearn.dl2.h5" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --type_model "$CTLEARN_TUTORIAL_OUTPUT_DIR/pytorch-classification/ctlearn_model.pth" \\
        --energy_model "$CTLEARN_TUTORIAL_OUTPUT_DIR/pytorch-energy/ctlearn_model.pth" \\
        --cameradirection_model "$CTLEARN_TUTORIAL_OUTPUT_DIR/pytorch-cameradirection/ctlearn_model.pth" \\
        --PredictCTLearnModel.batch_size 16 \\
        --overwrite"""
)
run_command_line(
    """ctlearn-predict-mono-model --verbose \\
        --input_url "$CTLEARN_TUTORIAL_CONVERTED_DIR/proton_with_images_01.dl2.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/predictions/proton_with_images_01.pytorch.ctlearn.dl2.h5" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --type_model "$CTLEARN_TUTORIAL_OUTPUT_DIR/pytorch-classification/ctlearn_model.pth" \\
        --energy_model "$CTLEARN_TUTORIAL_OUTPUT_DIR/pytorch-energy/ctlearn_model.pth" \\
        --cameradirection_model "$CTLEARN_TUTORIAL_OUTPUT_DIR/pytorch-cameradirection/ctlearn_model.pth" \\
        --PredictCTLearnModel.batch_size 16 \\
        --overwrite"""
)

######################################################################
# *Part B3*: Exploring the produced CTLearn DL2 data
# ==================================================
#
# In the following cells, we will explore the DL2 predictions produced by the
# CTLearn models. Keep in mind that, for this hands-on tutorial, we trained
# the models on very small datasets with limited statistics to keep the
# training time manageable. The datasets used here represent only a tiny
# fraction of the statistics typically used for a production-level CNN-based
# analysis, where GPU resources are also used to accelerate the training
# process. Consequently, the performance metrics and resolution curves
# obtained in this tutorial should not be considered representative of a
# production analysis, compared directly with published results, or used for
# scientific conclusions. The main goal here is to understand how the
# analysis workflow is conducted: how the predictions are loaded, how the
# different performance metrics are calculated, and how the corresponding
# distributions and resolution curves are visualized. Once we are familiar
# with the complete workflow, we can increase the statistics and repeat the
# analysis using a full production of simulated data. The full production
# simulations used by the CTAO collaboration are not publicly available.
# However, the statistics available in the CTAO Open Data are sufficient to
# obtain meaningful proof-of-concept results and to explore the analysis
# workflow in much greater detail.
#
# The first cell loads the data using the **TableLoader** from ctapipe. The
# produced files follow the reference implementation of the CTAO data model,
# allowing the DL2 information to be read in a standardized format. The cell
# also selects and prepares the relevant quantities from the predictions so
# that they can be conveniently used in the following cells for plotting the
# classifier performance, energy reconstruction, angular reconstruction, and
# other performance metrics.


def read_valid_telescope_events(filename):
    with TableLoader(input_url=filename) as loader:
        telescope_events = loader.read_telescope_events(
            telescopes=["LST_LST_LSTCam"],
            dl1_images=False,
            dl1_parameters=False,
            dl1_muons=False,
            dl2=True,
            simulated=True,
            true_images=False,
            true_parameters=False,
            instrument=False,
            observation_info=False,
        )
    return telescope_events[telescope_events["CTLearnRegressor_tel_is_valid"]]


gamma_keras_events = read_valid_telescope_events(
    PREDICTION_DIR / "gamma-diffuse_with_images_01.ctlearn.dl2.h5"
)
proton_keras_events = read_valid_telescope_events(
    PREDICTION_DIR / "proton_with_images_01.ctlearn.dl2.h5"
)
gamma_pytorch_events = read_valid_telescope_events(
    PREDICTION_DIR / "gamma-diffuse_with_images_01.pytorch.ctlearn.dl2.h5"
)
proton_pytorch_events = read_valid_telescope_events(
    PREDICTION_DIR / "proton_with_images_01.pytorch.ctlearn.dl2.h5"
)

gamma_keras_predictions = np.asarray(
    gamma_keras_events["CTLearnClassifier_tel_prediction"]
)
proton_keras_predictions = np.asarray(
    proton_keras_events["CTLearnClassifier_tel_prediction"]
)
gamma_pytorch_predictions = np.asarray(
    gamma_pytorch_events["CTLearnClassifier_tel_prediction"]
)
proton_pytorch_predictions = np.asarray(
    proton_pytorch_events["CTLearnClassifier_tel_prediction"]
)
gamma_scores = gamma_keras_predictions[gamma_keras_predictions != 0.0]
proton_scores = proton_keras_predictions[proton_keras_predictions != 0.0]
gamma_pytorch_scores = gamma_pytorch_predictions[gamma_pytorch_predictions != 0.0]
proton_pytorch_scores = proton_pytorch_predictions[proton_pytorch_predictions != 0.0]

reco_energy = np.asarray(gamma_keras_events["CTLearnRegressor_tel_energy"])
true_energy = np.asarray(gamma_keras_events["true_energy"])
proton_true_energy = np.asarray(proton_keras_events["true_energy"])
reco_alt = np.asarray(gamma_keras_events["CTLearnCameraReconstructor_tel_alt"])
true_alt = np.asarray(gamma_keras_events["true_alt"])
reco_az = np.asarray(gamma_keras_events["CTLearnCameraReconstructor_tel_az"])
true_az = np.asarray(gamma_keras_events["true_az"])

######################################################################
# 1. Particle classification performance
# --------------------------------------
#
# This cell visualizes the particle classification performance of the CTLearn
# model, with the goal of discriminating gamma-ray signal events from proton
# background events. Three complementary plots are shown: on the left, the
# distribution of the classifier output, or gammaness, for gamma-ray and
# proton events; in the middle, the Receiver Operating Characteristic (ROC)
# curve, which shows the trade-off between the true positive rate and the
# false positive rate as the classification threshold is varied, together
# with the corresponding Area Under the Curve (AUC); and on the right, the AUC
# as a function of true energy, showing how the classification performance
# changes across the energy range of the simulated events. Together, these
# plots provide an overview of the classifier performance, both globally and
# as a function of energy. As with the other performance metrics presented in
# this tutorial, these results are intended to illustrate the analysis
# workflow rather than provide production-level performance estimates.
#
# The same plots compare the Keras and PyTorch classifiers.

y_true = np.concatenate([np.ones(len(gamma_scores)), np.zeros(len(proton_scores))])
y_score = np.concatenate([gamma_scores, proton_scores])
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
torch_y_true = np.concatenate(
    [
        np.ones(len(gamma_pytorch_scores)),
        np.zeros(len(proton_pytorch_scores)),
    ]
)
torch_y_score = np.concatenate([gamma_pytorch_scores, proton_pytorch_scores])
torch_fpr, torch_tpr, _ = roc_curve(torch_y_true, torch_y_score)
torch_roc_auc = auc(torch_fpr, torch_tpr)
gamma_energy_auc = true_energy[gamma_keras_predictions != 0.0]
proton_energy_auc = proton_true_energy[proton_keras_predictions != 0.0]
true_energy_auc = np.concatenate([gamma_energy_auc, proton_energy_auc])
energy_bins_auc = np.logspace(np.log10(0.03), np.log10(100), 15) * u.TeV
energy_bins_auc, auc_scores = ctaplot.roc_auc_per_energy(
    true_type=y_true,
    gammaness=y_score,
    true_energy=true_energy_auc * u.TeV,
    energy_bins=energy_bins_auc,
)
gamma_pytorch_energy_auc = np.asarray(gamma_pytorch_events["true_energy"])[
    gamma_pytorch_predictions != 0.0
]
proton_pytorch_energy_auc = np.asarray(proton_pytorch_events["true_energy"])[
    proton_pytorch_predictions != 0.0
]
pytorch_true_energy_auc = np.concatenate(
    [gamma_pytorch_energy_auc, proton_pytorch_energy_auc]
)
pytorch_energy_bins_auc, pytorch_auc_scores = ctaplot.roc_auc_per_energy(
    true_type=torch_y_true,
    gammaness=torch_y_score,
    true_energy=pytorch_true_energy_auc * u.TeV,
    energy_bins=np.logspace(np.log10(0.03), np.log10(100), 15) * u.TeV,
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
score_bins = np.linspace(0, 1, 50)
axes[0].hist(
    gamma_scores,
    bins=score_bins,
    histtype="step",
    linewidth=1.5,
    label="Keras signal (gamma)",
)
axes[0].hist(
    proton_scores,
    bins=score_bins,
    histtype="step",
    linewidth=1.5,
    label="Keras background (proton)",
)
axes[0].hist(
    gamma_pytorch_scores,
    bins=score_bins,
    histtype="step",
    linewidth=1.5,
    label="PyTorch signal (gamma)",
)
axes[0].hist(
    proton_pytorch_scores,
    bins=score_bins,
    histtype="step",
    linewidth=1.5,
    label="PyTorch background (proton)",
)
axes[0].set_xlabel("Classifier value (gammaness)")
axes[0].set_ylabel("Counts")
axes[0].set_title("Gammaness distribution")
axes[0].legend(loc="upper center", fontsize="small")

axes[1].plot(fpr, tpr, lw=2, label=f"Keras classifier (AUC = {roc_auc:.3f})")
axes[1].plot(
    torch_fpr,
    torch_tpr,
    lw=2,
    label=f"PyTorch classifier (AUC = {torch_roc_auc:.3f})",
)
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random classifier")
axes[1].set_xlabel("False positive rate")
axes[1].set_ylabel("True positive rate")
axes[1].set_title("ROC curve")
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)
axes[1].grid(alpha=0.3)
axes[1].legend(loc="lower right")

ctaplot.plot_roc_auc_per_energy(
    energy_bins=energy_bins_auc,
    auc_scores=auc_scores,
    label="Keras classifier",
    ax=axes[2],
)
ctaplot.plot_roc_auc_per_energy(
    energy_bins=pytorch_energy_bins_auc,
    auc_scores=pytorch_auc_scores,
    label="PyTorch classifier",
    ax=axes[2],
)
axes[2].set_title("AUC vs. energy")
axes[2].set_ylim(0.5, 1.0)
axes[2].legend()
plt.show()

######################################################################
# 2. Regression migration matrices
# --------------------------------
#
# The next cell shows the migration matrices for the regression tasks of the
# CTLearn model, covering the reconstructed energy and arrival direction. The
# figure contains three subplots: the energy migration matrix on the left,
# the altitude migration matrix in the middle, and the azimuth migration
# matrix on the right. In each case, the true value is shown on the x-axis and
# the reconstructed value on the y-axis, allowing us to visualize how the
# reconstructed quantities are distributed with respect to their true values.
# The azimuth is wrapped around (0º/360º) because the simulated observations
# are pointing towards the north, so (0º) and (360º) represent the same
# direction. The black diagonal lines indicate perfect reconstruction, where
# the reconstructed value is exactly equal to the true value. The closer the
# event distribution is concentrated around this diagonal, the more accurately
# the corresponding regression task reconstructs the physical quantity.

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
energy_bins = np.logspace(np.log10(1), np.log10(100), 50)
hist_energy, true_energy_edges, reco_energy_edges = np.histogram2d(
    true_energy, reco_energy, bins=[energy_bins, energy_bins]
)
energy_mesh = axes[0].pcolormesh(
    true_energy_edges,
    reco_energy_edges,
    np.ma.masked_where(hist_energy == 0, hist_energy).T,
    cmap="viridis",
    shading="auto",
)
axes[0].plot(
    energy_bins,
    energy_bins,
    "k--",
    linewidth=1.5,
    label="Perfect reconstruction",
)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("True energy [TeV]")
axes[0].set_ylabel("Reconstructed energy [TeV]")
axes[0].set_title("Energy migration")
axes[0].legend(loc="upper left")
fig.colorbar(energy_mesh, ax=axes[0]).set_label("Number of events")

altitude_bins = np.linspace(
    min(true_alt.min(), reco_alt.min()),
    max(true_alt.max(), reco_alt.max()),
    50,
)
hist_altitude, true_altitude_edges, reco_altitude_edges = np.histogram2d(
    true_alt, reco_alt, bins=[altitude_bins, altitude_bins]
)
altitude_mesh = axes[1].pcolormesh(
    true_altitude_edges,
    reco_altitude_edges,
    np.ma.masked_where(hist_altitude == 0, hist_altitude).T,
    cmap="viridis",
    shading="auto",
)
axes[1].plot(
    altitude_bins,
    altitude_bins,
    "k--",
    linewidth=1.5,
    label="Perfect reconstruction",
)
axes[1].set_xlabel("True altitude [deg]")
axes[1].set_ylabel("Reconstructed altitude [deg]")
axes[1].set_title("Altitude migration")
axes[1].legend(loc="upper left")
fig.colorbar(altitude_mesh, ax=axes[1]).set_label("Number of events")

true_az_wrapped = (true_az + 180) % 360 - 180
reco_az_wrapped = (reco_az + 180) % 360 - 180
azimuth_bins = np.linspace(-15, 15, 61)
hist_azimuth, true_azimuth_edges, reco_azimuth_edges = np.histogram2d(
    true_az_wrapped, reco_az_wrapped, bins=[azimuth_bins, azimuth_bins]
)
azimuth_mesh = axes[2].pcolormesh(
    true_azimuth_edges,
    reco_azimuth_edges,
    np.ma.masked_where(hist_azimuth == 0, hist_azimuth).T,
    cmap="viridis",
    shading="auto",
)
axes[2].plot(
    [-15, 15],
    [-15, 15],
    "k--",
    linewidth=1.5,
    label="Perfect reconstruction",
)
axes[2].set_xlim(-15, 15)
axes[2].set_ylim(-15, 15)
axes[2].set_xlabel("True azimuth relative to 0° [deg]")
axes[2].set_ylabel("Reconstructed azimuth relative to 0° [deg]")
axes[2].set_title("Azimuth migration")
axes[2].set_xticks([-15, -10, -5, 0, 5, 10, 15])
axes[2].set_yticks([-15, -10, -5, 0, 5, 10, 15])
axes[2].legend(loc="upper left")
fig.colorbar(azimuth_mesh, ax=axes[2]).set_label("Number of events")
plt.tight_layout()
plt.show()

######################################################################
# 3. IACT-specific performance metrics
# ------------------------------------
#
# The last cell shows some IACT-specific performance metrics, in particular
# the energy resolution and angular resolution of the reconstructed events.
# These plots provide a first visual impression of how well the model
# reconstructs the physical properties of the gamma-ray events. It is
# important to note that these are simple plots of the reconstruction
# performance without applying any selection cuts based on the gammaness
# classifier. Therefore, they should not be confused with the **Instrument
# Response Functions (IRFs)** used in standard IACT analyses. The production
# of IRFs involves a dedicated analysis workflow and additional selections
# and calculations. Separate tools within ctapipe are available to build
# these higher-level analysis products, and they can be applied directly to
# the CTLearn DL2 data because the files follow the reference implementation
# of the CTAO data model.
#
# For this tutorial, the purpose is simply to obtain a first visual impression
# of the energy and angular reconstruction performance and to become familiar
# with the corresponding analysis workflow. As emphasized throughout this
# tutorial, the available statistics are very limited, so the resulting
# curves should not be interpreted as meaningful or competitive instrument
# performance measurements, nor should they be compared directly with
# published IACT performance results.

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
ctaplot.plot_energy_resolution(
    true_energy * u.TeV,
    reco_energy * u.TeV,
    label="CTLearn",
    ax=axes[0],
)
ctaplot.plot_energy_resolution_cta_requirement("north", ax=axes[0], color="black")
axes[0].legend()
axes[0].set_title("Energy resolution")

ctaplot.plot_angular_resolution_per_energy(
    true_alt * u.deg,
    reco_alt * u.deg,
    true_az * u.deg,
    reco_az * u.deg,
    true_energy * u.TeV,
    label="CTLearn",
    ax=axes[1],
)
ctaplot.plot_angular_resolution_cta_requirement("north", ax=axes[1], color="black")
axes[1].legend()
axes[1].set_title("Angular resolution")
plt.tight_layout()
plt.show()


######################################################################
# 4. Browsing through the HDF5 files via ViTables
# -----------------------------------------------
#
# Use ViTables, a convenient GUI, to explore the CTLearn DL2 predictions. The
# file follows the reference implementation of the CTAO data model.
#
# .. code-block:: console
#
#    pip install vitables
#    vitables $HOME/.cache/ctlearn/tutorial-training/predictions/*.dl2.h5
#
# Note that the ViTables GUI is not available in this notebook environment, so
# the command is shown here for reference. In a local environment, you can run
# the command in a terminal to open the ViTables GUI and explore the DL2
# predictions interactively. The GUI allows you to browse through the tables,
# view the data and metadata, and perform basic operations on the HDF5 files.
# This can be useful for inspecting individual events, checking the structure
# of the data, and verifying that the predictions are stored correctly.

######################################################################
# *Part C3*: Stereoscopic CNN-based model inference
# =================================================
#
# We perform inference with the previously trained stereoscopic energy model.
# The telescope images are combined channel-wise and passed to the CNN,
# allowing the model to use information from multiple telescopes
# simultaneously. As above, the same configuration used during training
# should be used for inference, in particular the stereoscopic image reader
# settings and telescope stacking options. Since the stereoscopic input is
# larger than the monoscopic input, a reduced batch size may be required to
# account for the increased memory consumption.
#
# The following commands run the inference with both the Keras and PyTorch
# models.

######################################################################
# 1. Keras stereoscopic model prediction
# --------------------------------------

run_command_line(
    """ctlearn-predict-stereo-model --verbose \\
        --input_url "$CTLEARN_TUTORIAL_CONVERTED_DIR/gamma-diffuse_with_images_01.dl2.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/predictions/gamma-diffuse_with_images_01.ctlearn.stereo.dl2.h5" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --energy_model "$CTLEARN_TUTORIAL_OUTPUT_DIR/keras-stereo-energy/ctlearn_model.keras" \\
        --DLImageReader.mode stereo --DLImageReader.min_telescopes 2 \\
        --TableQualityQuery.quality_criteria "[]" \\
        --PredictCTLearnModel.stack_telescope_images True \\
        --PredictCTLearnModel.batch_size 2 \\
        --overwrite"""
)
run_command_line(
    """ctlearn-predict-stereo-model --verbose \\
        --input_url "$CTLEARN_TUTORIAL_CONVERTED_DIR/proton_with_images_01.dl2.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/predictions/proton_with_images_01.ctlearn.stereo.dl2.h5" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --energy_model "$CTLEARN_TUTORIAL_OUTPUT_DIR/keras-stereo-energy/ctlearn_model.keras" \\
        --DLImageReader.mode stereo --DLImageReader.min_telescopes 2 \\
        --TableQualityQuery.quality_criteria "[]" \\
        --PredictCTLearnModel.stack_telescope_images True \\
        --PredictCTLearnModel.batch_size 2 \\
        --overwrite"""
)

######################################################################
# 2. PyTorch stereoscopic model prediction
# ----------------------------------------

run_command_line(
    """ctlearn-predict-stereo-model --verbose \\
        --input_url "$CTLEARN_TUTORIAL_CONVERTED_DIR/gamma-diffuse_with_images_01.dl2.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/predictions/gamma-diffuse_with_images_01.pytorch.ctlearn.stereo.dl2.h5" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --energy_model "$CTLEARN_TUTORIAL_OUTPUT_DIR/pytorch-stereo-energy/ctlearn_model.pth" \\
        --DLImageReader.mode stereo --DLImageReader.min_telescopes 2 \\
        --TableQualityQuery.quality_criteria "[]" \\
        --PredictCTLearnModel.stack_telescope_images True \\
        --PredictCTLearnModel.batch_size 2 \\
        --overwrite"""
)
run_command_line(
    """ctlearn-predict-stereo-model --verbose \\
        --input_url "$CTLEARN_TUTORIAL_CONVERTED_DIR/proton_with_images_01.dl2.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/predictions/proton_with_images_01.pytorch.ctlearn.stereo.dl2.h5" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --energy_model "$CTLEARN_TUTORIAL_OUTPUT_DIR/pytorch-stereo-energy/ctlearn_model.pth" \\
        --DLImageReader.mode stereo --DLImageReader.min_telescopes 2 \\
        --TableQualityQuery.quality_criteria "[]" \\
        --PredictCTLearnModel.stack_telescope_images True \\
        --PredictCTLearnModel.batch_size 2 \\
        --overwrite"""
)

######################################################################
# *Part D3*: CNN-based model inference using waveform data
# ========================================================
#
# We perform inference with the previously trained mono energy model using
# waveform data. The configuration must be consistent with the training
# setup. An important configuration change is required to read waveform data:
# the ``--PredictCTLearnModel.dl1dh_reader_type`` option must be set to the
# appropriate DL1 data reader component for waveform reading
# (**DLWaveformReader**) rather than the default image reader.
#
# For this demonstration, we use the same data file as used during training,
# since suitable test data are currently not available. This is done only to
# demonstrate that the software workflow is functional. In a real analysis,
# using training data for testing is not valid practice.
#
# The following commands run the inference with both the Keras and PyTorch
# models.

WAVEFORM_FILE = DATA_DIR / f"gamma_prod5{MAX_EVENTS_SUFFIX}.r1.dl1.h5"
if not WAVEFORM_FILE.is_file():
    raise FileNotFoundError(
        f"Cached waveform data is missing: {WAVEFORM_FILE}. "
        "Run examples/tutorials/explore_public_data.py first."
    )

######################################################################
# 1. Keras waveform model prediction
# ----------------------------------

run_command_line(
    """ctlearn-predict-mono-model --verbose \\
        --input_url "$CTLEARN_TUTORIAL_DATA_DIR/gamma_prod5${CTLEARN_TUTORIAL_MAX_EVENTS_SUFFIX}.r1.dl1.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/predictions/gamma_prod5.keras.waveform.dl2.h5" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --energy_model "$CTLEARN_TUTORIAL_OUTPUT_DIR/keras-waveform-energy/ctlearn_model.keras" \\
        --PredictCTLearnModel.dl1dh_reader_type DLWaveformReader \\
        --PredictCTLearnModel.batch_size 2 \\
        --TableQualityQuery.quality_criteria "[]" \\
        --overwrite"""
)

######################################################################
# 2. PyTorch waveform model prediction
# ------------------------------------

run_command_line(
    """ctlearn-predict-mono-model --verbose \\
        --input_url "$CTLEARN_TUTORIAL_DATA_DIR/gamma_prod5${CTLEARN_TUTORIAL_MAX_EVENTS_SUFFIX}.r1.dl1.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/predictions/gamma_prod5.pytorch.waveform.dl2.h5" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --energy_model "$CTLEARN_TUTORIAL_OUTPUT_DIR/pytorch-waveform-energy/ctlearn_model.pth" \\
        --PredictCTLearnModel.dl1dh_reader_type DLWaveformReader \\
        --PredictCTLearnModel.batch_size 2 \\
        --TableQualityQuery.quality_criteria "[]" \\
        --overwrite"""
)

######################################################################
# *Part E3*: CNN-based model inference using LST-1 lstchain data
# ==============================================================
#
# The following three cells demonstrate the prediction tool for LST-1 DL1
# data in the lstchain format. Since LST-1 data are private to the LST-1
# collaboration and cannot be distributed, the first cell creates a mock
# dataset in the required format. The second cell displays the tool help and
# its available options. The final cell performs inference on the mock data
# using the models trained in the previous notebook. The tool predicts
# gammaness, energy, and arrival direction and writes the results to a ctapipe
# DL2 output file, together with the relevant LST-1 subarray, trigger,
# pointing, and DL1b parameter information.
#
# **Note**: This prediction tool is specifically designed for the lstchain
# DL1 format and should not be used with other input data formats.
#
# The inference commands below use the Keras and PyTorch models.

n_events = 25
rng = np.random.default_rng(1234)
subarray = get_lst1_subarray_description()
tel_id = 1
n_pixels = subarray.tel[tel_id].camera.geometry.n_pixels
event_ids = np.arange(1, n_events + 1, dtype=np.int64)
mock_lst1_file = PREDICTION_DIR / "mock_data_lst1.dl1.h5"

image_table = Table()
image_table["obs_id"] = np.ones(n_events, dtype=np.int64)
image_table["event_id"] = event_ids
image_table["tel_id"] = np.full(n_events, tel_id, dtype=np.int16)
image_table.add_column(
    Column(
        rng.uniform(80, 150, size=(n_events, n_pixels)).astype(np.float32),
        name="image",
        dtype=np.float32,
        shape=(n_pixels,),
    )
)
image_table.add_column(
    Column(
        rng.integers(0, 2, size=(n_events, n_pixels), dtype=bool),
        name="image_mask",
        dtype=bool,
        shape=(n_pixels,),
    )
)
image_table.add_column(
    Column(
        rng.normal(5.0, 0.5, size=(n_events, n_pixels)).astype(np.float32),
        name="peak_time",
        dtype=np.float32,
        shape=(n_pixels,),
    )
)

parameter_table = Table()
parameter_table["obs_id"] = np.ones(n_events, dtype=np.int64)
parameter_table["event_id"] = event_ids
parameter_table["tel_id"] = np.full(n_events, tel_id, dtype=np.int16)
parameter_table["intensity"] = rng.uniform(90, 140, size=n_events)
parameter_table["x"] = rng.normal(0.0, 0.05, size=n_events)
parameter_table["y"] = rng.normal(0.0, 0.05, size=n_events)
parameter_table["phi"] = rng.uniform(-np.pi, np.pi, size=n_events)
parameter_table["psi"] = rng.uniform(-np.pi, np.pi, size=n_events)
parameter_table["length"] = rng.uniform(0.05, 0.15, size=n_events)
parameter_table["length_uncertainty"] = rng.uniform(0.001, 0.003, size=n_events)
parameter_table["width"] = rng.uniform(0.02, 0.08, size=n_events)
parameter_table["width_uncertainty"] = rng.uniform(0.001, 0.003, size=n_events)
parameter_table["skewness"] = rng.normal(0.0, 0.2, size=n_events)
parameter_table["kurtosis"] = rng.normal(0.0, 0.2, size=n_events)
parameter_table["time_gradient"] = rng.normal(0.0, 0.01, size=n_events)
parameter_table["intercept"] = rng.normal(0.0, 0.01, size=n_events)
parameter_table["n_pixels"] = np.full(n_events, n_pixels, dtype=np.int16)
parameter_table["n_islands"] = np.zeros(n_events, dtype=np.int16)
parameter_table["event_type"] = np.full(n_events, 32, dtype=np.int16)
parameter_table["az_tel"] = np.full(n_events, 1.0)
parameter_table["alt_tel"] = np.full(n_events, 1.2)
parameter_table["dragon_time"] = np.linspace(1_700_000_000, 1_700_000_300, n_events)

subarray.to_hdf(mock_lst1_file, overwrite=True)
write_table(
    image_table,
    mock_lst1_file,
    "/dl1/event/telescope/image/LST_LSTCam",
    overwrite=True,
)
write_table(
    parameter_table,
    mock_lst1_file,
    "/dl1/event/telescope/parameters/LST_LSTCam",
    overwrite=True,
)

run_command_line("ctlearn-predict-LST1 -h")

######################################################################
# 1. Keras model prediction using LST-1 lstchain mock data
# --------------------------------------------------------

run_command_line(
    """ctlearn-predict-LST1 --verbose \\
        --input_url "$CTLEARN_TUTORIAL_OUTPUT_DIR/predictions/mock_data_lst1.dl1.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/predictions/mock_data_lst1.keras.ctlearn.dl2.h5" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --type_model "$CTLEARN_TUTORIAL_OUTPUT_DIR/keras-classification/ctlearn_model.keras" \\
        --energy_model "$CTLEARN_TUTORIAL_OUTPUT_DIR/keras-energy/ctlearn_model.keras" \\
        --cameradirection_model "$CTLEARN_TUTORIAL_OUTPUT_DIR/keras-cameradirection/ctlearn_model.keras" \\
        --LST1PredictionTool.batch_size 5 \\
        --LST1PredictionTool.channels cleaned_image \\
        --LST1PredictionTool.channels cleaned_relative_peak_time \\
        --LST1PredictionTool.image_mapper_type BilinearMapper \\
        --TableQualityQuery.quality_criteria "[]" \\
        --overwrite"""
)

######################################################################
# 2. PyTorch model prediction using LST-1 lstchain mock data
# ----------------------------------------------------------

run_command_line(
    """ctlearn-predict-LST1 --verbose \\
        --input_url "$CTLEARN_TUTORIAL_OUTPUT_DIR/predictions/mock_data_lst1.dl1.h5" \\
        --output "$CTLEARN_TUTORIAL_OUTPUT_DIR/predictions/mock_data_lst1.pytorch.ctlearn.dl2.h5" \\
        --config "$CTLEARN_TUTORIAL_CONFIG_DIR/dl1dh_example_config.json" \\
        --type_model "$CTLEARN_TUTORIAL_OUTPUT_DIR/pytorch-classification/ctlearn_model.pth" \\
        --energy_model "$CTLEARN_TUTORIAL_OUTPUT_DIR/pytorch-energy/ctlearn_model.pth" \\
        --cameradirection_model "$CTLEARN_TUTORIAL_OUTPUT_DIR/pytorch-cameradirection/ctlearn_model.pth" \\
        --LST1PredictionTool.batch_size 5 \\
        --LST1PredictionTool.channels cleaned_image \\
        --LST1PredictionTool.channels cleaned_relative_peak_time \\
        --LST1PredictionTool.image_mapper_type BilinearMapper \\
        --TableQualityQuery.quality_criteria "[]" \\
        --overwrite"""
)
