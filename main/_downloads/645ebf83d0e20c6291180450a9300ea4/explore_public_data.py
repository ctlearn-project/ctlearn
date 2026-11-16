"""
**Tutorial 1** Explore Public CTAO Data
=======================================

Download public CTAO simulation data, convert it with ctapipe, and explore
waveforms and images using DL1DataHandler.

The four Zenodo files used for the stereo example total about 3.5 GB. They are
downloaded only when this script is run, and existing downloads and converted
files are reused.
"""

######################################################################
# Imports
# -------
#
# Import the tools used to download and process the public data, then read
# and visualize the resulting images and waveforms.

import os
from pathlib import Path

from ctapipe import __version__ as ctapipe_version
from ctapipe.core import run_tool
from ctapipe.tools.process import ProcessorTool
from ctapipe.utils.download import download_file
from dl1_data_handler.reader import DLImageReader, DLWaveformReader
from matplotlib import animation as ani
from matplotlib import pyplot as plt
from traitlets.config.loader import Config

try:
    from ctapipe.io.metadata import ProductMetadata
except ImportError:
    ProductMetadata = None

######################################################################
# Set a limit on the number of events (optional)
# ----------------------------------------------
#
# By default, the tutorial processes all events. To speed up a test run,
# set ``CTLEARN_TUTORIAL_MAX_EVENTS`` to a positive integer before running
# this script, for example ``CTLEARN_TUTORIAL_MAX_EVENTS=500``. The limit
# applies to each input file. Leave the variable unset to process all events.

plt.style.use("ggplot")
print_log = True

MAX_EVENTS = os.environ.get("CTLEARN_TUTORIAL_MAX_EVENTS")
if MAX_EVENTS is not None and int(MAX_EVENTS) < 1:
    raise ValueError("CTLEARN_TUTORIAL_MAX_EVENTS must be a positive integer")
MAX_EVENTS_ARGUMENT = f"--max-events={MAX_EVENTS}" if MAX_EVENTS is not None else None
MAX_EVENTS_SUFFIX = f"-max{MAX_EVENTS}" if MAX_EVENTS is not None else ""

TESTDATA_DIR = Path(
    os.environ.get(
        "CTLEARN_TUTORIAL_DATA_DIR",
        str(Path.home() / ".cache" / "ctlearn" / "tutorial-data"),
    )
)
TESTDATA_DIR.mkdir(parents=True, exist_ok=True)

######################################################################
# Download and process a small SimTel sample
# ------------------------------------------
#
# First, download a small public gamma-ray simulation and process it to
# produce calibrated waveforms and images for the monoscopic examples.

SIMTEL_URL = (
    "https://minio-cta.zeuthen.desy.de/dpps-testdata-public/data/"
    "ctapipe-test-data/v1.1.0/gamma_prod5.simtel.zst"
)
simtel_path = TESTDATA_DIR / "gamma_prod5.simtel.zst"
gamma_dl1_path = TESTDATA_DIR / f"gamma_prod5{MAX_EVENTS_SUFFIX}.r1.dl1.h5"

if not simtel_path.is_file():
    download_file(SIMTEL_URL, simtel_path, progress=True)
else:
    print(f"Using existing download: {simtel_path}")

if not gamma_dl1_path.is_file():
    argv = [
        f"--input={simtel_path}",
        f"--output={gamma_dl1_path}",
        "--write-images",
        "--DataWriter.write_r1_waveforms=True",
        "--SimTelEventSource.focal_length_choice=EQUIVALENT",
        "--overwrite",
    ]
    if MAX_EVENTS_ARGUMENT is not None:
        argv.append(MAX_EVENTS_ARGUMENT)
    if ProductMetadata is not None:
        argv.append("--ProductMetadata.instance=site_id=CTAO-South")
    result = run_tool(ProcessorTool(), argv=argv, cwd=str(TESTDATA_DIR))
    if result != 0:
        raise RuntimeError(f"ctapipe-process failed for {simtel_path}")
else:
    print(f"Using existing processed file: {gamma_dl1_path}")

######################################################################
# *Part A1* Read calibrated waveforms
# ===================================
#
# DL1DataHandler can read calibrated waveforms and map each camera into a
# regular image. Here we display an animation of the waveform samples for
# each telescope type.

telescope_types = {
    "LST": [3, 4],
    "MST_FlashCam": [8, 9, 14],
    "MST_NectarCam": [109, 112, 118, 119, 121, 122, 124],
    "SST": [43, 46, 53, 57, 92, 98],
}

image_mapper_types = {
    "LST": "BilinearMapper",
    "MST_FlashCam": "OversamplingMapper",
    "MST_NectarCam": "AxialMapper",
    "SST": "SquareMapper",
}

for tel_type, allowed_tels in telescope_types.items():
    config = Config(
        {
            "TableQualityQuery": {
                "quality_criteria": [("> 50 phe", "hillas_intensity > 50")],
            },
            "DLWaveformReader": {
                "allowed_tels": allowed_tels,
                "image_mapper_type": image_mapper_types[tel_type],
                "sequence_length": None,
                "sequence_position": "center",
                "cleaning_type": None,
                "focal_length_choice": "EQUIVALENT",
            },
        },
    )
    waveform_reader = DLWaveformReader(
        input_url_signal=[str(gamma_dl1_path)],
        config=config,
    )
    batch = waveform_reader.generate_mono_batch(0)

    if print_log:
        print(f"DLWaveformReader: {waveform_reader}")
        print(f"Subarray information for {tel_type}:")
        print(waveform_reader.subarray.info())
        print(f"Number of gamma events: {waveform_reader.n_signal_events}")
        print("First batch:")
        print("\n".join(batch.pformat(max_lines=-1, max_width=-1)))

    calibrated_waveforms = batch["features"][0]
    fig, ax = plt.subplots(figsize=(5, 5))
    camera_image = calibrated_waveforms[:, :, 0]
    image = ax.imshow(camera_image, cmap="viridis", interpolation="nearest")
    ax.grid(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("p.e.", fontsize=10)
    title = ax.set_title(f"{tel_type}: calibrated waveform (t = 0 ns)")

    def update(
        frame,
        waveform=calibrated_waveforms,
        image_artist=image,
        title_artist=title,
        telescope_type=tel_type,
    ):
        frame_image = waveform[:, :, frame]
        image_artist.set_data(frame_image)
        image_artist.set_clim(frame_image.min(), frame_image.max())
        title_artist.set_text(f"{telescope_type}: calibrated waveform (t = {frame} ns)")
        return image_artist, title_artist

    waveform_animation = ani.FuncAnimation(
        fig,
        update,
        frames=calibrated_waveforms.shape[-1],
        interval=500,
        blit=False,
    )
    # Sphinx-Gallery scrapes animations from module globals after execution.
    globals()[f"{tel_type.lower()}_waveform_animation"] = waveform_animation
    plt.show()

######################################################################
# *Part B1* Read monoscopic images
# ================================
#
# The image reader returns integrated charge and relative peak-time images.
# The mapper is selected for each camera type.

for tel_type, allowed_tels in telescope_types.items():
    config = Config(
        {
            "TableQualityQuery": {
                "quality_criteria": [("> 50 phe", "hillas_intensity > 50")],
            },
            "DLImageReader": {
                "allowed_tels": allowed_tels,
                "channels": ["image", "relative_peak_time"],
                "image_mapper_type": image_mapper_types[tel_type],
                "focal_length_choice": "EQUIVALENT",
            },
        },
    )
    image_reader = DLImageReader(
        input_url_signal=[str(gamma_dl1_path)],
        config=config,
    )
    batch = image_reader.generate_mono_batch(0)

    if print_log:
        print(f"DLImageReader: {image_reader}")
        print(f"Subarray information for {tel_type}:")
        print(image_reader.subarray.info())
        print(f"Number of gamma events: {image_reader.n_signal_events}")
        print("First batch:")
        print("\n".join(batch.pformat(max_lines=-1, max_width=-1)))

    for channel_index, channel_name, unit in [
        (0, "charge image", "p.e."),
        (1, "relative peak arrival time", "ns"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 5))
        image = ax.imshow(batch["features"][0, :, :, channel_index], cmap="viridis")
        ax.set_title(f"{tel_type} - {channel_name}")
        ax.grid(False)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(unit, fontsize=12)
        plt.show()

######################################################################
# Download and convert the stereoscopic simulated CTAO data
# ---------------------------------------------------------
#
# The Zenodo files contain CTAO open-source gamma and proton simulations.
# Convert each file with the installed ctapipe version, and store converted
# outputs in a version-specific directory so they are not confused with the
# original downloads.

ZENODO_FILES = [
    "gamma-diffuse_with_images_00.dl2.h5",
    "gamma-diffuse_with_images_01.dl2.h5",
    "proton_with_images_00.dl2.h5",
    "proton_with_images_01.dl2.h5",
]
ZENODO_FILES_URL = "https://zenodo.org/api/records/7298569/files"
converted_path = TESTDATA_DIR / f"ctapipe-{ctapipe_version}{MAX_EVENTS_SUFFIX}"
converted_path.mkdir(parents=True, exist_ok=True)

for filename in ZENODO_FILES:
    source_path = TESTDATA_DIR / filename
    if not source_path.is_file():
        download_file(
            url=f"{ZENODO_FILES_URL}/{filename}/content",
            path=source_path,
            progress=True,
        )
    else:
        print(f"Using existing download: {source_path}")

    output_path = converted_path / filename
    if not output_path.is_file():
        temporary_output = converted_path / f".{filename}.processing.h5"
        argv = [
            f"--input={source_path}",
            f"--output={temporary_output}",
            "--write-images",
            "--write-parameters",
            "--write-showers",
            "--overwrite",
        ]
        if MAX_EVENTS_ARGUMENT is not None:
            argv.append(MAX_EVENTS_ARGUMENT)
        if ProductMetadata is not None:
            argv.append("--ProductMetadata.instance=site_id=CTAO-North")
        result = run_tool(ProcessorTool(), argv=argv, cwd=str(TESTDATA_DIR))
        if result != 0:
            raise RuntimeError(f"ctapipe-process failed for {source_path}")
        temporary_output.replace(output_path)
    else:
        print(f"Using existing ctapipe output: {output_path}")

######################################################################
# *Part C1* Read stereo images
# ============================
#
# Finally, combine gamma and proton events in stereo mode and plot the
# integrated charge and relative peak-time image from each telescope.

gamma_stereo_path = converted_path / "gamma-diffuse_with_images_00.dl2.h5"
proton_stereo_path = converted_path / "proton_with_images_00.dl2.h5"

for tel_type in ["LST_LST_LSTCam", "MST_MST_NectarCam"]:
    config = Config(
        {
            "TableQualityQuery": {
                "quality_criteria": [("> 500 phe", "hillas_intensity > 500")],
            },
            "DLImageReader": {
                "mode": "stereo",
                "allowed_tel_types": [tel_type],
                "min_telescopes": 4,
                "channels": ["image", "relative_peak_time"],
                "image_mapper_type": "BilinearMapper",
            },
        },
    )
    stereo_reader = DLImageReader(
        input_url_signal=[str(gamma_stereo_path)],
        input_url_background=[str(proton_stereo_path)],
        config=config,
    )
    stereo_reader.subarray.peek()
    batch = stereo_reader.generate_stereo_batch(0)

    if print_log:
        print(f"DLImageReader: {stereo_reader}")
        print("Subarray information:")
        print(stereo_reader.subarray.info())
        print(f"Number of gamma events: {stereo_reader.n_signal_events}")
        print(f"Number of proton events: {stereo_reader.n_bkg_events}")
        print("First batch:")
        print("\n".join(batch.pformat(max_lines=-1, max_width=-1)))

    fig, axes = plt.subplots(
        2,
        len(batch["tel_id"]),
        figsize=(24, 14),
        constrained_layout=True,
        squeeze=False,
    )
    for telescope_index, tel_id in enumerate(batch["tel_id"]):
        for channel_index, label, unit in [
            (0, "Charge image", "p.e."),
            (1, "Peak arrival time", "ns"),
        ]:
            image = axes[channel_index, telescope_index].imshow(
                batch["features"][telescope_index, :, :, channel_index],
                cmap="viridis",
            )
            axes[channel_index, telescope_index].set_title(f"Tel-{tel_id} - {label}")
            axes[channel_index, telescope_index].grid(False)
            colorbar = fig.colorbar(
                image,
                ax=axes[channel_index, telescope_index],
                fraction=0.046,
                pad=0.04,
            )
            colorbar.set_label(unit, fontsize=12)
    plt.show()
