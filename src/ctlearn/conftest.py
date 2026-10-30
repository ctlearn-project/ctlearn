"""
common pytest fixtures for tests in ctlearn.
"""

import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
import tables
from astropy import units as u
from astropy.table import Column, Table
from ctapipe.core import run_tool
from ctapipe.io import read_table, write_table
from ctapipe.io.hdf5dataformat import (
    DL0_TEL_POINTING_GROUP,
    DL1_TEL_TRIGGER_TABLE,
    FIXED_POINTING_GROUP,
    SIMULATION_GROUP,
    SIMULATION_RUN_TABLE,
)
from filelock import FileLock
from traitlets.config.loader import Config

# TODO: Bump minimum ctapipe version to v0.33.0. Clean up imports!
try:
    from ctapipe.io.metadata import ProductMetadata as _ProductMetadata
except ImportError:
    _ProductMetadata = None
try:
    from ctapipe.io.metadata import read_ctao_metadata, write_product_metadata_hdf5
except ImportError:
    read_ctao_metadata = None
from ctapipe.tools.process import ProcessorTool
from ctapipe.utils import get_dataset_path

from ctlearn.tools.keras.train_model import TrainCTLearnKerasModel
from ctlearn.tools.pytorch.train_model import TrainCTLearnPyTorchModel
from ctlearn.tools.utils import get_lst1_subarray_description

TRAINING_TOOLS = {"Keras": TrainCTLearnKerasModel, "PyTorch": TrainCTLearnPyTorchModel}
MODEL_FILE_FORMATS = {"Keras": "keras", "PyTorch": "pth"}


@pytest.fixture(scope="session")
def gamma_simtel_path(tmp_path_factory):
    root_tmp_dir = tmp_path_factory.getbasetemp().parent
    with FileLock(str(root_tmp_dir / "gamma_test_large.simtel.gz.lock")):
        return get_dataset_path("gamma_test_large.simtel.gz")


@pytest.fixture(scope="session")
def proton_simtel_path(tmp_path_factory):
    root_tmp_dir = tmp_path_factory.getbasetemp().parent
    with FileLock(str(root_tmp_dir / "proton_test.simtel.zst.lock")):
        return get_dataset_path(
            "proton_20deg_0deg_run4___cta-prod5-paranal_desert-2147m-Paranal-dark-100evts.simtel.zst"
        )


@pytest.fixture(scope="session")
def dl1_tmp_path(tmp_path_factory):
    """Temporary directory for global dl1 test data"""
    return tmp_path_factory.mktemp("dl1_")


@pytest.fixture(scope="session")
def r1_tmp_path(tmp_path_factory):
    """Temporary directory for global r1 test data"""
    return tmp_path_factory.mktemp("r1_")


def _create_mock_lst1_dl1_file(
    output_path: str | Path,
    n_events: int = 4,
    random_seed: int | None = 0,
) -> Path:
    """Write a minimal DL1 file with fake data compatible with the LST1PredictionTool."""

    rng = np.random.default_rng(random_seed)
    output_path = Path(output_path)

    subarray = get_lst1_subarray_description()
    tel_id = 1
    n_pixels = subarray.tel[tel_id].camera.geometry.n_pixels

    obs_id = 1
    event_ids = np.arange(1, n_events + 1, dtype=np.int64)

    # Fake per-pixel data
    image = rng.uniform(80, 150, size=(n_events, n_pixels)).astype(np.float32)
    image_mask = rng.integers(0, 2, size=(n_events, n_pixels), dtype=bool)
    peak_time = rng.normal(5.0, 0.5, size=(n_events, n_pixels)).astype(np.float32)

    image_table = Table()
    image_table["obs_id"] = np.full(n_events, obs_id, dtype=np.int64)
    image_table["event_id"] = event_ids
    image_table["tel_id"] = np.full(n_events, tel_id, dtype=np.int16)
    image_table.add_column(
        Column(image, name="image", dtype=np.float32, shape=(n_pixels,))
    )
    image_table.add_column(
        Column(image_mask, name="image_mask", dtype=bool, shape=(n_pixels,))
    )
    image_table.add_column(
        Column(peak_time, name="peak_time", dtype=np.float32, shape=(n_pixels,))
    )

    # DL1 parameter columns required by LST1PredictionTool
    parameter_table = Table()
    parameter_table["obs_id"] = np.full(n_events, obs_id, dtype=np.int64)
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

    # Write to DL1 file the subarray description, image and parameter tables
    subarray.to_hdf(output_path, overwrite=True)
    write_table(
        image_table,
        output_path,
        "/dl1/event/telescope/image/LST_LSTCam",
        overwrite=True,
    )
    write_table(
        parameter_table,
        output_path,
        "/dl1/event/telescope/parameters/LST_LSTCam",
        overwrite=True,
    )
    with tables.open_file(output_path, mode="a") as output_file:
        image_table_node = output_file.get_node("/dl1/event/telescope/image/LST_LSTCam")
        image_attrs = image_table_node._v_attrs
        image_attrs.CTAFIELD_3_TRANSFORM_SCALE = 0.0
        image_attrs.CTAFIELD_3_TRANSFORM_OFFSET = 0
        image_attrs.CTAFIELD_4_TRANSFORM_SCALE = 0.0
        image_attrs.CTAFIELD_4_TRANSFORM_OFFSET = 0
        image_table_node.flush()
    return output_path


@pytest.fixture(scope="session")
def mock_lst1_dl1_file(tmp_path_factory):
    """Path to a session-scoped mock LST-1 DL1 HDF5 file for tests."""

    output = tmp_path_factory.mktemp("mock_lst1") / "mock_lst1_dl1.h5"
    return _create_mock_lst1_dl1_file(output)


@pytest.fixture(scope="session")
def dl1_gamma_tel2_file(dl1_tmp_path, gamma_simtel_path):
    """
    DL1 file containing both images and parameters from a gamma simulation set.
    """
    output = dl1_tmp_path / "gamma_tel2.dl1.h5"
    argv = [
        f"--input={gamma_simtel_path}",
        f"--output={output}",
        "--allowed-tels=2",
        "--max-events=10",
        "--write-images",
        "--SimTelEventSource.focal_length_choice=EQUIVALENT",
    ]
    if _ProductMetadata is not None:
        argv.append("--ProductMetadata.instance=site_id=CTAO-South")
    assert run_tool(ProcessorTool(), argv=argv, cwd=dl1_tmp_path) == 0
    return output


@pytest.fixture(scope="session")
def dl1_gamma_file(dl1_tmp_path, gamma_simtel_path):
    """
    DL1 file containing both images and parameters from a gamma simulation set.
    """
    output = dl1_tmp_path / "gamma.dl1.h5"
    argv = [
        f"--input={gamma_simtel_path}",
        f"--output={output}",
        "--write-images",
        "--SimTelEventSource.focal_length_choice=EQUIVALENT",
    ]
    if _ProductMetadata is not None:
        argv.append("--ProductMetadata.instance=site_id=CTAO-South")
    assert run_tool(ProcessorTool(), argv=argv, cwd=dl1_tmp_path) == 0
    return output


@pytest.fixture(scope="session")
def dl1_obs_mock_file(dl1_tmp_path, dl1_gamma_file):
    """
    DL1 file containing both images and parameters from a mock observation.
    """
    # Hardcopy the DL1 gamma file to the output path to avoid modifying the original file
    output = dl1_tmp_path / "dl1_obs_mock_file.dl1.h5"
    shutil.copy(dl1_gamma_file, output)
    # Read the trigger table to get the unique telescope IDs and their pointing information
    trigger_table = read_table(dl1_gamma_file, path=DL1_TEL_TRIGGER_TABLE)
    tel_ids = np.unique(trigger_table["tel_id"])
    time = trigger_table["time"][[0, -1]]
    # Mimic an observation of 20 mins duration by adjusting the time range for the tel pointing tables
    time[0] -= 20 * u.min
    for tel_id in tel_ids:
        tel_pointing = read_table(
            dl1_gamma_file, path=f"{FIXED_POINTING_GROUP}/tel_{tel_id:03d}"
        )
        azimuth = tel_pointing["telescope_pointing_azimuth"][0]
        altitude = tel_pointing["telescope_pointing_altitude"][0]
        tel_pointing_table = Table(
            {
                "time": time,
                "azimuth": [azimuth, azimuth] * u.rad,
                "altitude": [altitude, altitude] * u.rad,
            }
        )
        write_table(
            tel_pointing_table,
            output,
            f"{DL0_TEL_POINTING_GROUP}/tel_{tel_id:03d}",
        )
    # Modify the file to mimic an observation file
    # TODO: Bump minimum ctapipe version to v0.33.0. Clean up here!
    if read_ctao_metadata is not None:
        ctao_metadata = read_ctao_metadata(output)
        ctao_metadata.data.type = "Observation"
        with tables.open_file(output, mode="r+") as h5_file:
            write_product_metadata_hdf5(
                product=ctao_metadata,
                h5file=h5_file,
                remove_legacy=True,
            )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", tables.NaturalNameWarning)
        with tables.open_file(output, mode="r+") as h5_file:
            # TODO: Bump minimum ctapipe version to v0.33.0. Clean up here!
            if read_ctao_metadata is None:
                # Update CTA Activity metadata to mimic an observation file
                if "data.type" in h5_file.root._v_attrs:
                    h5_file.root._v_attrs["data.type"] = "Observation"
                elif "CTA PROCESS TYPE" in h5_file.root._v_attrs:
                    h5_file.root._v_attrs["CTA PROCESS TYPE"] = "Observation"
                else:
                    raise IOError("Unsupported or missing reference metadata")
                h5_file.flush()
            # Remove simulation-related group to mimic an observation file
            pointing_group = FIXED_POINTING_GROUP.rsplit("/", maxsplit=1)[0]
            simulation_run_group = SIMULATION_RUN_TABLE.rsplit("/", maxsplit=1)[0]
            for group in [SIMULATION_GROUP, simulation_run_group, pointing_group]:
                if group in h5_file:
                    h5_file.remove_node(group, recursive=True)
    return output


@pytest.fixture(scope="session")
def dl1_proton_file(dl1_tmp_path, proton_simtel_path):
    """
    DL1 file containing both images and parameters from a proton simulation set.
    """
    from ctapipe.tools.process import ProcessorTool

    output = dl1_tmp_path / "proton.dl1.h5"
    argv = [
        f"--input={proton_simtel_path}",
        f"--output={output}",
        "--write-images",
        "--SimTelEventSource.focal_length_choice=EQUIVALENT",
    ]
    if _ProductMetadata is not None:
        argv.append("--ProductMetadata.instance=site_id=CTAO-South")
    assert run_tool(ProcessorTool(), argv=argv, cwd=dl1_tmp_path) == 0
    return output


@pytest.fixture(scope="session")
def r1_gamma_file(r1_tmp_path, gamma_simtel_path):
    """
    R1 file containing both waveforms and parameters from a gamma simulation set.
    """
    output = r1_tmp_path / "gamma.r1.h5"
    allowed_tels = [1, 2]
    argv = [
        f"--input={gamma_simtel_path}",
        f"--output={output}",
        "--DataWriter.write_r1_waveforms=True",
        "--SimTelEventSource.focal_length_choice=EQUIVALENT",
        f"--SimTelEventSource.allowed_tels={allowed_tels}",
    ]
    if _ProductMetadata is not None:
        argv.append("--ProductMetadata.instance=site_id=CTAO-South")
    assert run_tool(ProcessorTool(), argv=argv, cwd=r1_tmp_path) == 0
    return output


@pytest.fixture(scope="session")
def r1_proton_file(r1_tmp_path, proton_simtel_path):
    """
    R1 file containing both waveforms and parameters from a proton simulation set.
    """
    # Restrict to two LSTs for R1 tests to reduce computational load
    allowed_tels = [1, 2]
    output = r1_tmp_path / "proton.r1.h5"
    argv = [
        f"--input={proton_simtel_path}",
        f"--output={output}",
        "--DataWriter.write_r1_waveforms=True",
        "--SimTelEventSource.focal_length_choice=EQUIVALENT",
        f"--SimTelEventSource.allowed_tels={allowed_tels}",
    ]
    if _ProductMetadata is not None:
        argv.append("--ProductMetadata.instance=site_id=CTAO-South")
    assert run_tool(ProcessorTool(), argv=argv, cwd=r1_tmp_path) == 0
    return output


@pytest.fixture(scope="session")
def ctlearn_trained_r1_mono_models(r1_gamma_file, r1_proton_file, tmp_path_factory):
    """
    Test training CTLearn model using the R1 gamma and proton files for all reconstruction tasks.
    Each test run gets its own isolated temp directories.
    """
    tmp_path = tmp_path_factory.mktemp("ctlearn_mono_models")

    # Temporary directories for signal and background
    signal_dir = tmp_path / "gamma_r1"
    signal_dir.mkdir(parents=True, exist_ok=True)

    background_dir = tmp_path / "proton_r1"
    background_dir.mkdir(parents=True, exist_ok=True)

    # Hardcopy R1 gamma file to the signal directory
    shutil.copy(r1_gamma_file, signal_dir)
    # Hardcopy R1 proton file to the background directory
    shutil.copy(r1_proton_file, background_dir)

    # Configuration to disable quality cuts to increase
    # training statistics mainly needed for LSTs.
    config = Config(
        {
            "TableQualityQuery": {
                "quality_criteria": [],
            },
        },
    )

    # Restrict to two LSTs for R1 tests to reduce computational load
    telescope_type = "LST"
    # Loop over reconstruction tasks and train models for each combination
    ctlearn_trained_r1_mono_models = {}
    for reco_task in ["type", "energy", "cameradirection"]:
        # Build command-line arguments
        argv = [
            f"--signal={signal_dir}",
            "--pattern-signal=*.r1.h5",
            f"--reco={reco_task}",
            "--TrainCTLearnModel.n_epochs=1",
            "--TrainCTLearnModel.batch_size=2",
            "--TrainCTLearnModel.dl1dh_reader_type=DLWaveformReader",
            "--DLWaveformReader.sequence_length=5",
            "--DLWaveformReader.focal_length_choice=EQUIVALENT",
        ]

        # Include background only for classification task
        if reco_task == "type":
            argv.extend(
                [
                    f"--background={background_dir}",
                    "--pattern-background=*.r1.h5",
                    "--DLWaveformReader.enforce_subarray_equality=False",
                ]
            )

        # Run training tools
        for framework, training_tool in TRAINING_TOOLS.items():
            framework_argv = argv.copy()
            output_dir = tmp_path / f"ctlearn_{framework}_{telescope_type}_{reco_task}"
            framework_argv.append(f"--output={output_dir}")
            if framework == "PyTorch":
                framework_argv.append("--TrainCTLearnPyTorchModel.num_workers=2")

            assert (
                run_tool(
                    training_tool(config=config), argv=framework_argv, cwd=tmp_path
                )
                == 0
            )
            ctlearn_trained_r1_mono_models[
                f"{framework}_{telescope_type}_{reco_task}"
            ] = output_dir / f"ctlearn_model.{MODEL_FILE_FORMATS[framework]}"
            # Check that the trained model exists
            assert ctlearn_trained_r1_mono_models[
                f"{framework}_{telescope_type}_{reco_task}"
            ].exists()
    return ctlearn_trained_r1_mono_models


@pytest.fixture(scope="session")
def ctlearn_trained_dl1_mono_models(dl1_gamma_file, dl1_proton_file, tmp_path_factory):
    """
    Test training CTLearn model using the DL1 gamma and proton files for all reconstruction tasks.
    Each test run gets its own isolated temp directories.
    """
    tmp_path = tmp_path_factory.mktemp("ctlearn_mono_models")

    # Temporary directories for signal and background
    signal_dir = tmp_path / "gamma_dl1"
    signal_dir.mkdir(parents=True, exist_ok=True)

    background_dir = tmp_path / "proton_dl1"
    background_dir.mkdir(parents=True, exist_ok=True)

    # Hardcopy DL1 gamma file to the signal directory
    shutil.copy(dl1_gamma_file, signal_dir)
    # Hardcopy DL1 proton file to the background directory
    shutil.copy(dl1_proton_file, background_dir)

    # Configuration to disable quality cuts to increase
    # training statistics mainly needed for LSTs.
    config = Config(
        {
            "TableQualityQuery": {
                "quality_criteria": [],
            },
        },
    )

    # Define telescope types and their allowed telescopes
    telescope_types = {
        "LST": [1, 2],
        "MST": [7, 13, 15, 16, 17, 19],
        # "SST": [30, 37, 43, 44, 53],
    }
    image_mapper_types = {
        "LST": "BilinearMapper",
        "MST": "OversamplingMapper",
        # "SST": "SquareMapper",
    }
    # Loop over telescope types and reconstruction tasks
    # and train models for each combination
    ctlearn_trained_dl1_mono_models = {}
    for telescope_type, allowed_tels in telescope_types.items():
        for reco_task in ["type", "energy", "cameradirection"]:
            # Build command-line arguments
            argv = [
                f"--signal={signal_dir}",
                "--pattern-signal=*.dl1.h5",
                f"--reco={reco_task}",
                "--TrainCTLearnModel.n_epochs=1",
                "--TrainCTLearnModel.batch_size=2",
                "--DLImageReader.focal_length_choice=EQUIVALENT",
                f"--DLImageReader.allowed_tels={allowed_tels}",
            ]
            # Include background only for classification task
            if reco_task == "type":
                argv.extend(
                    [
                        f"--background={background_dir}",
                        "--pattern-background=*.dl1.h5",
                        "--DLImageReader.enforce_subarray_equality=False",
                        f"--DLImageReader.image_mapper_type={image_mapper_types[telescope_type]}",
                    ]
                )
            # Run training tools
            for framework, training_tool in TRAINING_TOOLS.items():
                framework_argv = argv.copy()
                output_dir = (
                    tmp_path / f"ctlearn_{framework}_{telescope_type}_{reco_task}"
                )
                framework_argv.append(f"--output={output_dir}")
                assert (
                    run_tool(
                        training_tool(config=config), argv=framework_argv, cwd=tmp_path
                    )
                    == 0
                )
                ctlearn_trained_dl1_mono_models[
                    f"{framework}_{telescope_type}_{reco_task}"
                ] = output_dir / f"ctlearn_model.{MODEL_FILE_FORMATS[framework]}"
                # Check that the trained model exists
                assert ctlearn_trained_dl1_mono_models[
                    f"{framework}_{telescope_type}_{reco_task}"
                ].exists()
    return ctlearn_trained_dl1_mono_models


@pytest.fixture(scope="session")
def ctlearn_trained_dl1_stereo_models(
    dl1_gamma_file, dl1_proton_file, tmp_path_factory
):
    """
    Test training CTLearn model using the R1 gamma and proton files for all reconstruction tasks.
    Each test run gets its own isolated temp directories.
    """
    tmp_path = tmp_path_factory.mktemp("ctlearn_stereo_models")

    # Temporary directories for signal and background
    signal_dir = tmp_path / "gamma_dl1"
    signal_dir.mkdir(parents=True, exist_ok=True)

    background_dir = tmp_path / "proton_dl1"
    background_dir.mkdir(parents=True, exist_ok=True)

    # Hardcopy DL1 gamma file to the signal directory
    shutil.copy(dl1_gamma_file, signal_dir)
    # Hardcopy DL1 proton file to the background directory
    shutil.copy(dl1_proton_file, background_dir)

    # Configuration to disable quality cuts to increase
    # training statistics mainly needed for LSTs.
    config = Config(
        {
            "TableQualityQuery": {
                "quality_criteria": [],
            },
        },
    )

    # Restrict to three MSTs for DL1 tests to reduce computational load
    telescope_type = "MST"
    allowed_tels = [7, 13, 15]

    # Loop over reconstruction tasks and train models for each combination
    ctlearn_trained_dl1_stereo_models = {}
    for reco_task in ["type", "energy", "skydirection"]:
        # Build command-line arguments
        argv = [
            f"--signal={signal_dir}",
            "--pattern-signal=*.dl1.h5",
            f"--reco={reco_task}",
            "--TrainCTLearnModel.n_epochs=1",
            "--TrainCTLearnModel.batch_size=2",
            "--TrainCTLearnModel.stack_telescope_images=True",
            "--DLImageReader.mode=stereo",
            "--DLImageReader.focal_length_choice=EQUIVALENT",
            f"--DLImageReader.allowed_tels={allowed_tels}",
        ]

        # Include background only for classification task
        if reco_task == "type":
            argv.extend(
                [
                    f"--background={background_dir}",
                    "--pattern-background=*.dl1.h5",
                    "--DLImageReader.enforce_subarray_equality=False",
                ]
            )

        # Run training tools
        for framework, training_tool in TRAINING_TOOLS.items():
            framework_argv = argv.copy()
            output_dir = tmp_path / f"ctlearn_{framework}_{telescope_type}_{reco_task}"
            framework_argv.append(f"--output={output_dir}")
            assert (
                run_tool(
                    training_tool(config=config), argv=framework_argv, cwd=tmp_path
                )
                == 0
            )
            ctlearn_trained_dl1_stereo_models[
                f"{framework}_{telescope_type}_{reco_task}"
            ] = output_dir / f"ctlearn_model.{MODEL_FILE_FORMATS[framework]}"
            # Check that the trained model exists
            assert ctlearn_trained_dl1_stereo_models[
                f"{framework}_{telescope_type}_{reco_task}"
            ].exists()
    return ctlearn_trained_dl1_stereo_models
