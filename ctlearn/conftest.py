"""
common pytest fixtures for tests in ctlearn.
"""

import pytest
import shutil
from traitlets.config.loader import Config

from ctapipe.core import run_tool
from ctapipe.utils import get_dataset_path
from ctlearn.tools import TrainCTLearnModel


@pytest.fixture(scope="session")
def gamma_simtel_path():
    return get_dataset_path("gamma_test_large.simtel.gz")


@pytest.fixture(scope="session")
def proton_simtel_path():
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


@pytest.fixture(scope="session")
def dl1_gamma_file(dl1_tmp_path, gamma_simtel_path):
    """
    DL1 file containing both images and parameters from a gamma simulation set.
    """
    from ctapipe.tools.process import ProcessorTool

    output = dl1_tmp_path / "gamma.dl1.h5"
    argv = [
        f"--input={gamma_simtel_path}",
        f"--output={output}",
        "--write-images",
        "--SimTelEventSource.focal_length_choice=EQUIVALENT",
    ]
    assert run_tool(ProcessorTool(), argv=argv, cwd=dl1_tmp_path) == 0
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
    assert run_tool(ProcessorTool(), argv=argv, cwd=dl1_tmp_path) == 0
    return output


@pytest.fixture(scope="session")
def r1_gamma_file(r1_tmp_path, gamma_simtel_path):
    """
    R1 file containing both waveforms and parameters from a gamma simulation set.
    """
    from ctapipe.tools.process import ProcessorTool

    output = r1_tmp_path / "gamma.r1.h5"

    allowed_tels = [1, 2]
    argv = [
        f"--input={gamma_simtel_path}",
        f"--output={output}",
        f"--DataWriter.write_r1_waveforms=True",
        "--SimTelEventSource.focal_length_choice=EQUIVALENT",
        f"--SimTelEventSource.allowed_tels={allowed_tels}",
    ]
    assert run_tool(ProcessorTool(), argv=argv, cwd=r1_tmp_path) == 0
    return output


@pytest.fixture(scope="session")
def r1_proton_file(r1_tmp_path, proton_simtel_path):
    """
    R1 file containing both waveforms and parameters from a proton simulation set.
    """
    from ctapipe.tools.process import ProcessorTool

    # Restrict to two LSTs for R1 tests to reduce computational load
    allowed_tels = [1, 2]
    output = r1_tmp_path / "proton.r1.h5"
    argv = [
        f"--input={proton_simtel_path}",
        f"--output={output}",
        f"--DataWriter.write_r1_waveforms=True",
        "--SimTelEventSource.focal_length_choice=EQUIVALENT",
        f"--SimTelEventSource.allowed_tels={allowed_tels}",
    ]
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
        # Output directory for trained model
        output_dir = tmp_path / f"ctlearn_{telescope_type}_{reco_task}"

        # Build command-line arguments
        argv = [
            f"--signal={signal_dir}",
            "--pattern-signal=*.r1.h5",
            f"--output={output_dir}",
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

        # Run training
        assert run_tool(TrainCTLearnModel(config=config), argv=argv, cwd=tmp_path) == 0

        ctlearn_trained_r1_mono_models[f"{telescope_type}_{reco_task}"] = (
            output_dir / "ctlearn_model.keras"
        )
        # Check that the trained model exists
        assert ctlearn_trained_r1_mono_models[f"{telescope_type}_{reco_task}"].exists()
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
            # Output directory for trained model
            output_dir = tmp_path / f"ctlearn_{telescope_type}_{reco_task}"

            # Build command-line arguments
            argv = [
                f"--signal={signal_dir}",
                "--pattern-signal=*.dl1.h5",
                f"--output={output_dir}",
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

            # Run training
            assert (
                run_tool(TrainCTLearnModel(config=config), argv=argv, cwd=tmp_path) == 0
            )

            ctlearn_trained_dl1_mono_models[f"{telescope_type}_{reco_task}"] = (
                output_dir / "ctlearn_model.keras"
            )
            # Check that the trained model exists
            assert ctlearn_trained_dl1_mono_models[
                f"{telescope_type}_{reco_task}"
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
        # Output directory for trained model
        output_dir = tmp_path / f"ctlearn_{telescope_type}_{reco_task}"

        # Build command-line arguments
        argv = [
            f"--signal={signal_dir}",
            "--pattern-signal=*.dl1.h5",
            f"--output={output_dir}",
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

        # Run training
        assert run_tool(TrainCTLearnModel(config=config), argv=argv, cwd=tmp_path) == 0

        ctlearn_trained_dl1_stereo_models[f"{telescope_type}_{reco_task}"] = (
            output_dir / "ctlearn_model.keras"
        )
        # Check that the trained model exists
        assert ctlearn_trained_dl1_stereo_models[
            f"{telescope_type}_{reco_task}"
        ].exists()
    return ctlearn_trained_dl1_stereo_models
