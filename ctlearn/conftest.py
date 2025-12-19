"""
common pytest fixtures for tests in ctlearn.
"""

import pytest
import shutil

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

    allowed_tels = {7, 13, 15, 16, 17, 19}
    output = dl1_tmp_path / "gamma.dl1.h5"
    argv = [
        f"--input={gamma_simtel_path}",
        f"--output={output}",
        "--write-images",
        "--SimTelEventSource.focal_length_choice=EQUIVALENT",
        f"--SimTelEventSource.allowed_tels={allowed_tels}",
    ]
    assert run_tool(ProcessorTool(), argv=argv, cwd=dl1_tmp_path) == 0
    return output

@pytest.fixture(scope="session")
def dl1_proton_file(dl1_tmp_path, proton_simtel_path):
    """
    DL1 file containing both images and parameters from a proton simulation set.
    """
    from ctapipe.tools.process import ProcessorTool

    allowed_tels = {7, 13, 15, 16, 17, 19}
    output = dl1_tmp_path / "proton.dl1.h5"
    argv = [
        f"--input={proton_simtel_path}",
        f"--output={output}",
        "--write-images",
        "--SimTelEventSource.focal_length_choice=EQUIVALENT",
        f"--SimTelEventSource.allowed_tels={allowed_tels}",
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

    argv = [
        f"--input={gamma_simtel_path}",
        f"--output={output}",
        f"--DataWriter.write_r1_waveforms=True",
        "--SimTelEventSource.focal_length_choice=EQUIVALENT",
    ]
    assert run_tool(ProcessorTool(), argv=argv, cwd=r1_tmp_path) == 0
    return output

@pytest.fixture(scope="session")
def ctlearn_trained_dl1_models(dl1_gamma_file, dl1_proton_file, tmp_path_factory):
    """
    Test training CTLearn model using the DL1 gamma and proton files for all reconstruction tasks.
    Each test run gets its own isolated temp directories.
    """
    tmp_path = tmp_path_factory.mktemp("ctlearn_models")

    # Temporary directories for signal and background
    signal_dir = tmp_path / "gamma_dl1"
    signal_dir.mkdir(parents=True, exist_ok=True)

    background_dir = tmp_path / "proton_dl1"
    background_dir.mkdir(parents=True, exist_ok=True)

    # Hardcopy DL1 gamma file to the signal directory
    shutil.copy(dl1_gamma_file, signal_dir)
    # Hardcopy DL1 proton file to the background directory
    shutil.copy(dl1_proton_file, background_dir)

    ctlearn_trained_dl1_models = {}
    for reco_task in ["type", "energy", "cameradirection"]:
        # Output directory for trained model
        output_dir = tmp_path / f"ctlearn_{reco_task}"
        
        # Build command-line arguments
        argv = [
            f"--signal={signal_dir}",
            "--pattern-signal=*.dl1.h5",
            f"--output={output_dir}",
            f"--reco={reco_task}",
            "--TrainCTLearnModel.n_epochs=1",
            "--TrainCTLearnModel.batch_size=2",
            "--DLImageReader.focal_length_choice=EQUIVALENT",
        ]

        # Include background only for classification task
        if reco_task == "type":
            argv.extend([
                f"--background={background_dir}",
                "--pattern-background=*.dl1.h5",
                "--DLImageReader.enforce_subarray_equality=False",
            ])

        # Run training
        assert run_tool(TrainCTLearnModel(), argv=argv, cwd=tmp_path) == 0

        ctlearn_trained_dl1_models[reco_task] = output_dir / "ctlearn_model.keras"
        # Check that the trained model exists
        assert ctlearn_trained_dl1_models[reco_task].exists()
    return ctlearn_trained_dl1_models