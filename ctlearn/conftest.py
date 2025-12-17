"""
common pytest fixtures for tests in ctlearn.
"""

import pytest
from ctapipe.core import run_tool
from ctapipe.utils import get_dataset_path

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

    argv = [
        f"--input={gamma_simtel_path}",
        f"--output={output}",
        f"--DataWriter.write_r1_waveforms=True",
        "--SimTelEventSource.focal_length_choice=EQUIVALENT",
    ]
    assert run_tool(ProcessorTool(), argv=argv, cwd=r1_tmp_path) == 0
    return output