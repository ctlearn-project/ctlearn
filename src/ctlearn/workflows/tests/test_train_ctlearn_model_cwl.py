"""Test CTLearn model training workflows."""

import csv
import json
import os
import shutil
import subprocess
import sys
from importlib.resources import as_file, files
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest
from ctapipe.core.provenance import json_config_handler


@pytest.fixture
def training_config_file(tmp_path):
    """Create a configuration for a small training run."""
    config_file = tmp_path / "example_training_config.json"
    config = {
        "TrainCTLearnModel": {
            "model_type": "SingleCNN",
            "n_epochs": 1,
            "batch_size": 4,
        },
        "DLImageReader": {
            "focal_length_choice": "EQUIVALENT",
            "allowed_tels": [7, 13, 15, 16, 17, 19],
            "enforce_subarray_equality": False,
        },
    }
    with config_file.open("w") as file:
        json.dump(config, file, default=json_config_handler, indent=4)
    return config_file


@pytest.mark.parametrize("framework", ["keras", "pytorch"])
@pytest.mark.parametrize(
    "reco_task",
    [
        pytest.param("energy", marks=pytest.mark.verifies_usecase("DPPS-UC-130-1.3.1")),
        pytest.param("type", marks=pytest.mark.verifies_usecase("DPPS-UC-130-1.3.2")),
        pytest.param(
            "cameradirection", marks=pytest.mark.verifies_usecase("DPPS-UC-130-1.3.3")
        ),
        pytest.param(
            "skydirection", marks=pytest.mark.verifies_usecase("DPPS-UC-130-1.3.3")
        ),
    ],
)
def test_train_ctlearn_model(
    tmp_path,
    dl1_gamma_file,
    training_config_file,
    framework,
    reco_task,
    request,
):
    """Train a reconstruction model and check its training outputs."""
    cwltool = shutil.which("cwltool")
    assert cwltool is not None, "cwltool is required for workflow tests"

    signal = tmp_path / "signal"
    signal.mkdir()
    shutil.copy(dl1_gamma_file, signal)
    model_suffix = "keras" if framework == "keras" else "pth"
    filenames = {
        "model_path": f"ctlearn_model_custom_path.{model_suffix}",
        "csv_summary_filename": "summary.csv",
        "tensorboard_train_log_filename": "train/tfevents.train",
        "tensorboard_val_log_filename": "validation/tfevents.val",
    }
    job = {
        "framework": framework,
        "signal_input": {"class": "Directory", "path": str(signal)},
        "pattern_signal": ["*.dl1.h5"],
        "output_directory": "cwl_training",
        "config_files": [{"class": "File", "path": str(training_config_file)}],
        "reconstruction_task": reco_task,
        **filenames,
    }
    if reco_task == "type":
        background = tmp_path / "background"
        background.mkdir()
        shutil.copy(request.getfixturevalue("dl1_proton_file"), background)
        job["background_input"] = {"class": "Directory", "path": str(background)}
        job["pattern_background"] = ["*.dl1.h5"]
    job_path = tmp_path / "cwltool_job.json"
    with job_path.open("w") as file:
        json.dump(job, file, default=json_config_handler, indent=4)
    outdir = tmp_path / "results"
    env = os.environ.copy()
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env['PATH']}"
    workflow = files("ctlearn").joinpath("workflows", "train_ctlearn_model.cwl")
    with as_file(workflow) as workflow_path:
        result = subprocess.run(
            [
                cwltool,
                "--no-container",
                "--outdir",
                str(outdir),
                str(workflow_path),
                str(job_path),
            ],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
    assert result.returncode == 0, result.stderr
    outputs = json.loads(result.stdout)
    training_dir = outdir / "cwl_training"
    model_path = training_dir / f"ctlearn_model.{model_suffix}"
    assert model_path.is_file()
    assert outputs["trained_model"]["basename"] == filenames["model_path"]
    model_location = urlparse(outputs["trained_model"]["location"])
    assert model_location.scheme == "file"
    assert (
        Path(url2pathname(model_location.path)).read_bytes() == model_path.read_bytes()
    )
    with (training_dir / "training_log.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["val_loss"]

    for output, input_name, original in (
        ("csv_summary", "csv_summary_filename", training_dir / "training_log.csv"),
        (
            "tensorboard_train_log",
            "tensorboard_train_log_filename",
            next((training_dir / "train").glob("events.out.tfevents.*")),
        ),
        (
            "tensorboard_val_log",
            "tensorboard_val_log_filename",
            next((training_dir / "validation").glob("events.out.tfevents.*")),
        ),
    ):
        assert outputs[output]["basename"] == Path(filenames[input_name]).name
        location = urlparse(outputs[output]["location"])
        assert location.scheme == "file"
        copied_file = Path(url2pathname(location.path))
        assert copied_file.read_bytes() == original.read_bytes()
