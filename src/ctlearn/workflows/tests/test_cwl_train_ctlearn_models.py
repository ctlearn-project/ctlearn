"""Test CTLearn model training workflows."""

import json
import shutil
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import pandas as pd
import pytest
from ctapipe.core.provenance import json_config_handler

from ctlearn.workflows.tests.utils import run_cwl_workflow

TRAINING_WORKFLOW = files("ctlearn").joinpath("workflows", "train_ctlearn_models.cwl")

DL1_MONO_TRAINING_CONFIG = {
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

DL1_STEREO_TRAINING_CONFIG = {
    "TrainCTLearnModel": {
        "model_type": "SingleCNN",
        "n_epochs": 1,
        "batch_size": 2,
        "stack_telescope_images": True,
    },
    "DLImageReader": {
        "mode": "stereo",
        "focal_length_choice": "EQUIVALENT",
        "allowed_tels": [7, 13, 15],
        "enforce_subarray_equality": False,
    },
    "TableQualityQuery": {"quality_criteria": []},
}

R1_MONO_TRAINING_CONFIG = {
    "TrainCTLearnModel": {
        "model_type": "SingleCNN",
        "n_epochs": 1,
        "batch_size": 2,
        "dl1dh_reader_type": "DLWaveformReader",
    },
    "DLWaveformReader": {
        "sequence_length": 5,
        "focal_length_choice": "EQUIVALENT",
        "enforce_subarray_equality": False,
    },
    "TableQualityQuery": {"quality_criteria": []},
}


@pytest.fixture
def dl1_mono_training_config_file(tmp_path):
    """Create a configuration for mono image training."""
    config_file = tmp_path / "dl1_mono_training_config.json"
    with config_file.open("w") as file:
        json.dump(DL1_MONO_TRAINING_CONFIG, file, default=json_config_handler, indent=4)
    return config_file


@pytest.fixture
def dl1_stereo_training_config_file(tmp_path):
    """Create a configuration for stereo image training."""
    config_file = tmp_path / "dl1_stereo_training_config.json"
    with config_file.open("w") as file:
        json.dump(
            DL1_STEREO_TRAINING_CONFIG, file, default=json_config_handler, indent=4
        )
    return config_file


@pytest.fixture
def r1_mono_training_config_file(tmp_path):
    """Create a configuration for mono waveform training."""
    config_file = tmp_path / "r1_mono_training_config.json"
    with config_file.open("w") as file:
        json.dump(R1_MONO_TRAINING_CONFIG, file, default=json_config_handler, indent=4)
    return config_file


@pytest.mark.cwl_workflow
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
def test_cwl_train_ctlearn_models_with_dl1_mono_data(
    tmp_path,
    dl1_gamma_file,
    dl1_mono_training_config_file,
    framework,
    reco_task,
    request,
):
    """Train a reconstruction model and check its training outputs."""
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
        "config_files": [{"class": "File", "path": str(dl1_mono_training_config_file)}],
        "reconstruction_task": reco_task,
        **filenames,
    }
    if reco_task == "type":
        background = tmp_path / "background"
        background.mkdir()
        shutil.copy(request.getfixturevalue("dl1_proton_file"), background)
        job["background_input"] = {"class": "Directory", "path": str(background)}
        job["pattern_background"] = ["*.dl1.h5"]
    outdir, outputs = run_cwl_workflow(tmp_path, TRAINING_WORKFLOW, job)
    training_dir = outdir / "cwl_training"
    model_path = training_dir / f"ctlearn_model.{model_suffix}"
    assert model_path.is_file()
    assert outputs["trained_model"]["basename"] == filenames["model_path"]
    model_location = urlparse(outputs["trained_model"]["location"])
    assert model_location.scheme == "file"
    assert (
        Path(url2pathname(model_location.path)).read_bytes() == model_path.read_bytes()
    )
    log_df = pd.read_csv(training_dir / "training_log.csv")
    num_epochs_logged = len(log_df)
    assert num_epochs_logged == 1, (
        f"Expected one epoch, found {num_epochs_logged} for {reco_task}"
    )
    assert "val_loss" in log_df.columns, (
        f"'val_loss' column missing in training_log.csv for {reco_task}"
    )
    val_loss = log_df["val_loss"].dropna()
    assert not val_loss.empty, f"'val_loss' column is empty for {reco_task}"
    max_val_loss = 3.5 if reco_task == "skydirection" else 1.5
    assert ((val_loss >= 0.0) & (val_loss <= max_val_loss)).all(), (
        f"'val_loss' values out of range [0.0, {max_val_loss}] for {reco_task}: "
        f"{val_loss.tolist()}"
    )
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


@pytest.mark.cwl_workflow
@pytest.mark.parametrize("framework", ["keras", "pytorch"])
@pytest.mark.parametrize(
    "reco_task",
    [
        pytest.param("energy", marks=pytest.mark.verifies_usecase("DPPS-UC-130-1.3.1")),
        pytest.param("type", marks=pytest.mark.verifies_usecase("DPPS-UC-130-1.3.2")),
        pytest.param(
            "skydirection", marks=pytest.mark.verifies_usecase("DPPS-UC-130-1.3.3")
        ),
    ],
)
def test_cwl_train_ctlearn_models_with_dl1_stereo_data(
    tmp_path,
    dl1_gamma_file,
    dl1_stereo_training_config_file,
    framework,
    reco_task,
    request,
):
    """Train a stereo model from DL1 images using the CWL's default output paths."""
    signal = tmp_path / "signal"
    signal.mkdir()
    shutil.copy(dl1_gamma_file, signal)
    job = {
        "framework": framework,
        "signal_input": {"class": "Directory", "path": str(signal)},
        "pattern_signal": ["*.dl1.h5"],
        "output_directory": "cwl_training",
        "config_files": [
            {"class": "File", "path": str(dl1_stereo_training_config_file)}
        ],
        "reconstruction_task": reco_task,
    }
    if reco_task == "type":
        background = tmp_path / "background"
        background.mkdir()
        shutil.copy(request.getfixturevalue("dl1_proton_file"), background)
        job["background_input"] = {"class": "Directory", "path": str(background)}
        job["pattern_background"] = ["*.dl1.h5"]

    outdir, outputs = run_cwl_workflow(tmp_path, TRAINING_WORKFLOW, job)
    training_dir = outdir / "cwl_training"
    location = urlparse(outputs["model_directory"]["location"])
    assert location.scheme == "file"
    assert Path(url2pathname(location.path)) == training_dir
    model_suffix = "keras" if framework == "keras" else "pth"
    assert (training_dir / f"ctlearn_model.{model_suffix}").is_file()
    saved_config = json.loads((training_dir / "config.json").read_text())
    assert saved_config["TrainCTLearnModel"]["stack_telescope_images"] is True
    assert saved_config["DLImageReader"]["mode"] == "stereo"
    log_df = pd.read_csv(training_dir / "training_log.csv")
    assert len(log_df) == 1
    assert "val_loss" in log_df.columns
    val_loss = log_df["val_loss"].dropna()
    assert not val_loss.empty
    max_val_loss = 3.5 if reco_task == "skydirection" else 1.5
    assert ((val_loss >= 0.0) & (val_loss <= max_val_loss)).all(), (
        f"'val_loss' values out of range [0.0, {max_val_loss}] for {reco_task}: "
        f"{val_loss.tolist()}"
    )
    for subfolder in ("train", "validation"):
        assert list((training_dir / subfolder).glob("events.out.tfevents.*"))
    for output in (
        "trained_model",
        "csv_summary",
        "tensorboard_train_log",
        "tensorboard_val_log",
    ):
        assert outputs[output] is None


@pytest.mark.cwl_workflow
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
def test_cwl_train_ctlearn_models_with_r1_mono_data(
    tmp_path, r1_gamma_file, r1_mono_training_config_file, framework, reco_task, request
):
    """Train a model from R1 waveforms using the CWL's default output paths."""
    signal = tmp_path / "signal"
    signal.mkdir()
    shutil.copy(r1_gamma_file, signal)

    job = {
        "framework": framework,
        "signal_input": {"class": "Directory", "path": str(signal)},
        "pattern_signal": ["*.r1.h5"],
        "output_directory": "cwl_training",
        "config_files": [{"class": "File", "path": str(r1_mono_training_config_file)}],
        "reconstruction_task": reco_task,
    }
    if reco_task == "type":
        background = tmp_path / "background"
        background.mkdir()
        shutil.copy(request.getfixturevalue("r1_proton_file"), background)
        job["background_input"] = {"class": "Directory", "path": str(background)}
        job["pattern_background"] = ["*.r1.h5"]

    outdir, outputs = run_cwl_workflow(tmp_path, TRAINING_WORKFLOW, job)
    training_dir = outdir / "cwl_training"
    location = urlparse(outputs["model_directory"]["location"])
    assert location.scheme == "file"
    assert Path(url2pathname(location.path)) == training_dir
    model_suffix = "keras" if framework == "keras" else "pth"
    assert (training_dir / f"ctlearn_model.{model_suffix}").is_file()
    log_df = pd.read_csv(training_dir / "training_log.csv")
    num_epochs_logged = len(log_df)
    assert num_epochs_logged == 1, (
        f"Expected one epoch, found {num_epochs_logged} for {reco_task}"
    )
    assert "val_loss" in log_df.columns, (
        f"'val_loss' column missing in training_log.csv for {reco_task}"
    )
    val_loss = log_df["val_loss"].dropna()
    assert not val_loss.empty, f"'val_loss' column is empty for {reco_task}"
    max_val_loss = 3.5 if reco_task == "skydirection" else 1.5
    assert ((val_loss >= 0.0) & (val_loss <= max_val_loss)).all(), (
        f"'val_loss' values out of range [0.0, {max_val_loss}] for {reco_task}: "
        f"{val_loss.tolist()}"
    )
    for subfolder in ("train", "validation"):
        assert list((training_dir / subfolder).glob("events.out.tfevents.*"))
    for output in (
        "trained_model",
        "csv_summary",
        "tensorboard_train_log",
        "tensorboard_val_log",
    ):
        assert outputs[output] is None
