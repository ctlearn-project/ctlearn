import pandas as pd
import pytest
import shutil

from ctapipe.core import run_tool
from ctlearn.conftest import TRAINING_TOOLS, MODEL_FILE_FORMATS

@pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
@pytest.mark.parametrize("model", ["SingleCNN", "ResNet", "LoadedModel"])
@pytest.mark.parametrize("reco_task", ["type", "energy", "cameradirection"])
def test_train_ctlearn_model(framework, model, reco_task, dl1_gamma_file, dl1_proton_file, ctlearn_trained_dl1_mono_models, tmp_path):
    """
    Test training CTLearn model using the DL1 gamma and proton files for all reconstruction tasks.
    Each test run gets its own isolated temp directories.
    """
    # Restrict to MST array
    telescope_type = "MST"
    allowed_tels = [7, 13, 15, 16, 17, 19]
    # Temporary directories for signal and background
    signal_dir = tmp_path / "gamma_dl1"
    signal_dir.mkdir(parents=True, exist_ok=True)
    background_dir = tmp_path / "proton_dl1"
    background_dir.mkdir(parents=True, exist_ok=True)
    # Hardcopy DL1 gamma file to the signal directory
    shutil.copy(dl1_gamma_file, signal_dir)
    # Hardcopy DL1 proton file to the background directory
    shutil.copy(dl1_proton_file, background_dir)
    # Hardcopy the trained models to the model directory
    if model == "LoadedModel":
        model_dir = tmp_path / "pretrained_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        key = f"{framework}_{telescope_type}_{reco_task}"
        shutil.copy(
            ctlearn_trained_dl1_mono_models[key],
            model_dir / f"ctlearn_mono_model_{key}.{MODEL_FILE_FORMATS[framework]}",
        )
        model_file = model_dir / f"ctlearn_mono_model_{key}.{MODEL_FILE_FORMATS[framework]}"
        assert model_file.exists(), f"Trained {framework} mono model file not found for {key}"
    # Output directory for trained model
    output_dir = tmp_path / f"ctlearn_{framework}_{model}_{reco_task}"
    # Build command-line arguments
    argv = [
        f"--signal={signal_dir}",
        "--pattern-signal=*.dl1.h5",
        f"--output={output_dir}",
        f"--reco={reco_task}",
        "--TrainCTLearnModel.n_epochs=2",
        "--TrainCTLearnModel.batch_size=4",
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
    argv.append(f"--TrainCTLearnModel.model_type={model}")
    if model == "LoadedModel":
        argv.append(f"--LoadedModel.load_model_from={model_file}")
    assert run_tool(TRAINING_TOOLS[framework](), argv=argv, cwd=tmp_path) == 0
    # --- Additional checks ---
    # Check that the trained model exists
    model_file = output_dir / f"ctlearn_model.{MODEL_FILE_FORMATS[framework]}"
    assert model_file.exists(), f"Trained model file not found for {reco_task}"
    # Check training_log.csv exists
    log_file = output_dir / "training_log.csv"
    assert log_file.exists(), f"Training log file not found for {reco_task}"
    # Read CSV and verify number of epochs
    log_df = pd.read_csv(log_file)
    num_epochs_logged = log_df.shape[0]
    assert (
        num_epochs_logged == 2
    ), f"Expected two epochs, found {num_epochs_logged} for {reco_task}"
    # Check that val_loss column exists
    assert (
        "val_loss" in log_df.columns
    ), f"'val_loss' column missing in training_log.csv for {reco_task}"
    val_loss = log_df["val_loss"].dropna()
    assert not val_loss.empty, f"'val_loss' column is empty for {reco_task}"
    assert ((val_loss >= 0.0) & (val_loss <= 1.0)).all(), (
        f"'val_loss' values out of range [0.0, 1.0] for {reco_task}: "
        f"{val_loss.tolist()}"
    )
    # Check that the event file for TensorBoard is created for train and validation 
    for subfolder in ["train", "validation"]:
        subfolder_path = output_dir / subfolder
        assert subfolder_path.is_dir(), f"TensorBoard '{subfolder}' directory missing in {output_dir}"
        # Check that at least one file starting with 'events.out.tfevents.' exists
        event_files = [
            f for f in subfolder_path.iterdir() 
            if f.is_file() and f.name.startswith("events.out.tfevents.")
        ]
        assert event_files, (
            f"No TensorBoard event file starting with 'events.out.tfevents.' "
            f"found in {subfolder_path}"
        )