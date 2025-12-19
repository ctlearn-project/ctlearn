import pandas as pd
import pytest
import shutil

from ctapipe.core import run_tool
from ctlearn.tools import TrainCTLearnModel

@pytest.mark.parametrize("reco_task", ["type", "energy", "cameradirection", "skydirection"])
def test_train_ctlearn_model(reco_task, dl1_gamma_file, dl1_proton_file, tmp_path):
    """
    Test training CTLearn model using the DL1 gamma and proton files for all reconstruction tasks.
    Each test run gets its own isolated temp directories.
    """
    # Temporary directories for signal and background
    signal_dir = tmp_path / "gamma_dl1"
    signal_dir.mkdir(parents=True, exist_ok=True)

    background_dir = tmp_path / "proton_dl1"
    background_dir.mkdir(parents=True, exist_ok=True)

    # Hardcopy DL1 gamma file to the signal directory
    shutil.copy(dl1_gamma_file, signal_dir)
    # Hardcopy DL1 proton file to the background directory
    shutil.copy(dl1_proton_file, background_dir)

    # Output directory for trained model
    output_dir = tmp_path / f"ctlearn_{reco_task}"
    
    # Build command-line arguments
    argv = [
        f"--signal={signal_dir}",
        "--pattern-signal=*.dl1.h5",
        f"--output={output_dir}",
        f"--reco={reco_task}",
        "--TrainCTLearnModel.n_epochs=2",
        "--TrainCTLearnModel.batch_size=4",
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

    # --- Additional checks ---
    # Check that the trained model exists
    model_file = output_dir / "ctlearn_model.keras"
    assert model_file.exists(), f"Trained model file not found for {reco_task}"
    # Check training_log.csv exists
    log_file = output_dir / "training_log.csv"
    assert log_file.exists(), f"Training log file not found for {reco_task}"
    # Read CSV and verify number of epochs
    log_df = pd.read_csv(log_file)
    num_epochs_logged = log_df.shape[0]
    assert num_epochs_logged == 2, f"Expected two epochs, found {num_epochs_logged} for {reco_task}"
    # Check that val_loss column exists
    assert "val_loss" in log_df.columns, (
        f"'val_loss' column missing in training_log.csv for {reco_task}"
    )
    val_loss_min= 0.0
    val_loss_max= 1.5 if reco_task == "skydirection" else 1.0
    # Check val_loss values are between 0.0 and 1.0 (or 1.5 for skydirection)
    val_loss = log_df["val_loss"].dropna()
    assert not val_loss.empty, (
        f"'val_loss' column is empty for {reco_task}"
    )
    assert ((val_loss >= val_loss_min) & (val_loss <= val_loss_max)).all(), (
        f"'val_loss' values out of range [{val_loss_min}, {val_loss_max}] for {reco_task}: "
        f"{val_loss.tolist()}"
    )