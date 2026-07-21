import pandas as pd
import pytest
import shutil
from unittest import mock

from ctapipe.core import run_tool
from ctlearn.tools import DLFrameWork


@pytest.mark.parametrize("reco_task", ["type", "energy", "cameradirection"])
@mock.patch("ctapipe.instrument.SubarrayDescription.__eq__", return_value=True)
def test_train_ctlearn_model(mock_eq, reco_task, dl1_gamma_file, dl1_proton_file, tmp_path):
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
    allowed_tels = [7, 13, 15, 16, 17, 19]

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

    if reco_task == "type":
        argv.extend(
            [
                f"--background={background_dir}",
                "--pattern-background=*.dl1.h5",
            ]
        )

    # Run training
    assert run_tool(DLFrameWork(), argv=argv, cwd=tmp_path) == 0

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
