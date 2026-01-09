import shutil
import numpy as np
import pytest

from ctapipe.core import run_tool
from ctapipe.io import TableLoader
from ctlearn.tools import MonoPredictCTLearnModel

# Columns that should be present in the output DL2 file
REQUIRED_COLUMNS = [
    "event_id",
    "obs_id",
    "CTLearn_alt",
    "true_alt",
    "CTLearn_az",
    "true_az",
    "CTLearn_prediction",
    "true_shower_primary_id",
    "CTLearn_energy",
    "true_energy",
    "CTLearn_is_valid",
    "CTLearn_telescopes",
    "tels_with_trigger"
]


@pytest.mark.parametrize("dl2_tel_flag", ["dl2-telescope", "no-dl2-telescope"])
def test_predict_model(tmp_path, ctlearn_trained_dl1_models, dl1_gamma_file, dl2_tel_flag):
    """
    Test training CTLearn model using the DL1 gamma and proton files for all reconstruction tasks.
    Each test run gets its own isolated temp directories.
    """

    model_dir = tmp_path / "trained_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    dl2_dir = tmp_path / "dl2_output"
    dl2_dir.mkdir(parents=True, exist_ok=True)

    # Hardcopy the trained models to the model directory
    for reco_task in ["type", "energy", "cameradirection"]:
        shutil.copy(ctlearn_trained_dl1_models[f"{reco_task}"], model_dir / f"ctlearn_model_{reco_task}.keras")
        model_file = model_dir / f"ctlearn_model_{reco_task}.keras"
        assert model_file.exists(), f"Trained model file not found for {reco_task}"

    # Build command-line arguments
    argv = [
        f"--input_url={dl1_gamma_file}",
        "--PredictCTLearnModel.batch_size=4",
        "--DLImageReader.focal_length_choice=EQUIVALENT",
        f"--PredictCTLearnModel.load_type_model_from={model_dir}/ctlearn_model_type.keras",
        f"--PredictCTLearnModel.load_energy_model_from={model_dir}/ctlearn_model_energy.keras",
        f"--PredictCTLearnModel.load_cameradirection_model_from={model_dir}/ctlearn_model_cameradirection.keras",
        "--no-dl1-images",
        "--no-true-images",
        f"--{dl2_tel_flag}",
    ]
    # Test with different allowed telescope lists (single and multiple telescope IDs)
    allowed_tels_dict = {"single_tel_id" : [7], "multiple_tel_ids": [7, 13, 15, 16, 17, 19]}
    for name, allowed_tels in allowed_tels_dict.items():
        output_file = dl2_dir / f"gamma_{dl2_tel_flag}_{name}.dl2.h5"
        # Run Prediction tool
        assert run_tool(
            MonoPredictCTLearnModel(),
            argv = argv + [
                f"--output={output_file}",
                f"--DLImageReader.allowed_tels={allowed_tels}",
            ],
            cwd=tmp_path
        ) == 0

        # Check that the output DL2 file was created
        assert output_file.exists(), "Output DL2 file not created"
        # Check that the created DL2 file can be read with the TableLoader 
        with TableLoader(output_file, pointing=True, focal_length_choice="EQUIVALENT") as loader:
            # Check telescope-wise data
            tel_events = loader.read_telescope_events_by_id(telescopes=allowed_tels, dl1_parameters=True, dl2=True)
            for tel_id in allowed_tels:
                assert len(tel_events[tel_id]) > 0
                for col in REQUIRED_COLUMNS + [
                    "tel_id",
                    "hillas_intensity",
                    "leakage_pixels_width_2",
                    "telescope_pointing_azimuth",
                    "telescope_pointing_altitude",
                ]:
                    assert col in tel_events[tel_id].colnames, f"{col} missing in DL2 file {output_file.name}"
                    assert tel_events[tel_id][col][0] is not np.nan, f"{col} has NaN values in DL2 file {output_file.name}"
            # Check subarray-wise data
            subarray_events = loader.read_subarray_events(start=0, stop=2, dl2=True)
            assert len(subarray_events) > 0
            for col in REQUIRED_COLUMNS:
                assert col in subarray_events.colnames, f"{col} missing in DL2 file {output_file.name}"
                assert subarray_events[col][0] is not np.nan, f"{col} has NaN values in DL2 file {output_file.name}"