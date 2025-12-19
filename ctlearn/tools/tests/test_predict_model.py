import shutil
import numpy as np

from ctapipe.core import run_tool
from ctapipe.io import TableLoader
from ctlearn.tools import MonoPredictCTLearnModel

def test_predict_model(ctlearn_trained_dl1_models, dl1_gamma_file, tmp_path):
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
    output_file = dl2_dir / "gamma.dl2.h5"
    argv = [
        f"--input_url={dl1_gamma_file}",
        f"--output={output_file}",
        "--PredictCTLearnModel.batch_size=4",
        "--DLImageReader.focal_length_choice=EQUIVALENT",
    ]
 
    # Run Prediction for energy and type together
    assert run_tool(
        MonoPredictCTLearnModel(),
        argv = argv + [
            f"--PredictCTLearnModel.load_type_model_from={model_dir}/ctlearn_model_type.keras",
            f"--PredictCTLearnModel.load_energy_model_from={model_dir}/ctlearn_model_energy.keras",
            "--use-HDF5Merger",
            "--no-dl1-images",
            "--no-true-images",
        ],
        cwd=tmp_path
    ) == 0

    assert run_tool(
        MonoPredictCTLearnModel(),
        argv= argv + [
            f"--PredictCTLearnModel.load_cameradirection_model_from="
            f"{model_dir}/ctlearn_model_cameradirection.keras",
            "--no-use-HDF5Merger",
        ],
        cwd=tmp_path,
    ) == 0


    allowed_tels = [7, 13, 15, 16, 17, 19]
    required_columns = [
        "telescope_pointing_azimuth",
        "telescope_pointing_altitude",
        "CTLearn_alt",
        "CTLearn_az",
        "CTLearn_prediction",
        "CTLearn_energy",
    ]
    # Check that the output DL2 file was created
    assert output_file.exists(), "Output DL2 file not created"
    # Check that the created DL2 file can be read with the TableLoader 
    with TableLoader(output_file, pointing=True, focal_length_choice="EQUIVALENT") as loader:
        events = loader.read_telescope_events_by_id(telescopes=allowed_tels)
        for tel_id in allowed_tels:
            assert len(events[tel_id]) > 0
            for col in required_columns:
                assert col in events[tel_id].colnames, f"{col} missing in DL2 file {output_file.name}"
                assert events[tel_id][col][0] is not np.nan, f"{col} has NaN values in DL2 file {output_file.name}"