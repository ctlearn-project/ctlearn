import shutil
import numpy as np
import pytest

from ctapipe.core import run_tool
from ctapipe.io import TableLoader
from ctlearn.conftest import MODEL_FILE_FORMATS
from ctlearn.tools import LST1PredictionTool

# Columns that should be present in the output DL2 file
REQUIRED_COLUMNS = [
    "event_id",
    "obs_id",
    "CTLearnCameraReconstructor_tel_alt",
    "CTLearnCameraReconstructor_alt",
    "CTLearnCameraReconstructor_tel_az",
    "CTLearnCameraReconstructor_az",
    "CTLearnClassifier_tel_prediction",
    "CTLearnClassifier_prediction",
    "CTLearnRegressor_tel_energy",
    "CTLearnRegressor_energy",
    "CTLearnCameraReconstructor_tel_is_valid",
    "CTLearnCameraReconstructor_is_valid",
    "CTLearnClassifier_tel_is_valid",
    "CTLearnClassifier_is_valid",
    "CTLearnRegressor_tel_is_valid",
    "CTLearnRegressor_is_valid",
    "CTLearnCameraReconstructor_telescopes",
    "CTLearnClassifier_telescopes",
    "CTLearnRegressor_telescopes",
]


@pytest.mark.verifies_usecase("DPPS-UC-130-1.2.2")
@pytest.mark.parametrize("framework", ["Keras"])
def test_predict_mono_model_with_lst1_mock_data(
    tmp_path, ctlearn_trained_dl1_mono_models, mock_lst1_dl1_file, framework
):
    """
    Test LST1PredictionTool using trained mono models and mock LST-1 DL1 files.
    Each test run gets its own isolated temp directories.
    """

    model_dir = tmp_path / "trained_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    dl2_dir = tmp_path / "dl2_output"
    dl2_dir.mkdir(parents=True, exist_ok=True)

    # Hardcopy the trained models to the model directory
    telescope_type = "LST"
    for reco_task in ["type", "energy", "cameradirection"]:
        key = f"{framework}_{telescope_type}_{reco_task}"
        shutil.copy(
            ctlearn_trained_dl1_mono_models[key],
            model_dir / f"ctlearn_mono_model_{key}.keras",
        )
        model_file = model_dir / f"ctlearn_mono_model_{key}.keras"
        assert model_file.exists(), f"Trained mono model file not found for {key}"

    # Check that the mock LST1 DL1 file was created
    assert mock_lst1_dl1_file.exists(), "Mock LST1 DL1 file not found"

    output_file = dl2_dir / f"mock_lst1_{framework}_predictions.dl2.h5"

    # Build command-line arguments for LST1PredictionTool
    argv = [
        f"--input_url={mock_lst1_dl1_file}",
        f"--output={output_file}",
        "--LST1PredictionTool.batch_size=2",
        "--LST1PredictionTool.channels=cleaned_image",
        "--LST1PredictionTool.channels=cleaned_relative_peak_time",
        "--LST1PredictionTool.image_mapper_type=BilinearMapper",
        f"--type_model={model_dir}/ctlearn_mono_model_{framework}_{telescope_type}_type.{MODEL_FILE_FORMATS[framework]}",
        f"--energy_model={model_dir}/ctlearn_mono_model_{framework}_{telescope_type}_energy.{MODEL_FILE_FORMATS[framework]}",
        f"--cameradirection_model={model_dir}/ctlearn_mono_model_{framework}_{telescope_type}_cameradirection.{MODEL_FILE_FORMATS[framework]}",
        "--dl2-telescope",
        "--overwrite",
    ]

    # Run LST1PredictionTool
    assert run_tool(LST1PredictionTool(), argv=argv, cwd=tmp_path) == 0

    # Check that the output DL2 file was created
    assert output_file.exists(), "Output DL2 file not created"

    # Check that the created DL2 file can be read with the TableLoader
    allowed_tels = [1]
    with TableLoader(
        output_file, pointing=True, focal_length_choice="EFFECTIVE"
    ) as loader:
        # Check telescope-wise data
        tel_events = loader.read_telescope_events_by_id(
            telescopes=allowed_tels, dl1_parameters=True, dl2=True
        )
        for tel_id in allowed_tels:
            assert len(tel_events[tel_id]) > 0
            for col in REQUIRED_COLUMNS + [
                "tel_id",
                "hillas_intensity",
                "telescope_pointing_azimuth",
                "telescope_pointing_altitude",
            ]:
                if "_tel_" not in col:
                    continue
                assert (
                    col in tel_events[tel_id].colnames
                ), f"{col} missing in DL2 file {output_file.name}"
                assert (
                    tel_events[tel_id][col][0] is not np.nan
                ), f"{col} has NaN values in DL2 file {output_file.name}"

        # Check subarray-wise data
        subarray_events = loader.read_subarray_events(start=0, stop=2, dl2=True)
        assert len(subarray_events) > 0
        for col in REQUIRED_COLUMNS:
            if "_tel_" in col:
                continue
            assert (
                col in subarray_events.colnames
            ), f"{col} missing in DL2 file {output_file.name}"
            assert (
                subarray_events[col][0] is not np.nan
            ), f"{col} has NaN values in DL2 file {output_file.name}"
