import shutil
import numpy as np
import pytest

from ctapipe.core import run_tool
from ctapipe.io import read_table, TableLoader
from ctapipe.io.hdf5dataformat import DL1_SUBARRAY_GROUP
from ctlearn.conftest import MODEL_FILE_FORMATS
from ctlearn.tools.predict_stereo_model import StereoPredictCTLearnModel

# Columns that should be present in the output DL2 file
REQUIRED_COLUMNS = [
    "event_id",
    "obs_id",
    "CTLearnCameraReconstructor_tel_alt",
    "CTLearnCameraReconstructor_alt",
    "true_alt",
    "CTLearnCameraReconstructor_tel_az",
    "CTLearnCameraReconstructor_az",
    "true_az",
    "CTLearnClassifier_tel_prediction",
    "CTLearnClassifier_prediction",
    "true_shower_primary_id",
    "CTLearnRegressor_tel_energy",
    "CTLearnRegressor_energy",
    "true_energy",
    "CTLearnCameraReconstructor_tel_is_valid",
    "CTLearnCameraReconstructor_is_valid",
    "CTLearnClassifier_tel_is_valid",
    "CTLearnClassifier_is_valid",
    "CTLearnRegressor_tel_is_valid",
    "CTLearnRegressor_is_valid",
    "CTLearnCameraReconstructor_telescopes",
    "CTLearnClassifier_telescopes",
    "CTLearnRegressor_telescopes",
    "tels_with_trigger",
]

@pytest.mark.verifies_usecase("DPPS-UC-130-1.2.2")
@pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
@pytest.mark.parametrize(
    ("input_file_fixture", "input_suffix"),
    [
        ("dl1_gamma_file", "simulation"),
        ("dl1_obs_mock_file", "obs_mock"),
    ],
)
def test_predict_stereo_model_with_dl1_images(
    tmp_path,
    request,
    ctlearn_trained_dl1_stereo_models,
    input_file_fixture,
    input_suffix,
    framework,
):
    """
    Test training CTLearn stereo model using the DL1 gamma and proton files for all reconstruction tasks
    and predicting DL2 from DL1 images. Each test run gets its own isolated temp directories.
    """
    model_dir = tmp_path / "trained_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    dl2_dir = tmp_path / "dl2_output"
    dl2_dir.mkdir(parents=True, exist_ok=True)
    # Define telescope types and their available telescopes
    telescope_type = "MST"
    allowed_tels = [7, 13, 15]
    # Hardcopy the trained models to the model directory
    for reco_task in ["type", "energy", "skydirection"]:
        key = f"{framework}_{telescope_type}_{reco_task}"
        shutil.copy(
            ctlearn_trained_dl1_stereo_models[key],
            model_dir / f"ctlearn_stereo_model_{key}.{MODEL_FILE_FORMATS[framework]}",
        )
        model_file = model_dir / f"ctlearn_stereo_model_{key}.{MODEL_FILE_FORMATS[framework]}"
        assert model_file.exists(), f"Trained stereo model file not found for {key}"
    # Build command-line arguments
    argv = [
        f"--input_url={request.getfixturevalue(input_file_fixture)}",
        "--PredictCTLearnModel.batch_size=2",
        "--PredictCTLearnModel.stack_telescope_images=True",
        "--DLImageReader.mode=stereo",
        "--DLImageReader.focal_length_choice=EQUIVALENT",
        f"--DLImageReader.allowed_tels={allowed_tels}",
        "--dl1-features",
        "--no-dl1-images",
        "--no-true-images",
    ]
    output_file = (
        dl2_dir
        / f"gamma_{framework}_{telescope_type}_stereo_from_images_{input_suffix}.dl2.h5"
    )
    # Run Prediction tool
    assert (
        run_tool(
            StereoPredictCTLearnModel(),
            argv=argv
            + [
                f"--output={output_file}",
                f"--PredictCTLearnModel.load_type_model_from={model_dir}/ctlearn_stereo_model_{framework}_{telescope_type}_type.{MODEL_FILE_FORMATS[framework]}",
                f"--PredictCTLearnModel.load_energy_model_from={model_dir}/ctlearn_stereo_model_{framework}_{telescope_type}_energy.{MODEL_FILE_FORMATS[framework]}",
                f"--PredictCTLearnModel.load_skydirection_model_from={model_dir}/ctlearn_stereo_model_{framework}_{telescope_type}_skydirection.{MODEL_FILE_FORMATS[framework]}",
            ],
            cwd=tmp_path,
        )
        == 0
    )
    # Check that the output DL2 file was created
    assert output_file.exists(), "Output DL2 file not created"
    # Check that the created DL2 file can be read with the TableLoader
    with TableLoader(
        output_file, pointing=True, focal_length_choice="EQUIVALENT"
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
                "leakage_pixels_width_2",
                "telescope_pointing_azimuth",
                "telescope_pointing_altitude",
            ]:
                if "_tel_" in col:
                    continue
                if input_suffix == "obs_mock" and col.startswith("true_"):
                    continue
                if "CTLearnCameraReconstructor" in col:
                    col = col.replace(
                        "CTLearnCameraReconstructor", "CTLearnSkyReconstructor"
                    )
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
            if input_suffix == "obs_mock" and col.startswith("true_"):
                continue
            if "CTLearnCameraReconstructor" in col:
                col = col.replace(
                    "CTLearnCameraReconstructor", "CTLearnSkyReconstructor"
                )
            assert (
                col in subarray_events.colnames
            ), f"{col} missing in DL2 file {output_file.name}"
            assert (
                subarray_events[col][0] is not np.nan
            ), f"{col} has NaN values in DL2 file {output_file.name}"
    # Check that the DL1 feature table was created and contains the expected columns
    feature_table_path = (
        f"{DL1_SUBARRAY_GROUP}/features/CTLearn"
    )
    feature_table = read_table(output_file, feature_table_path)
    assert len(feature_table) > 0, (
        f"DL1 feature table missing or empty at {feature_table_path}"
    )
    for col in ["particletype", "energy", "geometry"]:
        assert (
            f"CTLearn_{col}_feature_vectors" in feature_table.colnames
        ), f"{col} missing in DL1 feature table {feature_table_path}"
