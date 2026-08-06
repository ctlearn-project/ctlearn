import shutil
import numpy as np
import pytest

from ctapipe.core import run_tool
from ctapipe.io import TableLoader
from ctlearn.conftest import MODEL_FILE_FORMATS
from ctlearn.tools import MonoPredictCTLearnModel, StereoPredictCTLearnModel

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


@pytest.mark.verifies_usecase("DPPS-UC-130-1.2")
@pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
def test_predict_mono_model_with_r1_waveforms(
    tmp_path, ctlearn_trained_r1_mono_models, r1_gamma_file, framework
):
    """
    Test training CTLearn mono model using the R1 gamma and proton files for all reconstruction tasks
    and predicting DL2 from R1 waveforms. Each test run gets its own isolated temp directories.
    """
    model_dir = tmp_path / "trained_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    dl2_dir = tmp_path / "dl2_output"
    dl2_dir.mkdir(parents=True, exist_ok=True)
    # Define telescope types and their available telescopes
    telescope_type = "LST"
    available_tels = [1, 2]
    # Hardcopy the trained models to the model directory
    for reco_task in ["type", "energy", "cameradirection"]:
        key = f"{framework}_{telescope_type}_{reco_task}"
        shutil.copy(
            ctlearn_trained_r1_mono_models[key],
            model_dir / f"ctlearn_mono_model_{key}.{MODEL_FILE_FORMATS[framework]}",
        )
        model_file = model_dir / f"ctlearn_mono_model_{key}.{MODEL_FILE_FORMATS[framework]}"
        assert model_file.exists(), f"Trained mono model file not found for {key}"
    # Build command-line arguments
    argv = [
        f"--input_url={r1_gamma_file}",
        "--PredictCTLearnModel.batch_size=2",
        "--PredictCTLearnModel.dl1dh_reader_type=DLWaveformReader",
        "--DLWaveformReader.sequence_length=5",
        "--DLWaveformReader.focal_length_choice=EQUIVALENT",
        "--no-r1-waveforms",
        "--dl2-telescope",
    ]
    output_file = dl2_dir / f"gamma_{framework}_{telescope_type}_mono_from_waveforms.dl2.h5"
    # Run Prediction tool
    assert (
        run_tool(
            MonoPredictCTLearnModel(),
            argv=argv
            + [
                f"--output={output_file}",
                f"--PredictCTLearnModel.load_type_model_from={model_dir}/ctlearn_mono_model_{framework}_{telescope_type}_type.{MODEL_FILE_FORMATS[framework]}",
                f"--PredictCTLearnModel.load_energy_model_from={model_dir}/ctlearn_mono_model_{framework}_{telescope_type}_energy.{MODEL_FILE_FORMATS[framework]}",
                f"--PredictCTLearnModel.load_cameradirection_model_from={model_dir}/ctlearn_mono_model_{framework}_{telescope_type}_cameradirection.{MODEL_FILE_FORMATS[framework]}",
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
            telescopes=available_tels, dl1_parameters=True, dl2=True
        )
        for tel_id in available_tels:
            assert len(tel_events[tel_id]) > 0
            for col in REQUIRED_COLUMNS + [
                "tel_id",
                "hillas_intensity",
                "leakage_pixels_width_2",
                "telescope_pointing_azimuth",
                "telescope_pointing_altitude",
            ]:
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


@pytest.mark.verifies_usecase("DPPS-UC-130-1.2.2")
@pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
@pytest.mark.parametrize("dl2_tel_flag", ["dl2-telescope", "no-dl2-telescope"])
def test_predict_mono_model_with_dl1_images(
    tmp_path, ctlearn_trained_dl1_mono_models, dl1_gamma_file, framework, dl2_tel_flag
):
    """
    Test training CTLearn model using the DL1 gamma and proton files for all reconstruction tasks
    and predicting DL2 from DL1 images. Each test run gets its own isolated temp directories.
    """
    model_dir = tmp_path / "trained_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    dl2_dir = tmp_path / "dl2_output"
    dl2_dir.mkdir(parents=True, exist_ok=True)
    # Define telescope types and their allowed telescopes
    telescope_types = {
        "LST": [2],
        "MST": [7, 13, 15, 16, 17, 19],
        # "SST": [30, 37, 43, 44, 53],
    }
    image_mapper_types = {
        "LST": "BilinearMapper",
        "MST": "OversamplingMapper",
        # "SST": "SquareMapper",
    }
    # Hardcopy the trained models to the model directory
    for telescope_type in telescope_types.keys():
        for reco_task in ["type", "energy", "cameradirection"]:
            key = f"{framework}_{telescope_type}_{reco_task}"
            shutil.copy(
                ctlearn_trained_dl1_mono_models[key],
                model_dir / f"ctlearn_mono_model_{key}.{MODEL_FILE_FORMATS[framework]}",
            )
            model_file = model_dir / f"ctlearn_mono_model_{key}.{MODEL_FILE_FORMATS[framework]}"
            assert model_file.exists(), f"Trained mono model file not found for {key}"
    # Build command-line arguments
    argv = [
        f"--input_url={dl1_gamma_file}",
        "--PredictCTLearnModel.batch_size=2",
        "--DLImageReader.focal_length_choice=EQUIVALENT",
        "--no-dl1-images",
        "--no-true-images",
        f"--{dl2_tel_flag}",
    ]
    for telescope_type, allowed_tels in telescope_types.items():
        output_file = (
            dl2_dir / f"gamma_{dl2_tel_flag}_{telescope_type}_{framework}_mono_from_images.dl2.h5"
        )
        # Run Prediction tool
        assert (
            run_tool(
                MonoPredictCTLearnModel(),
                argv=argv
                + [
                    f"--output={output_file}",
                    f"--DLImageReader.allowed_tels={allowed_tels}",
                    f"--DLImageReader.image_mapper_type={image_mapper_types[telescope_type]}",
                    f"--PredictCTLearnModel.load_type_model_from={model_dir}/ctlearn_mono_model_{framework}_{telescope_type}_type.{MODEL_FILE_FORMATS[framework]}",
                    f"--PredictCTLearnModel.load_energy_model_from={model_dir}/ctlearn_mono_model_{framework}_{telescope_type}_energy.{MODEL_FILE_FORMATS[framework]}",
                    f"--PredictCTLearnModel.load_cameradirection_model_from={model_dir}/ctlearn_mono_model_{framework}_{telescope_type}_cameradirection.{MODEL_FILE_FORMATS[framework]}",
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
                    if dl2_tel_flag == "no-dl2-telescope" and "_tel_" in col:
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


@pytest.mark.verifies_usecase("DPPS-UC-130-1.2.2")
@pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
def test_predict_stereo_model_with_dl1_images(
    tmp_path, ctlearn_trained_dl1_stereo_models, dl1_gamma_file, framework
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
        f"--input_url={dl1_gamma_file}",
        "--PredictCTLearnModel.batch_size=2",
        "--PredictCTLearnModel.stack_telescope_images=True",
        "--DLImageReader.mode=stereo",
        "--DLImageReader.focal_length_choice=EQUIVALENT",
        f"--DLImageReader.allowed_tels={allowed_tels}",
        "--no-dl1-images",
        "--no-true-images",
    ]
    output_file = dl2_dir / f"gamma_{framework}_{telescope_type}_stereo_from_images.dl2.h5"
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
