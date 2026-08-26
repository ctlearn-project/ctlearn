import numpy as np

from ctlearn.core.hexagdly_mapper import HexagdlyMapper
from ctlearn.core.hexagdly_model import HexCNN
from ctlearn.utils import get_lst1_subarray_description


def _lst1_input_shape(n_channels=2):
    subarray = get_lst1_subarray_description()
    geometry = subarray.tel[1].camera.geometry
    mapper = HexagdlyMapper(geometry=geometry)
    return (mapper.image_shape, mapper.image_shape, n_channels)


def test_hexcnn_builds_and_predicts_single_task():
    input_shape = _lst1_input_shape()
    model = HexCNN(
        input_shape=input_shape,
        tasks=["type"],
        architecture=[
            {"filters": 4, "kernel_size": 1, "number": 1},
            {"filters": 8, "kernel_size": 1, "number": 1},
        ],
        attention_mechanism=None,
    )

    rng = np.random.default_rng(0)
    batch = rng.uniform(size=(2, *input_shape)).astype(np.float32)
    output = model.model.predict(batch, verbose=0)

    # Single-task 'type' output: (batch, 2) softmax logits.
    assert output.shape == (2, 2)
    np.testing.assert_allclose(output.sum(axis=-1), np.ones(2), rtol=1e-4)


def test_hexcnn_builds_multi_task_with_batchnorm_and_bottleneck():
    input_shape = _lst1_input_shape()
    model = HexCNN(
        input_shape=input_shape,
        tasks=["type", "energy"],
        architecture=[{"filters": 4, "kernel_size": 1, "number": 1}],
        batchnorm=True,
        bottleneck_filters=6,
        attention_mechanism="Channel-SE",
        attention_reduction_ratio=2,
        head_layers={"type": [8, 2], "energy": [8, 1]},
    )

    rng = np.random.default_rng(1)
    batch = rng.uniform(size=(2, *input_shape)).astype(np.float32)
    outputs = model.model.predict(batch, verbose=0)

    assert outputs["type"].shape == (2, 2)
    assert outputs["energy"].shape == (2, 1)
