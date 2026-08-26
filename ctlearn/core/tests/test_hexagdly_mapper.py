import numpy as np
import pytest
from ctapipe.instrument import CameraGeometry

from ctlearn.core.hexagdly_mapper import HexagdlyMapper
from ctlearn.utils import get_lst1_subarray_description


def _lst1_geometry():
    subarray = get_lst1_subarray_description()
    return subarray.tel[1].camera.geometry


def test_image_shape_is_square_padded():
    geometry = _lst1_geometry()
    mapper = HexagdlyMapper(geometry=geometry)

    assert mapper.image_shape == max(mapper.grid_transform.H, mapper.grid_transform.W)
    assert mapper.mapping_table.shape == (
        geometry.n_pixels,
        mapper.image_shape * mapper.image_shape,
    )


def test_map_image_roundtrip():
    """Every pixel value must land at exactly its HexGridTransform grid cell,
    for every channel, with nothing lost or smeared (nearest-neighbour, weight 1.0)."""
    geometry = _lst1_geometry()
    mapper = HexagdlyMapper(geometry=geometry)

    rng = np.random.default_rng(0)
    n_channels = 2
    raw_vector = rng.uniform(size=(geometry.n_pixels, n_channels)).astype(np.float32)

    image = mapper.map_image(raw_vector)

    assert image.shape == (mapper.image_shape, mapper.image_shape, n_channels)
    grid = mapper.grid_transform
    for channel in range(n_channels):
        np.testing.assert_allclose(
            image[grid.row_idx, grid.col_idx, channel],
            raw_vector[:, channel],
            rtol=1e-5,
        )
    # Padding cells introduced by squaring (H, W) up to image_shape must be zero.
    assert image.sum() == pytest.approx(raw_vector.sum(), rel=1e-4)


@pytest.mark.remote_data
@pytest.mark.parametrize("camera_name", ["CHEC", "SCTCam"])
def test_square_pixel_camera_raises(camera_name):
    """Uses ``CameraGeometry.from_name``, which needs network access (or a
    warm cache) -- run with ``--remote-data``."""
    geometry = CameraGeometry.from_name(camera_name)
    with pytest.raises(ValueError):
        HexagdlyMapper(geometry=geometry)
