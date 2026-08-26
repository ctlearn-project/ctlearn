import numpy as np
import pytest
from ctapipe.instrument import CameraGeometry

from ctlearn.core.hexagdly_geometry import HexGridTransform
from ctlearn.utils import get_lst1_subarray_description


def test_lst1_geometry_zero_mismatch():
    """The bundled LST-1 geometry (offline, no download) must map with zero
    neighbour mismatches -- this is the camera CTLearn ships test resources for."""
    subarray = get_lst1_subarray_description()
    geometry = subarray.tel[1].camera.geometry

    grid = HexGridTransform(geometry)

    assert grid.neighbor_mismatch_count == 0
    assert grid.H > 0 and grid.W > 0
    assert len(grid.row_idx) == geometry.n_pixels
    assert len(grid.col_idx) == geometry.n_pixels
    assert grid.row_idx.max() < grid.H
    assert grid.col_idx.max() < grid.W


@pytest.mark.remote_data
@pytest.mark.parametrize("camera_name", ["MAGICCam", "NectarCam", "FlashCam"])
def test_other_hex_cameras_zero_mismatch(camera_name):
    """Broader validation across other real CTA hex-pixel camera geometries.

    Uses ``CameraGeometry.from_name``, which fetches reference camera data and
    needs network access (or a warm cache) -- run with ``--remote-data``.
    """
    geometry = CameraGeometry.from_name(camera_name)
    assert geometry.n_pixels > 0  # sanity: geometry actually loaded

    grid = HexGridTransform(geometry)

    assert grid.neighbor_mismatch_count == 0


@pytest.mark.remote_data
@pytest.mark.parametrize("camera_name", ["CHEC", "SCTCam"])
def test_square_pixel_cameras_raise(camera_name):
    """Square-pixel cameras have no 60-degree neighbour structure to fold into
    and must raise, not silently produce a meaningless mapping."""
    geometry = CameraGeometry.from_name(camera_name)

    with pytest.raises(ValueError):
        HexGridTransform(geometry)


def test_scatter_roundtrip():
    """Every real pixel's value must land at exactly its assigned grid cell,
    and every other cell must stay zero."""
    subarray = get_lst1_subarray_description()
    geometry = subarray.tel[1].camera.geometry
    grid = HexGridTransform(geometry)

    rng = np.random.default_rng(0)
    values = rng.uniform(size=geometry.n_pixels).astype(np.float32)

    out = grid.scatter(values)

    assert out.shape == (grid.H, grid.W)
    np.testing.assert_allclose(out[grid.row_idx, grid.col_idx], values)
    assert out.sum() == pytest.approx(values.sum(), rel=1e-5)


def test_scatter_batched():
    """`scatter` must work on batched/multi-channel input via leading dims."""
    subarray = get_lst1_subarray_description()
    geometry = subarray.tel[1].camera.geometry
    grid = HexGridTransform(geometry)

    rng = np.random.default_rng(1)
    values = rng.uniform(size=(3, 2, geometry.n_pixels)).astype(np.float32)

    out = grid.scatter(values)

    assert out.shape == (3, 2, grid.H, grid.W)
    np.testing.assert_allclose(out[..., grid.row_idx, grid.col_idx], values)
