"""
This module defines ``HexagdlyMapper``, an ``ImageMapper`` that maps a hexagonal
camera image onto the exact hex-addressed grid ``keras_hexagdly`` convolutions
expect, instead of interpolating onto an axis-aligned square grid.
"""

import numpy as np
from scipy.sparse import csr_matrix

from ctapipe.instrument.camera import PixelShape
from dl1_data_handler.image_mapper import ImageMapper

from ctlearn.core.hexagdly_geometry import HexGridTransform

__all__ = ["HexagdlyMapper"]


class HexagdlyMapper(ImageMapper):
    """Map a hexagonal camera image onto a ``keras_hexagdly``-addressed grid.

    Unlike the other ``ImageMapper`` subclasses (``BilinearMapper``,
    ``AxialMapper``, ...), which resample the hexagonal pixel grid onto an
    axis-aligned square grid via interpolation, ``HexagdlyMapper`` places each
    pixel value at its *exact* nearest-neighbour cell on the offset-column grid
    that ``keras_hexagdly.layers.Conv2d``/``MaxPool2d`` expect (see
    :class:`ctlearn.core.hexagdly_geometry.HexGridTransform`). No interpolation
    is involved -- every output cell is either exactly one camera pixel's value
    or empty padding.

    The mapping is built as a one-hot ``mapping_table`` (weight 1.0 at each
    pixel's grid cell), so the inherited ``ImageMapper.map_image`` is reused
    unmodified, following the same pattern as e.g. ``SquareMapper``. Because
    ``DLDataReader`` assumes a square ``(image_shape, image_shape, channels)``
    input shape, the grid is padded to a square of size
    ``max(HexGridTransform.H, HexGridTransform.W)``; the extra rows/columns are
    all-zero padding cells, same as the padding every other mapper already
    introduces at camera edges.

    Note the row axis follows ``HexGridTransform``'s own convention (row 0 at
    the bottom, matching ``origin="lower"`` display), which is flipped relative
    to the top-left-origin convention the other mappers produce via
    ``ImageMapper._get_sparse_mapping_matrix``. This has no effect on training
    (it's a fixed permutation applied consistently at map time and inference
    time); it only means a raw ``CameraDisplay``-style comparison against
    e.g. ``BilinearMapper`` output would show it upside down.
    """

    def __init__(
        self,
        geometry,
        config=None,
        parent=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        geometry : ctapipe.instrument.CameraGeometry
            Geometry of the camera to map. Must have hexagonal pixels.
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            This is mutually exclusive with passing a ``parent``.
        parent : ctapipe.core.Component or ctapipe.core.Tool
            Parent of this component in the configuration hierarchy,
            this is mutually exclusive with passing ``parent``.
        """
        super().__init__(
            geometry=geometry,
            config=config,
            parent=parent,
            **kwargs,
        )

        if self.geometry.pix_type != PixelShape.HEXAGON:
            raise ValueError(
                f"HexagdlyMapper is only available for hexagonal pixel cameras. "
                f"Pixel type of the selected camera is '{self.geometry.pix_type}'."
            )

        # Build the hex-grid addressing for this camera geometry.
        grid = HexGridTransform(self.geometry)
        if grid.neighbor_mismatch_count > 0:
            raise ValueError(
                f"HexGridTransform produced {grid.neighbor_mismatch_count} "
                f"neighbour mismatch(es) for camera '{self.camera_type}' -- the "
                "resulting hex addressing is not reliable for this geometry."
            )
        self.grid_transform = grid

        # Pad the (H, W) grid to a square, since DLDataReader assumes a square
        # image_shape. row_idx/col_idx are already < H <= image_shape and
        # < W <= image_shape, so no adjustment to their values is needed.
        self.image_shape = max(grid.H, grid.W)

        # One-hot mapping table: each pixel maps to exactly one grid cell,
        # weight 1.0. This lets the inherited `ImageMapper.map_image` be reused
        # unmodified.
        flat_index = (
            grid.row_idx.astype(np.int64) * self.image_shape
            + grid.col_idx.astype(np.int64)
        )
        self.mapping_table = csr_matrix(
            (
                np.ones(self.n_pixels, dtype=np.float32),
                (np.arange(self.n_pixels), flat_index),
            ),
            shape=(self.n_pixels, self.image_shape * self.image_shape),
            dtype=np.float32,
        )
