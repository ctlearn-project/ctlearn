"""
This module maps a hexagonal-pixel ``ctapipe.instrument.CameraGeometry`` onto the
2-D offset-column addressing grid that ``keras_hexagdly`` (and upstream HexagDLy)
expects.
"""

import numpy as np

__all__ = ["HexGridTransform"]


class HexGridTransform:
    """Map a 1-D hexagonal camera image to a 2-D ``keras_hexagdly`` grid.

    ``keras_hexagdly`` (like upstream HexagDLy) works on a 2-D array where the
    hexagonal neighbour structure is encoded by an offset-column layout. This
    class finds the row/column of every camera pixel on that grid from its
    ``ctapipe`` ``CameraGeometry``.

    Camera geometries are not generally axis-aligned (e.g. ``LSTCam`` is rotated
    ~41 degrees), so neighbour pairs are not necessarily horizontal, which the
    offset-column addressing relies on. This is handled with an automatic
    de-rotation step (rotate the pixel coordinates so a neighbour pair becomes
    horizontal) before the offset-column assignment. The resulting mapping is
    verified against the camera's own neighbour graph
    (:attr:`neighbor_mismatch_count`); zero mismatches means every pixel's grid
    neighbours are exactly its physical camera neighbours.

    Only hexagonal-pixel cameras are supported -- construction raises on a
    square-pixel geometry (e.g. ``SCTCam``, ``CHEC``) via the "no horizontal
    neighbour vectors" check below, since a square pixel grid has no 60-degree
    neighbour structure to fold into.

    Parameters
    ----------
    geometry : ctapipe.instrument.CameraGeometry
        Camera geometry to map. Must have hexagonal pixels.

    Attributes
    ----------
    H, W : int
        Height and width of the 2-D grid the camera maps onto.
    row_idx, col_idx : numpy.ndarray
        Grid row/column of each camera pixel, in pixel-index order.
    neighbor_mismatch_count : int
        Number of camera pixels whose grid-neighbours don't exactly match their
        physical camera neighbours. Should be 0; a positive value means the
        addressing produced for this geometry is not trustworthy.
    """

    def __init__(self, geometry):
        x = geometry.pix_x.value.astype(np.float64)
        y = geometry.pix_y.value.astype(np.float64)
        x, y = self._derotate(x, y, geometry.neighbors)

        vecs = self._neighbor_vectors(x, y, geometry.neighbors)
        dist = np.linalg.norm(vecs, axis=1)
        horiz = np.abs(vecs[:, 1]) < np.median(dist) * 1e-3
        if not np.any(horiz):
            raise ValueError(
                "no horizontal neighbour vectors after de-rotation -- "
                "HexGridTransform only supports hexagonal-pixel cameras"
            )
        if not np.any(~horiz):
            raise ValueError(
                "no non-horizontal neighbour vectors after de-rotation -- "
                "HexGridTransform only supports hexagonal-pixel cameras"
            )
        horizontal_pitch = np.median(np.abs(vecs[horiz, 0]))
        vertical_pitch = np.median(np.abs(vecs[~horiz, 1]))

        best = None
        for y0 in (0.0, y.max(), y.min()):
            r = np.rint((y - y0) / vertical_pitch).astype(np.int64)
            q = np.rint((x - x.min()) / horizontal_pitch - r / 2).astype(np.int64)
            row, col = self._offset_from_axial(q, r)
            row, col = row - row.min(), col - col.min()
            mc = self._mismatch(geometry, row, col)
            if best is None or mc < best[0]:
                best = (mc, row, col)

        self.neighbor_mismatch_count, self.row_idx, self.col_idx = best
        self.H = int(self.row_idx.max()) + 1
        self.W = int(self.col_idx.max()) + 1
        if len(set(zip(self.row_idx, self.col_idx))) != geometry.n_pixels:
            raise RuntimeError(
                f"HexGridTransform produced colliding grid cells for camera "
                f"'{geometry.name}' -- no candidate grid origin gave every pixel "
                "a unique cell. This geometry's hex addressing is not reliable."
            )

    @staticmethod
    def _neighbor_vectors(x, y, neighbors):
        v = []
        for i, nb in enumerate(neighbors):
            for j in nb:
                if i < j:
                    v.append((x[j] - x[i], y[j] - y[i]))
        return np.asarray(v)

    @classmethod
    def _derotate(cls, x, y, neighbors):
        v = cls._neighbor_vectors(x, y, neighbors)
        # angle that makes a neighbour pair horizontal (fold into one 60deg sector)
        t = np.median(np.arctan2(v[:, 1], v[:, 0]) % (np.pi / 3))
        return x * np.cos(t) + y * np.sin(t), -x * np.sin(t) + y * np.cos(t)

    @staticmethod
    def _offset_from_axial(q, r):
        row = r + np.floor_divide(q - (q & 1), 2)
        return row.astype(np.int64), q.astype(np.int64)

    @staticmethod
    def _hex_neighbors_on_grid(grid, row, col):
        diag = (
            [(-1, -1), (-1, 1), (0, -1), (0, 1)]
            if col % 2 == 0
            else [(0, -1), (0, 1), (1, -1), (1, 1)]
        )
        out = []
        for dr, dc in [(-1, 0), (1, 0), *diag]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] >= 0:
                out.append(int(grid[nr, nc]))
        return sorted(out)

    @classmethod
    def _mismatch(cls, geometry, row, col):
        grid = np.full((int(row.max()) + 1, int(col.max()) + 1), -1, int)
        for i, (r, c) in enumerate(zip(row, col)):
            if grid[r, c] >= 0:
                return geometry.n_pixels  # collision -> reject this origin
            grid[r, c] = i
        m = 0
        for i, cam_nb in enumerate(geometry.neighbors):
            if cls._hex_neighbors_on_grid(grid, int(row[i]), int(col[i])) != sorted(
                map(int, cam_nb)
            ):
                m += 1
        return m

    def scatter(self, images):
        """Place per-pixel values onto the 2-D grid.

        Parameters
        ----------
        images : numpy.ndarray
            Array of shape ``(..., n_pixels)``.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(..., H, W)``. Grid cells with no corresponding
            camera pixel (padding) are zero.
        """
        out = np.zeros(images.shape[:-1] + (self.H, self.W), np.float32)
        out[..., self.row_idx, self.col_idx] = images
        return out
