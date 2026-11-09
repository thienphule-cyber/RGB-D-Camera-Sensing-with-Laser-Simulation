"""Shared helpers for the sensor models: ray casting and frame utilities."""

from __future__ import annotations

from typing import Tuple

import mujoco
import numpy as np

# MuJoCo camera axes: x right, y up, looks along -z.
# OpenCV camera axes: x right, y down, looks along +z.
# The matrix is its own inverse, so it converts in both directions.
MUJOCO_TO_OPENCV = np.diag([1.0, -1.0, -1.0])

# LiDAR and laser frames use x forward, y left, z up. The sensor sites share the
# world orientation while the assembly looks along world +Y, so the sensor axes
# (columns, expressed in the site frame) are: x -> +Y, y -> -X, z -> +Z.
FORWARD_Y_MOUNT = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)


def angle_grid_deg(lo: float, hi: float, step: float) -> np.ndarray:
    """Evenly spaced angles [deg] from lo to hi (hi included if (hi - lo) % step == 0)."""
    if step <= 0.0:
        raise ValueError("angular step must be positive")
    if hi < lo:
        raise ValueError("angle range must satisfy lo <= hi")
    count = int(np.floor((hi - lo) / step + 1e-9)) + 1
    return lo + step * np.arange(count)


def spherical_to_cartesian(r, azimuth, elevation) -> np.ndarray:
    """Convert (range, azimuth, elevation) [m, rad, rad] to (..., 3) Cartesian points.

    x = r cos(el) cos(az), y = r cos(el) sin(az), z = r sin(el).
    """
    r = np.asarray(r, dtype=float)
    az = np.asarray(azimuth, dtype=float)
    el = np.asarray(elevation, dtype=float)
    cos_el = np.cos(el)
    return np.stack(
        [r * cos_el * np.cos(az), r * cos_el * np.sin(az), r * np.sin(el)], axis=-1
    )


def transform_points(points, rotation, translation) -> np.ndarray:
    """Apply p' = R p + t to an (N, 3) array of points."""
    pts = np.asarray(points, dtype=float)
    return pts @ np.asarray(rotation, dtype=float).T + np.asarray(translation, dtype=float)


class RayCaster:
    """Thin wrapper around mujoco.mj_ray.

    Hides signature differences between MuJoCo versions (newer releases add an
    extra `normal` output argument) by probing the function once at creation.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, exclude_body_id: int = -1) -> None:
        self._model = model
        self._data = data
        self._exclude = int(exclude_body_id)
        self._geomid = np.full(1, -1, dtype=np.int32)
        self._normal = np.zeros(3, dtype=np.float64)
        self._use_normal = False
        self._probe()

    def _call(self, origin: np.ndarray, direction: np.ndarray) -> float:
        if self._use_normal:
            return mujoco.mj_ray(
                self._model, self._data, origin, direction, None, 1,
                self._exclude, self._geomid, self._normal,
            )
        return mujoco.mj_ray(
            self._model, self._data, origin, direction, None, 1,
            self._exclude, self._geomid,
        )

    def _probe(self) -> None:
        origin = np.zeros(3)
        direction = np.array([0.0, 0.0, -1.0])
        try:
            self._call(origin, direction)
            return
        except TypeError:
            self._use_normal = True
        try:
            self._call(origin, direction)
        except TypeError as exc:
            raise RuntimeError("Unsupported mujoco.mj_ray signature in this MuJoCo version") from exc

    def cast(self, origin, direction) -> Tuple[float, int]:
        """Cast one ray. Returns (distance, geom_id); (-1.0, -1) when nothing is hit.

        `direction` must be a unit vector so the distance is in meters.
        """
        origin = np.ascontiguousarray(origin, dtype=np.float64)
        direction = np.ascontiguousarray(direction, dtype=np.float64)
        dist = self._call(origin, direction)
        return float(dist), int(self._geomid[0])

    def cast_many(self, origin, directions) -> Tuple[np.ndarray, np.ndarray]:
        """Cast N rays from one origin. Returns (distances (N,), geom_ids (N,))."""
        origin = np.ascontiguousarray(origin, dtype=np.float64)
        dirs = np.ascontiguousarray(directions, dtype=np.float64)
        n = dirs.shape[0]
        dist = np.empty(n, dtype=np.float64)
        geom_ids = np.empty(n, dtype=np.int64)
        for i in range(n):
            dist[i] = self._call(origin, dirs[i])
            geom_ids[i] = self._geomid[0]
        return dist, geom_ids