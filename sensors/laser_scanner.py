"""Simulated laser line scanner (laser plane + camera triangulation).

MuJoCo has no laser line scanner, so it is modelled geometrically:
  * The laser projects a vertical plane through the laser site. The plane yaw
    is its rotation about the vertical axis (0 = plane contains the forward axis).
  * Rays are cast inside that plane (a fan). Each hit is a laser spot on a surface.
  * A real scanner triangulates spots with the camera, so a spot is only measurable
    if the camera can see it: spots outside the camera field of view or hidden
    behind other objects are dropped.
  * Triangulation error grows with distance: sigma = noise_std + noise_std_per_m2 * r^2.

Laser frame: x forward, y left, z up. A spot at yaw psi and fan angle phi is
    x = r cos(phi) cos(psi),  y = r cos(phi) sin(psi),  z = r sin(phi)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import mujoco
import numpy as np

from simulation.environment import Environment

from .common import (
    FORWARD_Y_MOUNT,
    MUJOCO_TO_OPENCV,
    RayCaster,
    angle_grid_deg,
    spherical_to_cartesian,
)
from .rgb_camera import CameraIntrinsics


@dataclass
class LaserConfig:
    """Scan pattern and error model of the laser line scanner."""

    fan_range_deg: Tuple[float, float] = (-30.0, 30.0)  # angle inside the laser plane
    fan_step_deg: float = 1.0
    sweep_range_deg: Tuple[float, float] = (-60.0, 60.0)  # plane yaw range for a sweep
    sweep_step_deg: float = 2.0
    min_range: float = 0.2  # [m]
    max_range: float = 5.0  # [m]
    noise_std: float = 0.0  # [m], constant part of the triangulation error
    noise_std_per_m2: float = 0.0  # [m / m^2], distance-dependent part
    dropout_prob: float = 0.0
    check_camera_visibility: bool = True

    def __post_init__(self) -> None:
        if self.fan_step_deg <= 0.0 or self.sweep_step_deg <= 0.0:
            raise ValueError("angular steps must be positive")
        if self.fan_range_deg[0] > self.fan_range_deg[1]:
            raise ValueError("fan_range_deg must be (low, high)")
        if self.sweep_range_deg[0] > self.sweep_range_deg[1]:
            raise ValueError("sweep_range_deg must be (low, high)")
        if not 0.0 <= self.min_range < self.max_range:
            raise ValueError("need 0 <= min_range < max_range")
        if self.noise_std < 0.0 or self.noise_std_per_m2 < 0.0:
            raise ValueError("noise parameters must be >= 0")
        if not 0.0 <= self.dropout_prob <= 1.0:
            raise ValueError("dropout_prob must be in [0, 1]")


@dataclass
class LaserScan:
    """Laser spots on a (plane yaw x fan angle) grid."""

    time: float
    yaw: np.ndarray  # (n_lines,) [rad]
    fan: np.ndarray  # (n_fan,) [rad]
    ranges: np.ndarray  # (n_lines, n_fan) [m], NaN = no measurement
    geom_ids: np.ndarray  # (n_lines, n_fan) hit geom id, -1 = none (ground truth, evaluation only)

    @property
    def valid(self) -> np.ndarray:
        """Boolean mask (n_lines, n_fan) of measured spots."""
        return np.isfinite(self.ranges)

    def points(self) -> np.ndarray:
        """Measured spots as (M, 3) Cartesian points in the laser frame."""
        yaw, fan = np.meshgrid(self.yaw, self.fan, indexing="ij")
        mask = self.valid
        return spherical_to_cartesian(self.ranges[mask], yaw[mask], fan[mask])


class LaserScanner:
    """Fixed laser line scanner mounted at the environment's laser site."""

    _VISIBILITY_TOLERANCE = 5e-3  # [m] slack when comparing camera-ray and laser-ray hits

    def __init__(
        self,
        env: Environment,
        config: Optional[LaserConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._env = env
        self.config = config or LaserConfig()
        model = env.model

        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, env.config.laser_site_name)
        if site_id < 0:
            raise RuntimeError(f"site {env.config.laser_site_name!r} not found")
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, env.config.camera_name)
        if cam_id < 0:
            raise RuntimeError(f"camera {env.config.camera_name!r} not found")
        self._site_id = int(site_id)
        self._cam_id = int(cam_id)
        # Exclude the sensor's own body so rays never hit the housing.
        self._caster = RayCaster(model, env.data, exclude_body_id=int(model.site_bodyid[site_id]))
        self._rng = np.random.default_rng(seed)
        self._intrinsics = CameraIntrinsics.from_fovy(
            *env.config.camera_resolution, env.config.camera_fovy_deg
        )

        cfg = self.config
        self.fan = np.radians(angle_grid_deg(*cfg.fan_range_deg, cfg.fan_step_deg))
        self._default_yaws_deg = angle_grid_deg(*cfg.sweep_range_deg, cfg.sweep_step_deg)

    # --------------------------------------------------------------- geometry
    def ground_truth_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Ground-truth laser frame pose in world: (position (3,), rotation (3, 3)).

        Use for calibration evaluation only, never in the estimator.
        """
        data = self._env.data
        rotation = data.site_xmat[self._site_id].reshape(3, 3) @ FORWARD_Y_MOUNT
        return data.site_xpos[self._site_id].copy(), rotation

    def plane_world(self, yaw_deg: float) -> Tuple[np.ndarray, np.ndarray]:
        """Ground-truth laser plane in world: (unit normal (3,), point on plane (3,)).

        Every spot p of that line satisfies normal . (p - point) = 0.
        """
        origin, rotation = self.ground_truth_pose()
        yaw = np.radians(yaw_deg)
        normal_laser = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        return rotation @ normal_laser, origin

    # ----------------------------------------------------------------- scans
    def scan_line(self, yaw_deg: float = 0.0) -> LaserScan:
        """Scan with a single laser plane at the given yaw."""
        return self._scan(np.array([float(yaw_deg)]))

    def scan_sweep(self, yaws_deg=None) -> LaserScan:
        """Scan with several plane yaws (swept laser).

        The scene is treated as a snapshot: obstacles do not move during the sweep.
        """
        yaws = self._default_yaws_deg if yaws_deg is None else np.atleast_1d(np.asarray(yaws_deg, dtype=float))
        return self._scan(yaws)

    def _scan(self, yaws_deg: np.ndarray) -> LaserScan:
        cfg = self.config
        origin, rotation = self.ground_truth_pose()
        yaw = np.radians(yaws_deg)
        yaw_grid, fan_grid = np.meshgrid(yaw, self.fan, indexing="ij")
        directions = spherical_to_cartesian(1.0, yaw_grid, fan_grid).reshape(-1, 3) @ rotation.T
        dist, geom_ids = self._caster.cast_many(origin, directions)

        ranges = np.where(dist >= 0.0, dist, np.nan)
        with np.errstate(invalid="ignore"):
            ranges[(ranges < cfg.min_range) | (ranges > cfg.max_range)] = np.nan
        if cfg.check_camera_visibility:
            ranges = self._drop_hidden_spots(ranges, origin, directions)

        valid = np.isfinite(ranges)
        if cfg.noise_std > 0.0 or cfg.noise_std_per_m2 > 0.0:
            sigma = cfg.noise_std + cfg.noise_std_per_m2 * ranges[valid] ** 2
            ranges[valid] += self._rng.normal(0.0, 1.0, size=sigma.shape) * sigma
        if cfg.dropout_prob > 0.0:
            ranges[self._rng.random(ranges.shape) < cfg.dropout_prob] = np.nan

        geom_ids = np.where(np.isfinite(ranges), geom_ids, -1)
        shape = (len(yaw), len(self.fan))
        return LaserScan(
            time=self._env.time,
            yaw=yaw,
            fan=self.fan.copy(),
            ranges=ranges.reshape(shape),
            geom_ids=geom_ids.reshape(shape),
        )

    # ------------------------------------------------------ camera visibility
    def _drop_hidden_spots(self, ranges: np.ndarray, origin: np.ndarray, directions: np.ndarray) -> np.ndarray:
        hit = np.flatnonzero(np.isfinite(ranges))
        if hit.size == 0:
            return ranges
        spots = origin + ranges[hit, None] * directions[hit]
        visible = self._visible_to_camera(spots)
        ranges = ranges.copy()
        ranges[hit[~visible]] = np.nan
        return ranges

    def _visible_to_camera(self, points_world: np.ndarray) -> np.ndarray:
        """True for spots inside the camera field of view with a clear line of sight."""
        data = self._env.data
        cam_pos = data.cam_xpos[self._cam_id].copy()
        rot_wc = data.cam_xmat[self._cam_id].reshape(3, 3) @ MUJOCO_TO_OPENCV
        points_cam = (points_world - cam_pos) @ rot_wc  # row-vector form of R^T (p - t)

        intr = self._intrinsics
        uv = intr.project(points_cam)
        with np.errstate(invalid="ignore"):
            in_view = (
                (points_cam[:, 2] > 1e-6)
                & (uv[:, 0] >= 0.0) & (uv[:, 0] <= intr.width - 1)
                & (uv[:, 1] >= 0.0) & (uv[:, 1] <= intr.height - 1)
            )

        visible = np.zeros(len(points_world), dtype=bool)
        for i in np.flatnonzero(in_view):
            offset = points_world[i] - cam_pos
            distance = float(np.linalg.norm(offset))
            hit, _ = self._caster.cast(cam_pos, offset / distance)
            visible[i] = hit < 0.0 or hit >= distance - self._VISIBILITY_TOLERANCE
        return visible