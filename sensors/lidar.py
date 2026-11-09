"""LiDAR sensor model based on MuJoCo ray casting.

LiDAR frame: x forward, y left, z up. Azimuth is measured from +x toward +y,
elevation above the xy plane:
    x = r cos(el) cos(az),  y = r cos(el) sin(az),  z = r sin(el)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import mujoco
import numpy as np

from simulation.environment import Environment

from .common import (
    FORWARD_Y_MOUNT,
    RayCaster,
    angle_grid_deg,
    spherical_to_cartesian,
)


@dataclass
class LidarConfig:
    """Scan pattern and error model of the LiDAR."""

    azimuth_range_deg: Tuple[float, float] = (-90.0, 90.0)
    elevation_range_deg: Tuple[float, float] = (-30.0, 30.0)
    resolution_deg: float = 1.0
    min_range: float = 0.1  # [m]
    max_range: float = 8.0  # [m]
    range_noise_std: float = 0.0  # [m], Gaussian
    dropout_prob: float = 0.0  # probability that a valid return is lost

    def __post_init__(self) -> None:
        if self.resolution_deg <= 0.0:
            raise ValueError("resolution_deg must be positive")
        if self.azimuth_range_deg[0] > self.azimuth_range_deg[1]:
            raise ValueError("azimuth_range_deg must be (low, high)")
        if self.elevation_range_deg[0] > self.elevation_range_deg[1]:
            raise ValueError("elevation_range_deg must be (low, high)")
        if not -90.0 <= self.elevation_range_deg[0] <= self.elevation_range_deg[1] <= 90.0:
            raise ValueError("elevation must stay within [-90, 90] degrees")
        if not 0.0 <= self.min_range < self.max_range:
            raise ValueError("need 0 <= min_range < max_range")
        if self.range_noise_std < 0.0:
            raise ValueError("range_noise_std must be >= 0")
        if not 0.0 <= self.dropout_prob <= 1.0:
            raise ValueError("dropout_prob must be in [0, 1]")


@dataclass
class LidarScan:
    """One LiDAR scan on an (elevation x azimuth) grid."""

    time: float
    azimuth: np.ndarray  # (n_az,) [rad]
    elevation: np.ndarray  # (n_el,) [rad]
    ranges: np.ndarray  # (n_el, n_az) [m], NaN = no return
    geom_ids: np.ndarray  # (n_el, n_az) hit geom id, -1 = none (ground truth, evaluation only)

    @property
    def valid(self) -> np.ndarray:
        """Boolean mask (n_el, n_az) of beams with a return."""
        return np.isfinite(self.ranges)

    def polar(self) -> np.ndarray:
        """Valid returns as (M, 3) rows [range, azimuth, elevation]."""
        az, el = np.meshgrid(self.azimuth, self.elevation)
        mask = self.valid
        return np.stack([self.ranges[mask], az[mask], el[mask]], axis=1)

    def points(self) -> np.ndarray:
        """Valid returns as (M, 3) Cartesian points in the LiDAR frame."""
        az, el = np.meshgrid(self.azimuth, self.elevation)
        mask = self.valid
        return spherical_to_cartesian(self.ranges[mask], az[mask], el[mask])


class Lidar:
    """Fixed LiDAR mounted at the environment's lidar site."""

    def __init__(
        self,
        env: Environment,
        config: Optional[LidarConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._env = env
        self.config = config or LidarConfig()
        model = env.model

        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, env.config.lidar_site_name)
        if site_id < 0:
            raise RuntimeError(f"site {env.config.lidar_site_name!r} not found")
        self._site_id = int(site_id)
        # Exclude the sensor's own body so rays never hit the housing.
        self._caster = RayCaster(model, env.data, exclude_body_id=int(model.site_bodyid[site_id]))
        self._rng = np.random.default_rng(seed)

        cfg = self.config
        self.azimuth = np.radians(angle_grid_deg(*cfg.azimuth_range_deg, cfg.resolution_deg))
        self.elevation = np.radians(angle_grid_deg(*cfg.elevation_range_deg, cfg.resolution_deg))
        az_grid, el_grid = np.meshgrid(self.azimuth, self.elevation)
        self._directions = spherical_to_cartesian(1.0, az_grid, el_grid).reshape(-1, 3)

    @property
    def num_beams(self) -> int:
        return int(self._directions.shape[0])

    def ground_truth_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Ground-truth LiDAR pose in world: (position (3,), rotation (3, 3)).

        Use for calibration evaluation only, never in the estimator.
        """
        data = self._env.data
        rotation = data.site_xmat[self._site_id].reshape(3, 3) @ FORWARD_Y_MOUNT
        return data.site_xpos[self._site_id].copy(), rotation

    def scan(self) -> LidarScan:
        """Cast all beams in the current scene state and return one scan."""
        cfg = self.config
        origin, rotation = self.ground_truth_pose()
        directions = self._directions @ rotation.T
        dist, geom_ids = self._caster.cast_many(origin, directions)

        ranges = np.where(dist >= 0.0, dist, np.nan)
        with np.errstate(invalid="ignore"):
            ranges[(ranges < cfg.min_range) | (ranges > cfg.max_range)] = np.nan

        valid = np.isfinite(ranges)
        if cfg.range_noise_std > 0.0:
            ranges[valid] += self._rng.normal(0.0, cfg.range_noise_std, size=int(valid.sum()))
        if cfg.dropout_prob > 0.0:
            ranges[self._rng.random(ranges.shape) < cfg.dropout_prob] = np.nan

        geom_ids = np.where(np.isfinite(ranges), geom_ids, -1)
        shape = (len(self.elevation), len(self.azimuth))
        return LidarScan(
            time=self._env.time,
            azimuth=self.azimuth.copy(),
            elevation=self.elevation.copy(),
            ranges=ranges.reshape(shape),
            geom_ids=geom_ids.reshape(shape),
        )