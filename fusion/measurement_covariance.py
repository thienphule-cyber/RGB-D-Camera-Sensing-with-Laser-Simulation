"""Measurement covariance models for detections (world frame).

The covariance of a position measurement is built in the line-of-sight (LOS)
frame of the sensor that produced it:

    R = sigma_los^2 * u u^T + sigma_cross^2 * (I - u u^T)

    u            unit vector from the sensor to the detection
    sigma_los    range noise (constant + growing with range^2) combined with the
                uncertainty of the visible-surface centroid bias
    sigma_cross  angular noise times the range (cross-range error)

The covariance is then inflated when the detection is less trustworthy:
few cluster points (the mean of a small cluster is noisy) or a cluster flagged
ambiguous (probably several touching objects merged into one).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from perception.object_detection import Detection


@dataclass(frozen=True)
class SensorNoise:
    """Noise model of one sensor.

    range_sigma: constant range noise [m].
    range_sigma_per_m2: range noise growth [m / m^2]; range std = range_sigma + this * r^2.
    angular_sigma: angular noise [rad]; cross-range std = angular_sigma * r.
    """

    range_sigma: float
    range_sigma_per_m2: float = 0.0
    angular_sigma: float = 0.0

    def __post_init__(self) -> None:
        if self.range_sigma < 0.0 or self.range_sigma_per_m2 < 0.0 or self.angular_sigma < 0.0:
            raise ValueError("noise parameters must be >= 0")

    def range_std(self, r: float) -> float:
        """Standard deviation along the line of sight [m] at range r [m]."""
        return self.range_sigma + self.range_sigma_per_m2 * float(r) ** 2

    def cross_range_std(self, r: float) -> float:
        """Standard deviation perpendicular to the line of sight [m] at range r [m]."""
        return self.angular_sigma * float(r)


def default_sensor_noise() -> Dict[str, SensorNoise]:
    """Initial noise assumptions for the three sensors (tune them against your sensor configs)."""
    return {
        "rgbd": SensorNoise(range_sigma=0.005, range_sigma_per_m2=0.002, angular_sigma=0.003),
        "laser": SensorNoise(range_sigma=0.010, range_sigma_per_m2=0.002, angular_sigma=0.004),
        "lidar": SensorNoise(range_sigma=0.010, range_sigma_per_m2=0.0, angular_sigma=0.010),
    }


@dataclass
class CovarianceConfig:
    """Parameters of the covariance model.

    sensors: noise model per sensor name ("rgbd", "laser", "lidar").
    centroid_bias_sigma: extra LOS uncertainty [m] for centroid-mode positions. The
        centroid of the visible surface is biased toward the sensor by about half the
        object depth, which a zero-mean covariance can only cover by being larger.
    sphere_bias_sigma: extra LOS uncertainty [m] for sphere-centre positions.
    reference_points: clusters with fewer points get variance * reference_points / n.
    ambiguous_inflation: variance factor for ambiguous clusters (>= 1).
    min_sigma: floor of every standard deviation [m] (keeps R positive definite).
    centroid_shift: centroid-mode positions are moved this far [m] away from the
        sensor along the line of sight to compensate the bias (0 = no correction).
    """

    sensors: Dict[str, SensorNoise] = field(default_factory=default_sensor_noise)
    centroid_bias_sigma: float = 0.05
    sphere_bias_sigma: float = 0.005
    reference_points: int = 50
    ambiguous_inflation: float = 9.0
    min_sigma: float = 1e-3
    centroid_shift: float = 0.0

    def __post_init__(self) -> None:
        if not self.sensors:
            raise ValueError("sensors must not be empty")
        if self.centroid_bias_sigma < 0.0 or self.sphere_bias_sigma < 0.0:
            raise ValueError("bias sigmas must be >= 0")
        if self.reference_points < 1:
            raise ValueError("reference_points must be >= 1")
        if self.ambiguous_inflation < 1.0:
            raise ValueError("ambiguous_inflation must be >= 1")
        if self.min_sigma <= 0.0:
            raise ValueError("min_sigma must be positive")
        if self.centroid_shift < 0.0:
            raise ValueError("centroid_shift must be >= 0")


def sensor_position_from_transform(t_w_s) -> np.ndarray:
    """World position (3,) of a sensor frame, given its 4x4 transform T_W_S."""
    mat = np.asarray(t_w_s, dtype=float)
    if mat.shape != (4, 4):
        raise ValueError("transform must be 4x4")
    return mat[:3, 3].copy()


def uses_sphere_center(detection: Detection) -> bool:
    """True when the detection position is the fitted sphere centre (not the centroid)."""
    sphere = detection.features.sphere
    return bool(
        sphere is not None
        and np.allclose(detection.position, sphere.center, rtol=0.0, atol=1e-9)
    )


def line_of_sight(position, sensor_position) -> Tuple[np.ndarray, float]:
    """Unit vector from the sensor to `position` and the range [m]."""
    offset = np.asarray(position, dtype=float).reshape(3) - np.asarray(sensor_position, dtype=float).reshape(3)
    rng = float(np.linalg.norm(offset))
    if rng < 1e-9:
        raise ValueError("measurement coincides with the sensor position")
    return offset / rng, rng


def los_covariance(u, sigma_los: float, sigma_cross: float) -> np.ndarray:
    """3x3 covariance with std sigma_los along the unit vector u and sigma_cross perpendicular."""
    u = np.asarray(u, dtype=float).reshape(3)
    uu = np.outer(u, u)
    return sigma_los**2 * uu + sigma_cross**2 * (np.eye(3) - uu)


def inflation_factor(detection: Detection, config: CovarianceConfig) -> float:
    """Variance multiplier for few cluster points and ambiguous clusters (>= 1)."""
    factor = 1.0
    n = detection.num_points
    if n < config.reference_points:
        factor *= config.reference_points / max(n, 1)
    if detection.ambiguous:
        factor *= config.ambiguous_inflation
    return factor


def measurement_covariance(
    detection: Detection,
    sensor_position,
    config: CovarianceConfig = None,
) -> np.ndarray:
    """Covariance R (3x3, world frame) of a detection's position.

    sensor_position is the world position of the sensor that produced the detection.
    """
    cfg = config or CovarianceConfig()
    if detection.sensor not in cfg.sensors:
        raise KeyError(f"no noise model for sensor {detection.sensor!r}")
    noise = cfg.sensors[detection.sensor]

    u, rng = line_of_sight(detection.position, sensor_position)
    bias = cfg.sphere_bias_sigma if uses_sphere_center(detection) else cfg.centroid_bias_sigma
    sigma_los = max(float(np.hypot(noise.range_std(rng), bias)), cfg.min_sigma)
    sigma_cross = max(noise.cross_range_std(rng), cfg.min_sigma)

    cov = los_covariance(u, sigma_los, sigma_cross) * inflation_factor(detection, cfg)
    return 0.5 * (cov + cov.T)