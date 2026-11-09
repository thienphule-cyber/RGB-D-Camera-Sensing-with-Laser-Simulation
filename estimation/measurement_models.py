"""Measurement models h(x) for the 6-state obstacle model.

State: x = [px, py, pz, vx, vy, vz] in the world frame.

A model provides:
    h(x)            predicted measurement
    jacobian(x)     H = dh/dx, shape (dim, 6)   (used by the EKF)
    residual(a, b)  a - b, with angle wrapping where needed
    mean(zs, w)     weighted mean of measurements (circular for angles; used by the UKF)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .common_constants import STATE_DIM


def wrap_angle(angle):
    """Wrap an angle (or an array of angles) [rad] into [-pi, pi)."""
    return (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


class MeasurementModel(ABC):
    """Base class of measurement models."""

    dim: int = 0

    @abstractmethod
    def h(self, x) -> np.ndarray:
        """Predicted measurement (dim,) for the state x (6,)."""

    @abstractmethod
    def jacobian(self, x) -> np.ndarray:
        """Jacobian H (dim, 6) of h at x."""

    def residual(self, z, z_pred) -> np.ndarray:
        """Innovation z - z_pred (overridden when components are angles)."""
        return np.asarray(z, dtype=float) - np.asarray(z_pred, dtype=float)

    def mean(self, zs, weights) -> np.ndarray:
        """Weighted mean of measurements zs (N, dim)."""
        return np.asarray(weights, dtype=float) @ np.asarray(zs, dtype=float)


class PositionMeasurementModel(MeasurementModel):
    """Direct (Cartesian) position measurement: z = [px, py, pz] in the world frame."""

    dim = 3

    def h(self, x) -> np.ndarray:
        return np.asarray(x, dtype=float)[:3].copy()

    def jacobian(self, x) -> np.ndarray:
        h = np.zeros((3, STATE_DIM))
        h[:, :3] = np.eye(3)
        return h


class SphericalMeasurementModel(MeasurementModel):
    """Range, azimuth, elevation measurement of a position, as a LiDAR reports it.

    The sensor frame has x forward, y left, z up; azimuth is measured from +x toward +y
    and elevation above the xy plane:
        r = |p|,  az = atan2(y, x),  el = atan2(z, sqrt(x^2 + y^2)),   p = R^T (p_world - s)

    Args:
        sensor_position: world position s of the sensor (3,).
        rotation: R, the rotation from the sensor frame to the world frame (3, 3).
            Identity when omitted.
    """

    dim = 3

    def __init__(self, sensor_position, rotation=None) -> None:
        self.sensor_position = np.asarray(sensor_position, dtype=float).reshape(3).copy()
        rot = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
        if rot.shape != (3, 3):
            raise ValueError("rotation must be 3x3")
        if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-6) or np.linalg.det(rot) <= 0.0:
            raise ValueError("rotation must be a proper rotation matrix")
        self.rotation = rot.copy()

    def _to_sensor(self, p_world) -> np.ndarray:
        return self.rotation.T @ (np.asarray(p_world, dtype=float).reshape(3) - self.sensor_position)

    def measure(self, p_world) -> np.ndarray:
        """[range, azimuth, elevation] of a world position."""
        px, py, pz = self._to_sensor(p_world)
        r = float(np.sqrt(px * px + py * py + pz * pz))
        if r < 1e-9:
            raise ValueError("measurement point coincides with the sensor")
        return np.array([r, np.arctan2(py, px), np.arctan2(pz, np.hypot(px, py))])

    def to_world(self, z) -> np.ndarray:
        """World position (3,) for a [range, azimuth, elevation] measurement."""
        r, az, el = (float(v) for v in np.asarray(z, dtype=float).reshape(3))
        p_sensor = np.array(
            [r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)]
        )
        return self.sensor_position + self.rotation @ p_sensor

    def h(self, x) -> np.ndarray:
        return self.measure(np.asarray(x, dtype=float)[:3])

    def jacobian(self, x) -> np.ndarray:
        px, py, pz = self._to_sensor(np.asarray(x, dtype=float)[:3])
        rho2 = px * px + py * py
        rho = np.sqrt(rho2)
        r2 = rho2 + pz * pz
        r = np.sqrt(r2)
        if r < 1e-9 or rho < 1e-9:
            raise ValueError("Jacobian is singular on the sensor's vertical axis")
        j_sensor = np.array(
            [
                [px / r, py / r, pz / r],
                [-py / rho2, px / rho2, 0.0],
                [-px * pz / (r2 * rho), -py * pz / (r2 * rho), rho / r2],
            ]
        )
        h = np.zeros((3, STATE_DIM))
        h[:, :3] = j_sensor @ self.rotation.T
        return h

    def residual(self, z, z_pred) -> np.ndarray:
        diff = np.asarray(z, dtype=float) - np.asarray(z_pred, dtype=float)
        diff = diff.copy()
        diff[1] = wrap_angle(diff[1])  # azimuth difference must not jump by 2*pi
        return diff

    def mean(self, zs, weights) -> np.ndarray:
        zs = np.asarray(zs, dtype=float)
        w = np.asarray(weights, dtype=float)
        out = w @ zs
        out[1] = np.arctan2(w @ np.sin(zs[:, 1]), w @ np.cos(zs[:, 1]))  # circular mean
        return out


POSITION_MODEL = PositionMeasurementModel()