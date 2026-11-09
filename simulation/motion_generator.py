"""Analytic trajectory generators.

Every trajectory returns an exact position and velocity for any time t, so the
ground-truth velocity is exact (no finite differences, no random collisions).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Tuple

import numpy as np


def _vec3(values: Sequence[float], name: str) -> np.ndarray:
    """Convert a sequence to a float array of shape (3,)."""
    arr = np.asarray(values, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have exactly 3 elements, got shape {arr.shape}")
    return arr


class Trajectory(ABC):
    """Base class for all obstacle trajectories."""

    @abstractmethod
    def position(self, t: float) -> np.ndarray:
        """Return the world position (3,) at time t [s]."""

    @abstractmethod
    def velocity(self, t: float) -> np.ndarray:
        """Return the world velocity (3,) at time t [s]."""

    def state(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (position, velocity) at time t."""
        return self.position(t), self.velocity(t)


def _reflect(p0: float, v: float, lo: float, hi: float, t: float) -> Tuple[float, float]:
    """Bounce a 1D point between lo and hi (triangle-wave motion).

    Returns (position, velocity) at time t. If the interval is degenerate
    (hi <= lo) the axis is pinned at lo with zero velocity.
    """
    length = hi - lo
    if length <= 0.0:
        return lo, 0.0
    if v == 0.0:
        return p0, 0.0
    unfolded = (p0 - lo) + v * t
    m = float(np.mod(unfolded, 2.0 * length))
    if m <= length:
        return lo + m, v
    return lo + 2.0 * length - m, -v


class LinearBounceTrajectory(Trajectory):
    """Straight-line motion that reflects off an axis-aligned box.

    To get 2D motion, set bounds_min[2] == bounds_max[2] == start[2] and
    velocity[2] == 0 (the Z axis is then pinned).
    """

    def __init__(
        self,
        start: Sequence[float],
        velocity: Sequence[float],
        bounds_min: Sequence[float],
        bounds_max: Sequence[float],
    ) -> None:
        self.start = _vec3(start, "start")
        self.vel = _vec3(velocity, "velocity")
        self.bounds_min = _vec3(bounds_min, "bounds_min")
        self.bounds_max = _vec3(bounds_max, "bounds_max")
        if np.any(self.bounds_max < self.bounds_min):
            raise ValueError("bounds_max must be >= bounds_min on every axis")
        if np.any(self.start < self.bounds_min) or np.any(self.start > self.bounds_max):
            raise ValueError("start must lie inside the bounds")

    def _axis_states(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        pos = np.zeros(3)
        vel = np.zeros(3)
        for i in range(3):
            pos[i], vel[i] = _reflect(
                self.start[i], self.vel[i], self.bounds_min[i], self.bounds_max[i], t
            )
        return pos, vel

    def position(self, t: float) -> np.ndarray:
        return self._axis_states(t)[0]

    def velocity(self, t: float) -> np.ndarray:
        return self._axis_states(t)[1]


class SinusoidalTrajectory(Trajectory):
    """Per-axis sinusoid: p = center + amplitude * sin(omega * t + phase).

    Different omegas per axis give a Lissajous curve with continuously varying
    velocity. Set amplitude[2] = 0 for planar (2D) motion.
    """

    def __init__(
        self,
        center: Sequence[float],
        amplitude: Sequence[float],
        omega: Sequence[float],
        phase: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.center = _vec3(center, "center")
        self.amplitude = _vec3(amplitude, "amplitude")
        self.omega = _vec3(omega, "omega")
        self.phase = _vec3(phase, "phase")

    def position(self, t: float) -> np.ndarray:
        return self.center + self.amplitude * np.sin(self.omega * t + self.phase)

    def velocity(self, t: float) -> np.ndarray:
        return self.amplitude * self.omega * np.cos(self.omega * t + self.phase)


class CircularTrajectory(Trajectory):
    """Circular motion in the XY plane with optional sinusoidal Z oscillation.

    With z_amplitude = 0 the motion is planar (2D); otherwise it is a 3D
    nonlinear (helix-like) path.
    """

    def __init__(
        self,
        center: Sequence[float],
        radius: float,
        omega: float,
        phase: float = 0.0,
        z_amplitude: float = 0.0,
        z_omega: float = 0.0,
    ) -> None:
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        self.center = _vec3(center, "center")
        self.radius = float(radius)
        self.omega = float(omega)
        self.phase = float(phase)
        self.z_amplitude = float(z_amplitude)
        self.z_omega = float(z_omega)

    def position(self, t: float) -> np.ndarray:
        a = self.omega * t + self.phase
        return np.array(
            [
                self.center[0] + self.radius * np.cos(a),
                self.center[1] + self.radius * np.sin(a),
                self.center[2] + self.z_amplitude * np.sin(self.z_omega * t),
            ]
        )

    def velocity(self, t: float) -> np.ndarray:
        a = self.omega * t + self.phase
        return np.array(
            [
                -self.radius * self.omega * np.sin(a),
                self.radius * self.omega * np.cos(a),
                self.z_amplitude * self.z_omega * np.cos(self.z_omega * t),
            ]
        )