"""Constant-velocity motion models for the 6-state obstacle model.

State: x = [px, py, pz, vx, vy, vz]. Process noise follows the white-noise
acceleration model, with q = accel_std^2 the acceleration noise intensity:

    Q = q * [[dt^4/4 I, dt^3/2 I],
             [dt^3/2 I, dt^2   I]]
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .common import STATE_DIM, check_dt


class MotionModel(ABC):
    """Base class of motion models x_{k+1} = f(x_k) + w."""

    planar: bool = False  # True when the model keeps z fixed (2D motion hypothesis)

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("name must not be empty")
        self.name = name

    @abstractmethod
    def transition_matrix(self, dt: float) -> np.ndarray:
        """State transition matrix F (6, 6)."""

    @abstractmethod
    def process_noise(self, dt: float) -> np.ndarray:
        """Process noise covariance Q (6, 6)."""

    def predict_state(self, x, dt: float) -> np.ndarray:
        """f(x): the propagated state (a new array)."""
        return self.transition_matrix(dt) @ np.asarray(x, dtype=float)

    def jacobian(self, x, dt: float) -> np.ndarray:
        """df/dx at x (equal to F for the linear models)."""
        return self.transition_matrix(dt)


def _acceleration_block(dt: float, q: float) -> np.ndarray:
    """2x2 white-noise-acceleration block for one (position, velocity) pair."""
    return q * np.array([[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]])


class ConstantVelocity3D(MotionModel):
    """Model B: free 3D constant-velocity motion."""

    planar = False

    def __init__(self, accel_std: float = 1.0, name: str = "CV3D") -> None:
        super().__init__(name)
        if accel_std < 0.0:
            raise ValueError("accel_std must be >= 0")
        self.accel_std = float(accel_std)

    def transition_matrix(self, dt: float) -> np.ndarray:
        dt = check_dt(dt)
        f = np.eye(STATE_DIM)
        f[0:3, 3:6] = dt * np.eye(3)
        return f

    def process_noise(self, dt: float) -> np.ndarray:
        dt = check_dt(dt)
        return np.kron(_acceleration_block(dt, self.accel_std**2), np.eye(3))


class ConstantVelocity2D(MotionModel):
    """Model A: planar constant-velocity motion; z is held and vz is forced to zero.

    F has a zero vz row, exactly like F_2D of the design. The XY axes use the
    white-noise-acceleration noise. For z a small random walk (z_walk_std in
    m / sqrt(s)) keeps the filter from becoming deaf to the measured height; there is
    no vertical velocity noise.
    """

    planar = True

    def __init__(self, accel_std: float = 1.0, z_walk_std: float = 0.005, name: str = "CV2D") -> None:
        super().__init__(name)
        if accel_std < 0.0 or z_walk_std < 0.0:
            raise ValueError("noise parameters must be >= 0")
        self.accel_std = float(accel_std)
        self.z_walk_std = float(z_walk_std)

    def transition_matrix(self, dt: float) -> np.ndarray:
        dt = check_dt(dt)
        f = np.eye(STATE_DIM)
        f[0, 3] = dt
        f[1, 4] = dt
        f[5, 5] = 0.0  # vz is not carried over
        return f

    def process_noise(self, dt: float) -> np.ndarray:
        dt = check_dt(dt)
        q = np.zeros((STATE_DIM, STATE_DIM))
        block = _acceleration_block(dt, self.accel_std**2)
        for axis in (0, 1):
            idx = [axis, axis + 3]
            q[np.ix_(idx, idx)] = block
        q[2, 2] = self.z_walk_std**2 * dt
        return q