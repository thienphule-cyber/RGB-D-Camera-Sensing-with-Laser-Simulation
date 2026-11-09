"""Shared helpers and the base class of the filters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .common_constants import STATE_DIM
from .measurement_models import POSITION_MODEL, MeasurementModel

__all__ = [
    "STATE_DIM",
    "StateFilter",
    "UpdateResult",
    "check_dt",
    "check_state",
    "gaussian_log_likelihood",
    "initial_state",
    "make_psd",
    "matrix_sqrt",
    "nees",
    "symmetrize",
]


# ------------------------------------------------------------------ linear algebra
def symmetrize(matrix) -> np.ndarray:
    """0.5 * (M + M^T): removes round-off asymmetry."""
    m = np.asarray(matrix, dtype=float)
    return 0.5 * (m + m.T)


def make_psd(matrix, floor: float = 0.0) -> np.ndarray:
    """Symmetric matrix with eigenvalues clipped at `floor` (repairs round-off)."""
    values, vectors = np.linalg.eigh(symmetrize(matrix))
    values = np.clip(values, floor, None)
    return symmetrize((vectors * values) @ vectors.T)


def matrix_sqrt(matrix) -> np.ndarray:
    """Symmetric positive semi-definite square root S, with S @ S = M.

    Works for singular matrices (unlike the Cholesky factorization), which the
    planar motion model produces.
    """
    values, vectors = np.linalg.eigh(symmetrize(matrix))
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)) @ vectors.T


def check_dt(dt) -> float:
    """Validate a time step [s] (finite and >= 0) and return it as a float."""
    value = float(dt)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("dt must be a finite number >= 0")
    return value


def check_state(x, p) -> Tuple[np.ndarray, np.ndarray]:
    """Validate a state mean (6,) and covariance (6, 6); returns copies."""
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    if x.shape != (STATE_DIM,):
        raise ValueError(f"state must have shape ({STATE_DIM},), got {x.shape}")
    if p.shape != (STATE_DIM, STATE_DIM):
        raise ValueError(f"covariance must have shape ({STATE_DIM}, {STATE_DIM}), got {p.shape}")
    if not (np.isfinite(x).all() and np.isfinite(p).all()):
        raise ValueError("state and covariance must be finite")
    if not np.allclose(p, p.T, rtol=1e-6, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    p = symmetrize(p)
    if np.linalg.eigvalsh(p).min() < -1e-9 * max(1.0, float(np.abs(p).max())):
        raise ValueError("covariance must be positive semi-definite")
    return x.copy(), p.copy()


# ---------------------------------------------------------------------- statistics
def gaussian_log_likelihood(innovation, s) -> Tuple[float, float]:
    """(log-likelihood, NIS) of an innovation with covariance s.

    log N(y; 0, S) = -0.5 * (m log(2 pi) + log det S + y^T S^-1 y);  NIS = y^T S^-1 y.
    """
    y = np.asarray(innovation, dtype=float).reshape(-1)
    s = np.asarray(s, dtype=float)
    try:
        chol = np.linalg.cholesky(s)  # succeeds only for positive definite matrices
    except np.linalg.LinAlgError as exc:
        raise ValueError("innovation covariance must be positive definite") from exc
    logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
    nis = float(y @ np.linalg.solve(s, y))
    return float(-0.5 * (len(y) * np.log(2.0 * np.pi) + logdet + nis)), nis


def nees(error, covariance) -> float:
    """Normalized estimation error squared: e^T P^-1 e (about the state dimension if consistent)."""
    e = np.asarray(error, dtype=float).reshape(-1)
    return float(e @ np.linalg.solve(np.asarray(covariance, dtype=float), e))


def initial_state(position, position_cov, velocity_std: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Initial (x0, P0) from the first position measurement.

    The velocity is unknown: zero mean with standard deviation `velocity_std` [m/s].
    """
    if velocity_std <= 0.0:
        raise ValueError("velocity_std must be positive")
    pos = np.asarray(position, dtype=float).reshape(3)
    cov = np.asarray(position_cov, dtype=float)
    if cov.shape != (3, 3):
        raise ValueError("position_cov must be 3x3")
    x = np.concatenate([pos, np.zeros(3)])
    p = np.zeros((STATE_DIM, STATE_DIM))
    p[:3, :3] = symmetrize(cov)
    p[3:, 3:] = velocity_std**2 * np.eye(3)
    return x, p


@dataclass(frozen=True, eq=False)
class UpdateResult:
    """Outcome of one measurement update."""

    innovation: np.ndarray  # y = z - h(x_pred) (angles wrapped)
    innovation_cov: np.ndarray  # S
    gain: np.ndarray  # K, shape (6, dim)
    nis: float  # y^T S^-1 y
    log_likelihood: float  # log N(y; 0, S)


# ------------------------------------------------------------------------ base class
class StateFilter(ABC):
    """Common interface of the EKF and the UKF.

    The filter keeps its own copy of the state mean x, covariance P and time [s].
    Measurements are (z, R) with an optional MeasurementModel (default: Cartesian position).
    """

    def __init__(self, model, x0, p0, time: float = 0.0) -> None:
        self.model = model
        self._x, self._p = check_state(x0, p0)
        self._time = float(time)

    # ----------------------------------------------------------------- state
    @property
    def x(self) -> np.ndarray:
        return self._x.copy()

    @property
    def P(self) -> np.ndarray:
        return self._p.copy()

    @property
    def time(self) -> float:
        return self._time

    def set_state(self, x, p, time: Optional[float] = None) -> None:
        """Replace the state (used by the IMM when it mixes the models)."""
        self._x, self._p = check_state(x, p)
        if time is not None:
            self._time = float(time)

    # ------------------------------------------------------------ prediction
    @abstractmethod
    def predict(self, dt: float) -> None:
        """Propagate the state and covariance by dt [s]."""

    def predict_to(self, time: float) -> None:
        """Propagate to an absolute time [s] (no backward prediction)."""
        dt = float(time) - self._time
        if dt < -1e-12:
            raise ValueError("cannot predict backwards in time")
        self.predict(max(dt, 0.0))

    # ----------------------------------------------------------- measurements
    @abstractmethod
    def predict_measurement(self, r, meas_model: Optional[MeasurementModel] = None):
        """Predicted measurement and innovation covariance: (z_pred (m,), S (m, m))."""

    @abstractmethod
    def update(self, z, r, meas_model: Optional[MeasurementModel] = None) -> UpdateResult:
        """Measurement update with measurement z and covariance r."""

    def update_measurement(self, measurement) -> UpdateResult:
        """Update with a Cartesian position measurement.

        Accepts any object with `position` (3,) and `covariance` (3, 3), such as
        fusion.Measurement.
        """
        return self.update(measurement.position, measurement.covariance)

    def gating_distance(self, z, r, meas_model: Optional[MeasurementModel] = None) -> float:
        """Squared Mahalanobis distance of z to the predicted measurement (for data association)."""
        mm, zv, rv = self._prepare_measurement(z, r, meas_model)
        z_pred, s = self.predict_measurement(rv, mm)
        y = mm.residual(zv, z_pred)
        return float(y @ np.linalg.solve(s, y))

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _resolve_model(meas_model: Optional[MeasurementModel]) -> MeasurementModel:
        return POSITION_MODEL if meas_model is None else meas_model

    @staticmethod
    def _check_covariance(r, dim: int) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        if r.shape != (dim, dim):
            raise ValueError(f"measurement covariance must be {dim}x{dim}, got shape {r.shape}")
        if not np.allclose(r, r.T, rtol=1e-6, atol=1e-12):
            raise ValueError("measurement covariance must be symmetric")
        r = symmetrize(r)
        if np.linalg.eigvalsh(r).min() <= 0.0:
            raise ValueError("measurement covariance must be positive definite")
        return r

    def _prepare_measurement(self, z, r, meas_model):
        mm = self._resolve_model(meas_model)
        zv = np.asarray(z, dtype=float).reshape(-1)
        if zv.shape != (mm.dim,):
            raise ValueError(f"measurement must have {mm.dim} elements, got {zv.shape[0]}")
        return mm, zv, self._check_covariance(r, mm.dim)