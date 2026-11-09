"""Unscented Kalman filter (UKF) with additive noise."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .common import (
    STATE_DIM,
    StateFilter,
    UpdateResult,
    check_dt,
    gaussian_log_likelihood,
    make_psd,
    matrix_sqrt,
    symmetrize,
)
from .measurement_models import MeasurementModel


@dataclass(frozen=True)
class UKFConfig:
    """Scaled unscented transform parameters.

    The defaults (alpha = 1, beta = 2, kappa = 0) give non-negative weights, so the
    predicted covariances stay positive semi-definite.
    """

    alpha: float = 1.0
    beta: float = 2.0
    kappa: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if self.beta < 0.0:
            raise ValueError("beta must be >= 0")
        if STATE_DIM + self.kappa <= 0.0:
            raise ValueError("kappa must satisfy n + kappa > 0")


def sigma_weights(n: int, config: UKFConfig) -> Tuple[np.ndarray, np.ndarray, float]:
    """(mean weights, covariance weights, lambda) of the 2n + 1 sigma points."""
    lam = config.alpha**2 * (n + config.kappa) - n
    c = n + lam
    wm = np.full(2 * n + 1, 1.0 / (2.0 * c))
    wc = wm.copy()
    wm[0] = lam / c
    wc[0] = lam / c + (1.0 - config.alpha**2 + config.beta)
    return wm, wc, float(lam)


def sigma_points(x, p, lam: float) -> np.ndarray:
    """The 2n + 1 sigma points (rows) of N(x, P).

    Uses the symmetric square root, so singular covariances (planar motion) are fine.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    root = matrix_sqrt((n + lam) * np.asarray(p, dtype=float))
    return np.vstack([x, x + root.T, x - root.T])


class UKF(StateFilter):
    """UKF: propagates sigma points through f and h instead of linearizing them.

    With a linear motion model and a linear measurement model it gives exactly the
    same result as the Kalman filter (and the EKF); the two differ only when h or f
    are nonlinear (for example LiDAR range/azimuth/elevation measurements).
    """

    def __init__(self, model, x0, p0, time: float = 0.0, config: Optional[UKFConfig] = None) -> None:
        super().__init__(model, x0, p0, time)
        self.config = config or UKFConfig()
        self._wm, self._wc, self._lam = sigma_weights(STATE_DIM, self.config)

    def predict(self, dt: float) -> None:
        dt = check_dt(dt)
        points = sigma_points(self._x, self._p, self._lam)
        propagated = np.array([self.model.predict_state(pt, dt) for pt in points])
        mean = self._wm @ propagated
        d = propagated - mean
        cov = (self._wc[:, None] * d).T @ d + self.model.process_noise(dt)
        self._x = mean
        self._p = make_psd(cov)
        self._time += dt

    def _propagate_measurement(self, mm: MeasurementModel):
        points = sigma_points(self._x, self._p, self._lam)
        zs = np.array([mm.h(pt) for pt in points])
        z_pred = mm.mean(zs, self._wm)
        dz = np.array([mm.residual(zi, z_pred) for zi in zs])
        return points, z_pred, dz

    def predict_measurement(self, r, meas_model: Optional[MeasurementModel] = None):
        mm = self._resolve_model(meas_model)
        r = self._check_covariance(r, mm.dim)
        _, z_pred, dz = self._propagate_measurement(mm)
        s = (self._wc[:, None] * dz).T @ dz + r
        return z_pred, symmetrize(s)

    def update(self, z, r, meas_model: Optional[MeasurementModel] = None) -> UpdateResult:
        mm, z, r = self._prepare_measurement(z, r, meas_model)
        x, p = self._x, self._p
        points, z_pred, dz = self._propagate_measurement(mm)
        dx = points - x

        s = symmetrize((self._wc[:, None] * dz).T @ dz + r)
        pxz = (self._wc[:, None] * dx).T @ dz
        k = np.linalg.solve(s, pxz.T).T  # Pxz S^-1
        y = mm.residual(z, z_pred)

        self._x = x + k @ y
        self._p = make_psd(p - k @ s @ k.T)

        log_likelihood, nis = gaussian_log_likelihood(y, s)
        return UpdateResult(y, s, k, nis, log_likelihood)