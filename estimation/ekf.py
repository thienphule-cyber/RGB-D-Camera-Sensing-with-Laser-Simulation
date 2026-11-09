"""Extended Kalman filter (EKF)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .common import (
    STATE_DIM,
    StateFilter,
    UpdateResult,
    check_dt,
    gaussian_log_likelihood,
    symmetrize,
)
from .measurement_models import MeasurementModel


class EKF(StateFilter):
    """EKF with a (possibly nonlinear) motion model and measurement model.

    Prediction:
        x = f(x),   P = F P F^T + Q                      (F = df/dx)
    Update:
        y = z - h(x),  S = H P H^T + R,  K = P H^T S^-1
        x = x + K y
        P = (I - K H) P (I - K H)^T + K R K^T            (Joseph form, keeps P positive semi-definite)
    """

    def predict(self, dt: float) -> None:
        dt = check_dt(dt)
        f = self.model.jacobian(self._x, dt)
        self._x = self.model.predict_state(self._x, dt)
        self._p = symmetrize(f @ self._p @ f.T + self.model.process_noise(dt))
        self._time += dt

    def predict_measurement(self, r, meas_model: Optional[MeasurementModel] = None):
        mm = self._resolve_model(meas_model)
        r = self._check_covariance(r, mm.dim)
        h = mm.jacobian(self._x)
        return mm.h(self._x), symmetrize(h @ self._p @ h.T + r)

    def update(self, z, r, meas_model: Optional[MeasurementModel] = None) -> UpdateResult:
        mm, z, r = self._prepare_measurement(z, r, meas_model)
        x, p = self._x, self._p
        h = mm.jacobian(x)
        y = mm.residual(z, mm.h(x))
        s = symmetrize(h @ p @ h.T + r)
        k = np.linalg.solve(s, h @ p).T  # P H^T S^-1 (S is symmetric)

        self._x = x + k @ y
        ikh = np.eye(STATE_DIM) - k @ h
        self._p = symmetrize(ikh @ p @ ikh.T + k @ r @ k.T)

        log_likelihood, nis = gaussian_log_likelihood(y, s)
        return UpdateResult(y, s, k, nis, log_likelihood)