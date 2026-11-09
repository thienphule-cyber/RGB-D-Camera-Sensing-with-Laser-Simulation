"""Interacting Multiple Model (IMM) estimator over several EKF/UKF models.

One cycle (Bar-Shalom):
    predict(dt)  mixing:   c_j = sum_i Pi_ij mu_i,   mu_{i|j} = Pi_ij mu_i / c_j
                        x0_j = sum_i mu_{i|j} x_i,  P0_j = sum_i mu_{i|j} (P_i + dx dx^T)
            then every filter predicts from its mixed state; mu = c.
    update(z,R)  every filter updates and returns its likelihood Lambda_j;
               mu_j is proportional to mu_j * Lambda_j (computed in the log domain).
    estimate     x = sum_j mu_j x_j,   P = sum_j mu_j (P_j + dx dx^T).

Several measurements at the same time (several sensors) are applied by calling
update() repeatedly: each call uses the current probabilities as the prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.special import logsumexp

from .common import StateFilter, check_dt, symmetrize
from .ekf import EKF
from .measurement_models import MeasurementModel
from .motion_models import ConstantVelocity2D, ConstantVelocity3D
from .ukf import UKF, UKFConfig


@dataclass(frozen=True, eq=False)
class IMMUpdateResult:
    """Outcome of one IMM update."""

    innovation: np.ndarray  # mixture innovation (mu-weighted mean of the model innovations)
    innovation_cov: np.ndarray  # moment-matched mixture innovation covariance
    nis: float
    log_likelihood: float  # log of the mixture likelihood, log sum_j mu_j Lambda_j
    model_log_likelihoods: np.ndarray  # (M,) log Lambda_j
    probabilities: np.ndarray  # (M,) posterior model probabilities


class IMM:
    """IMM estimator.

    Args:
        filters: EKF/UKF instances, one per motion model, at the same time. The model
            names (filter.model.name) must be unique.
        transition: Markov matrix, transition[i, j] = P(model j at k+1 | model i at k).
        initial_probabilities: initial model probabilities (uniform when omitted).
        min_probability: floor of every model probability; without it a model that
            reaches probability 0 could never recover.
    """

    def __init__(
        self,
        filters: Sequence[StateFilter],
        transition,
        initial_probabilities=None,
        min_probability: float = 1e-4,
    ) -> None:
        self.filters = list(filters)
        if not self.filters:
            raise ValueError("need at least one filter")
        m = len(self.filters)
        names = [f.model.name for f in self.filters]
        if len(set(names)) != len(names):
            raise ValueError("model names must be unique")
        t0 = self.filters[0].time
        if any(abs(f.time - t0) > 1e-12 for f in self.filters):
            raise ValueError("all filters must share the same time")

        trans = np.asarray(transition, dtype=float)
        if trans.shape != (m, m):
            raise ValueError(f"transition must be {m}x{m}")
        if np.any(trans < 0.0) or not np.allclose(trans.sum(axis=1), 1.0, atol=1e-9):
            raise ValueError("every row of the transition matrix must be a probability distribution")
        if np.any(trans.sum(axis=0) <= 0.0):
            raise ValueError("every model must be reachable (a zero column in transition)")
        if not 0.0 <= min_probability < 1.0 / m:
            raise ValueError("min_probability must be in [0, 1/M)")

        if initial_probabilities is None:
            mu = np.full(m, 1.0 / m)
        else:
            mu = np.asarray(initial_probabilities, dtype=float).reshape(-1)
            if mu.shape != (m,) or np.any(mu < 0.0) or not np.isclose(mu.sum(), 1.0, atol=1e-9):
                raise ValueError("initial_probabilities must be a probability distribution of length M")

        self._names = names
        self._transition = trans
        self._min_probability = float(min_probability)
        self._mu = self._floor(mu)

    def _floor(self, mu: np.ndarray) -> np.ndarray:
        mu = np.maximum(mu, self._min_probability)
        return mu / mu.sum()

    # ------------------------------------------------------------------ queries
    @property
    def model_names(self):
        return list(self._names)

    @property
    def probabilities(self) -> np.ndarray:
        return self._mu.copy()

    @property
    def probabilities_by_name(self) -> Dict[str, float]:
        return {name: float(p) for name, p in zip(self._names, self._mu)}

    def probability(self, name: str) -> float:
        return float(self._mu[self._names.index(name)])

    @property
    def most_likely_model(self) -> str:
        return self._names[int(np.argmax(self._mu))]

    @property
    def time(self) -> float:
        return self.filters[0].time

    def _combine(self):
        xs = np.array([f.x for f in self.filters])
        x = self._mu @ xs
        p = np.zeros((xs.shape[1], xs.shape[1]))
        for mu_j, f, xj in zip(self._mu, self.filters, xs):
            d = xj - x
            p += mu_j * (f.P + np.outer(d, d))
        return x, symmetrize(p)

    @property
    def x(self) -> np.ndarray:
        """Combined state estimate (probability-weighted mean)."""
        return self._combine()[0]

    @property
    def P(self) -> np.ndarray:
        """Combined covariance (includes the spread between the model estimates)."""
        return self._combine()[1]

    # ------------------------------------------------------------------ predict
    def predict(self, dt: float) -> None:
        dt = check_dt(dt)
        mu = self._mu
        c = self._transition.T @ mu  # predicted model probabilities
        mix = self._transition * mu[:, None] / c[None, :]  # mix[i, j] = mu_{i|j}

        xs = [f.x for f in self.filters]
        ps = [f.P for f in self.filters]
        mixed = []
        for j in range(len(self.filters)):
            x0 = sum(mix[i, j] * xs[i] for i in range(len(xs)))
            p0 = sum(mix[i, j] * (ps[i] + np.outer(xs[i] - x0, xs[i] - x0)) for i in range(len(xs)))
            mixed.append((x0, symmetrize(p0)))
        for f, (x0, p0) in zip(self.filters, mixed):
            f.set_state(x0, p0)
            f.predict(dt)
        self._mu = self._floor(c / c.sum())

    def predict_to(self, time: float) -> None:
        dt = float(time) - self.time
        if dt < -1e-12:
            raise ValueError("cannot predict backwards in time")
        self.predict(max(dt, 0.0))

    # ------------------------------------------------------------------- update
    def update(self, z, r, meas_model: Optional[MeasurementModel] = None) -> IMMUpdateResult:
        prior = self._mu.copy()
        results = [f.update(z, r, meas_model) for f in self.filters]
        log_lik = np.array([res.log_likelihood for res in results])

        with np.errstate(divide="ignore"):
            log_post = np.log(prior) + log_lik
        total = float(logsumexp(log_post))
        if np.isfinite(total):
            self._mu = self._floor(np.exp(log_post - total))

        ys = np.array([res.innovation for res in results])
        y_bar = prior @ ys
        s_mix = np.zeros_like(results[0].innovation_cov)
        for w, res, y in zip(prior, results, ys):
            d = y - y_bar
            s_mix += w * (res.innovation_cov + np.outer(d, d))
        s_mix = symmetrize(s_mix)
        nis = float(y_bar @ np.linalg.solve(s_mix, y_bar))

        return IMMUpdateResult(
            innovation=y_bar,
            innovation_cov=s_mix,
            nis=nis,
            log_likelihood=total,
            model_log_likelihoods=log_lik,
            probabilities=self._mu.copy(),
        )

    def update_measurement(self, measurement) -> IMMUpdateResult:
        """Update with an object that has `position` (3,) and `covariance` (3, 3)."""
        return self.update(measurement.position, measurement.covariance)

    def gating_distance(self, z, r, meas_model: Optional[MeasurementModel] = None) -> float:
        """Squared Mahalanobis distance of z to the moment-matched mixture prediction."""
        mm = StateFilter._resolve_model(meas_model)
        zv = np.asarray(z, dtype=float).reshape(-1)
        ys, ss = [], []
        for f in self.filters:
            z_pred, s = f.predict_measurement(r, mm)
            ys.append(mm.residual(zv, z_pred))
            ss.append(s)
        ys = np.array(ys)
        y_bar = self._mu @ ys
        s_mix = np.zeros_like(ss[0])
        for w, s, y in zip(self._mu, ss, ys):
            d = y - y_bar
            s_mix += w * (s + np.outer(d, d))
        return float(y_bar @ np.linalg.solve(symmetrize(s_mix), y_bar))


def build_imm(
    kind: str,
    x0,
    p0,
    accel_std_2d: float = 0.5,
    accel_std_3d: float = 1.0,
    z_walk_std: float = 0.005,
    stay_probability: float = 0.98,
    initial_probabilities=(0.5, 0.5),
    min_probability: float = 1e-4,
    time: float = 0.0,
    ukf_config: Optional[UKFConfig] = None,
) -> IMM:
    """Standard 2D / 3D IMM ("CV2D" and "CV3D") built from EKFs or UKFs.

    kind: "ekf" or "ukf". stay_probability is the probability of staying in the same
    model between two steps (the model switching probability is 1 - stay_probability).
    """
    kind = kind.lower()
    if kind not in ("ekf", "ukf"):
        raise ValueError("kind must be 'ekf' or 'ukf'")
    if not 0.0 < stay_probability < 1.0:
        raise ValueError("stay_probability must be in (0, 1)")

    models = [
        ConstantVelocity2D(accel_std_2d, z_walk_std),
        ConstantVelocity3D(accel_std_3d),
    ]
    if kind == "ekf":
        filters = [EKF(m, x0, p0, time) for m in models]
    else:
        filters = [UKF(m, x0, p0, time, ukf_config) for m in models]
    stay, leave = stay_probability, 1.0 - stay_probability
    transition = [[stay, leave], [leave, stay]]
    return IMM(filters, transition, initial_probabilities, min_probability)