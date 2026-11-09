"""Multi-sensor measurement fusion.

Two ways to use the detections of several sensors:

1. Sequential (asynchronous) updates - the estimator applies each Measurement as it
   arrives. RGB-D and the laser scanner share the camera, so their errors are
   correlated and applying both as independent measurements makes the filter
   overconfident. measurements_from_detections() therefore inflates their covariance
   (factor correlated_inflation) whenever two sensors of a correlation group are active.

2. Merged measurements - fuse() associates detections of different sensors that
   refer to the same object (Mahalanobis gate + Hungarian) and merges them into one
   Measurement. Sensors of the same correlation group are merged with covariance
   intersection (consistent under unknown correlation), and the groups are then merged
   with the information filter (independent sources). No extra inflation is applied here.

Ground truth is never used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

from perception.object_detection import Detection
from perception.object_features import ColorDescriptor, ObjectFeatures

from .measurement_covariance import (
    CovarianceConfig,
    line_of_sight,
    measurement_covariance,
    uses_sphere_center,
)


# ------------------------------------------------------------------ measurement
@dataclass(frozen=True, eq=False)
class Measurement:
    """A 3D position measurement in the world frame, ready for the estimator."""

    sensors: Tuple[str, ...]  # contributing sensors (one entry unless fused)
    time: float
    position: np.ndarray  # (3,)
    covariance: np.ndarray  # (3, 3), symmetric positive definite
    color: Optional[ColorDescriptor] = None
    ambiguous: bool = False
    num_points: int = 0
    features: Optional[ObjectFeatures] = None

    def __post_init__(self) -> None:
        sensors = tuple(self.sensors)
        if not sensors:
            raise ValueError("a measurement needs at least one sensor")
        position = np.asarray(self.position, dtype=float).reshape(3)
        cov = np.asarray(self.covariance, dtype=float)
        if cov.shape != (3, 3):
            raise ValueError(f"covariance must be 3x3, got shape {cov.shape}")
        if not np.allclose(cov, cov.T, rtol=1e-6, atol=1e-12):
            raise ValueError("covariance must be symmetric")
        cov = 0.5 * (cov + cov.T)
        if np.linalg.eigvalsh(cov).min() <= 0.0:
            raise ValueError("covariance must be positive definite")
        object.__setattr__(self, "sensors", sensors)
        object.__setattr__(self, "time", float(self.time))
        object.__setattr__(self, "position", position.copy())
        object.__setattr__(self, "covariance", cov.copy())

    @property
    def sensor(self) -> str:
        """Sensor name, or "a+b" for a fused measurement."""
        return "+".join(self.sensors)

    @property
    def is_fused(self) -> bool:
        return len(self.sensors) > 1


# ------------------------------------------------------------------ fusion maths
def mahalanobis_sq(z_a, cov_a, z_b, cov_b) -> float:
    """Squared Mahalanobis distance between two independent estimates (innovation S = Ca + Cb)."""
    diff = np.asarray(z_a, dtype=float).reshape(3) - np.asarray(z_b, dtype=float).reshape(3)
    s = np.asarray(cov_a, dtype=float) + np.asarray(cov_b, dtype=float)
    return float(diff @ np.linalg.solve(s, diff))


def fuse_independent(means, covariances) -> Tuple[np.ndarray, np.ndarray]:
    """Information-filter fusion of independent estimates. Returns (mean (3,), covariance (3, 3))."""
    mus = [np.asarray(m, dtype=float).reshape(3) for m in means]
    covs = [np.asarray(c, dtype=float).reshape(3, 3) for c in covariances]
    if not mus:
        raise ValueError("need at least one estimate")
    if len(mus) != len(covs):
        raise ValueError("means and covariances must have the same length")
    infos = [np.linalg.inv(c) for c in covs]
    info = sum(infos)
    cov = np.linalg.inv(info)
    cov = 0.5 * (cov + cov.T)
    mean = cov @ sum(i @ m for i, m in zip(infos, mus))
    return mean, cov


def covariance_intersection(means, covariances) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Covariance intersection of estimates with unknown cross-correlation.

        P^-1 = sum_i w_i P_i^-1,   x = P sum_i w_i P_i^-1 x_i,   w_i >= 0, sum w_i = 1

    The weights minimize trace(P). The result is consistent for any correlation, so it
    is never more confident than independent fusion. Returns (mean, covariance, weights).
    """
    mus = [np.asarray(m, dtype=float).reshape(3) for m in means]
    covs = [np.asarray(c, dtype=float).reshape(3, 3) for c in covariances]
    n = len(mus)
    if n == 0:
        raise ValueError("need at least one estimate")
    if n != len(covs):
        raise ValueError("means and covariances must have the same length")
    infos = [np.linalg.inv(c) for c in covs]

    def information(w) -> np.ndarray:
        return sum(float(wi) * info for wi, info in zip(w, infos))

    def cost(w) -> float:
        try:
            return float(np.trace(np.linalg.inv(information(w))))
        except np.linalg.LinAlgError:
            return 1e12

    if n == 1:
        weights = np.ones(1)
    else:
        uniform = np.full(n, 1.0 / n)
        scale = max(cost(uniform), 1e-30)  # normalize: covariances are tiny numbers
        result = minimize(
            lambda w: cost(w) / scale,
            uniform,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints=({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},),
        )
        candidates = [uniform]
        clipped = np.clip(result.x, 0.0, 1.0)
        if float(clipped.sum()) > 1e-12:
            candidates.append(clipped / clipped.sum())
        candidates.extend(np.eye(n)[i] for i in range(n))  # one estimate alone is always allowed
        costs = [cost(w) for w in candidates]
        best = min(costs)
        # First candidate within tolerance of the best: ties resolve to uniform weights.
        weights = next(w for w, c in zip(candidates, costs) if c <= best * (1.0 + 1e-9))

    cov = np.linalg.inv(information(weights))
    cov = 0.5 * (cov + cov.T)
    mean = cov @ sum(float(w) * (info @ mu) for w, info, mu in zip(weights, infos, mus))
    return mean, cov, np.asarray(weights, dtype=float)


# ------------------------------------------------------------------ configuration
@dataclass(frozen=True)
class FusionConfig:
    """Fusion parameters.

    correlation_groups: sensors whose errors are correlated. RGB-D and the laser
        scanner both rely on the same camera image, so they form one group.
    correlated_inflation: covariance factor (>= 1) for sequential updates of
        sensors that share a group with another active sensor.
    gate_chi2: Mahalanobis gate (squared) for cross-sensor association;
        11.34 is the 99% value of chi-square with 3 degrees of freedom.
    max_time_gap: detections further apart in time [s] are never merged.
        The position of a moving object changes by v * dt between the two
        timestamps, which the gate absorbs for small gaps.
    sensor_order: processing order of sensors when merging (unknown sensors go last).
    """

    correlation_groups: Tuple[Tuple[str, ...], ...] = (("rgbd", "laser"),)
    correlated_inflation: float = 2.0
    gate_chi2: float = 11.34
    max_time_gap: float = 0.01
    sensor_order: Tuple[str, ...] = ("rgbd", "laser", "lidar")

    def __post_init__(self) -> None:
        seen = set()
        for group in self.correlation_groups:
            for sensor in group:
                if sensor in seen:
                    raise ValueError(f"sensor {sensor!r} appears in more than one correlation group")
                seen.add(sensor)
        if self.correlated_inflation < 1.0:
            raise ValueError("correlated_inflation must be >= 1")
        if self.gate_chi2 <= 0.0:
            raise ValueError("gate_chi2 must be positive")
        if self.max_time_gap < 0.0:
            raise ValueError("max_time_gap must be >= 0")

    def group_key(self, sensor: str) -> str:
        """Identifier of the correlation group of a sensor (the sensor name if it has none)."""
        for index, group in enumerate(self.correlation_groups):
            if sensor in group:
                return f"group{index}"
        return sensor


# ------------------------------------------------------------------ fusion
class MeasurementFusion:
    """Converts detections to measurements and merges the ones of different sensors."""

    def __init__(
        self,
        sensor_positions: Mapping[str, Sequence[float]],
        covariance_config: Optional[CovarianceConfig] = None,
        config: Optional[FusionConfig] = None,
    ) -> None:
        if not sensor_positions:
            raise ValueError("sensor_positions must not be empty")
        self.covariance_config = covariance_config or CovarianceConfig()
        self.config = config or FusionConfig()
        self.sensor_positions: Dict[str, np.ndarray] = {}
        for name, position in sensor_positions.items():
            if name not in self.covariance_config.sensors:
                raise ValueError(f"no noise model for sensor {name!r}")
            self.sensor_positions[name] = np.asarray(position, dtype=float).reshape(3).copy()

    # --------------------------------------------------------------- queries
    @property
    def correlated_sensors(self) -> FrozenSet[str]:
        """Active sensors that share a correlation group with another active sensor."""
        counts: Dict[str, int] = {}
        for sensor in self.sensor_positions:
            key = self.config.group_key(sensor)
            counts[key] = counts.get(key, 0) + 1
        return frozenset(s for s in self.sensor_positions if counts[self.config.group_key(s)] >= 2)

    def _sensor_rank(self, sensor: str) -> Tuple[int, str]:
        order = self.config.sensor_order
        return (order.index(sensor) if sensor in order else len(order), sensor)

    # ------------------------------------------------------ detection -> measurement
    def measurement_from_detection(
        self, detection: Detection, inflate_correlated: bool = True
    ) -> Measurement:
        """Convert one detection to a Measurement (position, covariance, appearance).

        With inflate_correlated, the covariance of a sensor that shares a correlation
        group with another active sensor is multiplied by correlated_inflation.
        """
        sensor = detection.sensor
        if sensor not in self.sensor_positions:
            raise KeyError(f"sensor {sensor!r} is not active in this fusion")
        sensor_position = self.sensor_positions[sensor]

        cov = measurement_covariance(detection, sensor_position, self.covariance_config)
        position = np.asarray(detection.position, dtype=float).copy()
        shift = self.covariance_config.centroid_shift
        if shift > 0.0 and not uses_sphere_center(detection):
            u, _ = line_of_sight(position, sensor_position)
            position = position + shift * u  # compensate the visible-surface bias
        if inflate_correlated and sensor in self.correlated_sensors:
            cov = cov * self.config.correlated_inflation

        return Measurement(
            sensors=(sensor,),
            time=detection.time,
            position=position,
            covariance=cov,
            color=detection.color,
            ambiguous=detection.ambiguous,
            num_points=detection.num_points,
            features=detection.features,
        )

    def measurements_from_detections(
        self, detections: Sequence[Detection], inflate_correlated: bool = True
    ) -> List[Measurement]:
        """Convert detections, keeping their order."""
        return [self.measurement_from_detection(d, inflate_correlated) for d in detections]

    # ------------------------------------------------------------------ merging
    def combine(self, measurements: Sequence[Measurement]) -> Measurement:
        """Merge measurements already known to belong to one object (one per sensor).

        Sensors of the same correlation group go through covariance intersection, then
        the groups are merged with the information filter. Time = latest time; the color
        and features come from the member with the most points; the result is ambiguous
        only if every member is.
        """
        members = list(measurements)
        if not members:
            raise ValueError("need at least one measurement")
        if any(m.is_fused for m in members):
            raise ValueError("combine expects single-sensor measurements")
        names = [m.sensors[0] for m in members]
        if len(set(names)) != len(names):
            raise ValueError("at most one measurement per sensor")
        if len(members) == 1:
            return members[0]

        groups: Dict[str, List[Measurement]] = {}
        for m in members:
            groups.setdefault(self.config.group_key(m.sensors[0]), []).append(m)

        means, covs = [], []
        for group in groups.values():
            if len(group) == 1:
                means.append(group[0].position)
                covs.append(group[0].covariance)
            else:
                mean, cov, _ = covariance_intersection(
                    [m.position for m in group], [m.covariance for m in group]
                )
                means.append(mean)
                covs.append(cov)
        position, covariance = fuse_independent(means, covs)

        with_color = [m for m in members if m.color is not None]
        color = max(with_color, key=lambda m: m.num_points).color if with_color else None
        with_features = [m for m in members if m.features is not None]
        features = max(with_features, key=lambda m: m.num_points).features if with_features else None

        return Measurement(
            sensors=tuple(sorted(names, key=self._sensor_rank)),
            time=max(m.time for m in members),
            position=position,
            covariance=covariance,
            color=color,
            ambiguous=all(m.ambiguous for m in members),
            num_points=int(sum(m.num_points for m in members)),
            features=features,
        )

    def fuse(self, measurements: Sequence[Measurement]) -> List[Measurement]:
        """Associate single-sensor measurements across sensors and merge the matches.

        Sensors are processed in sensor_order. Each new sensor's measurements are
        matched to the fused estimates so far by Hungarian assignment on the squared
        Mahalanobis distance, gated by gate_chi2 and max_time_gap. Unmatched
        measurements stay on their own. Output order: measurements of the first
        sensor, then unmatched ones of the later sensors.
        """
        by_sensor: Dict[str, List[Measurement]] = {}
        for m in measurements:
            if m.is_fused:
                raise ValueError("fuse expects single-sensor measurements")
            by_sensor.setdefault(m.sensors[0], []).append(m)

        clusters: List[List[Measurement]] = []
        for sensor in sorted(by_sensor, key=self._sensor_rank):
            new = by_sensor[sensor]
            clusters = [[m] for m in new] if not clusters else self._merge(clusters, new)
        return [self.combine(cluster) for cluster in clusters]

    def fuse_detections(self, detections: Sequence[Detection]) -> List[Measurement]:
        """Convenience: detections -> measurements (no inflation) -> merged measurements."""
        return self.fuse(self.measurements_from_detections(detections, inflate_correlated=False))

    def _merge(
        self, clusters: List[List[Measurement]], new: List[Measurement]
    ) -> List[List[Measurement]]:
        cfg = self.config
        estimates = [self.combine(cluster) for cluster in clusters]
        big = 1e6
        cost = np.full((len(clusters), len(new)), big)
        for i, estimate in enumerate(estimates):
            for j, m in enumerate(new):
                if abs(estimate.time - m.time) > cfg.max_time_gap:
                    continue
                d2 = mahalanobis_sq(estimate.position, estimate.covariance, m.position, m.covariance)
                if d2 <= cfg.gate_chi2:
                    cost[i, j] = d2

        rows, cols = linear_sum_assignment(cost)
        matched = set()
        for i, j in zip(rows, cols):
            if cost[i, j] < big:
                clusters[i].append(new[j])
                matched.add(j)
        for j, m in enumerate(new):
            if j not in matched:
                clusters.append([m])
        return clusters