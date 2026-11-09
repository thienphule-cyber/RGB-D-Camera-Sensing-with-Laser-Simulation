"""Data association between predicted tracks and measurements.

Cost of a (track, measurement) pair:

    d2   = (z - z_pred)^T S^-1 (z - z_pred)       squared Mahalanobis distance
    cost = d2 + color_weight * (hue_difference / hue_scale)^2

A pair is forbidden (cost = +inf) when d2 exceeds the chi-square gate, or when
both objects have a hue and the hues differ by more than max_hue_diff_deg. The
appearance term only matters when two objects come close to each other, which is
exactly when identity switches happen. Colorless objects (LiDAR, laser, grey) are
matched by position alone.

A "track" is any object with gating_distance(z, R) -> float and a `color` attribute
(ColorDescriptor or None); a "measurement" is any object with position, covariance
and color (fusion.Measurement).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from perception.object_features import hue_difference_deg

from .hungarian import solve_assignment


@dataclass(frozen=True)
class AssociationConfig:
    """Association parameters.

    gate_chi2: squared Mahalanobis gate. 11.34 is the 99% value of chi-square with
        3 degrees of freedom (a 3D position measurement).
    use_color: use the hue as an additional cue.
    color_weight: weight of the hue term in the cost (0 = hue only gates).
    hue_scale_deg: hue difference [deg] that adds color_weight to the cost.
    max_hue_diff_deg: hue differences above this [deg] forbid the pair.
    """

    gate_chi2: float = 11.34
    use_color: bool = True
    color_weight: float = 2.0
    hue_scale_deg: float = 30.0
    max_hue_diff_deg: float = 60.0

    def __post_init__(self) -> None:
        if self.gate_chi2 <= 0.0:
            raise ValueError("gate_chi2 must be positive")
        if self.color_weight < 0.0:
            raise ValueError("color_weight must be >= 0")
        if self.hue_scale_deg <= 0.0:
            raise ValueError("hue_scale_deg must be positive")
        if not 0.0 < self.max_hue_diff_deg <= 180.0:
            raise ValueError("max_hue_diff_deg must be in (0, 180]")


@dataclass(frozen=True, eq=False)
class AssociationResult:
    """Outcome of one association step."""

    matches: List[Tuple[int, int]]  # (track index, measurement index)
    unmatched_tracks: List[int]
    unmatched_measurements: List[int]
    cost: np.ndarray  # (n_tracks, n_measurements), +inf for forbidden pairs


def hue_distance_deg(color_a, color_b) -> Optional[float]:
    """Smallest hue difference [deg] in [0, 180], or None if either hue is unavailable."""
    if color_a is None or color_b is None:
        return None
    hue_a, hue_b = color_a.hue_deg, color_b.hue_deg
    if np.isnan(hue_a) or np.isnan(hue_b):
        return None  # unsaturated (grey) objects carry no hue
    return hue_difference_deg(hue_a, hue_b)


def association_cost(track, measurement, config: Optional[AssociationConfig] = None) -> float:
    """Cost of associating a measurement with a track (+inf if the pair is forbidden)."""
    cfg = config or AssociationConfig()
    d2 = float(track.gating_distance(measurement.position, measurement.covariance))
    if not np.isfinite(d2) or d2 > cfg.gate_chi2:
        return float("inf")
    cost = d2
    if cfg.use_color:
        diff = hue_distance_deg(track.color, measurement.color)
        if diff is not None:
            if diff > cfg.max_hue_diff_deg:
                return float("inf")
            cost += cfg.color_weight * (diff / cfg.hue_scale_deg) ** 2
    return cost


def build_cost_matrix(
    tracks: Sequence, measurements: Sequence, config: Optional[AssociationConfig] = None
) -> np.ndarray:
    """(n_tracks, n_measurements) matrix of association costs."""
    cfg = config or AssociationConfig()
    cost = np.full((len(tracks), len(measurements)), np.inf)
    for i, track in enumerate(tracks):
        for j, measurement in enumerate(measurements):
            cost[i, j] = association_cost(track, measurement, cfg)
    return cost


def associate(
    tracks: Sequence, measurements: Sequence, config: Optional[AssociationConfig] = None
) -> AssociationResult:
    """Globally optimal one-to-one association (Hungarian) of tracks and measurements."""
    cost = build_cost_matrix(tracks, measurements, config)
    result = solve_assignment(cost)
    return AssociationResult(
        matches=result.matches,
        unmatched_tracks=result.unmatched_rows,
        unmatched_measurements=result.unmatched_cols,
        cost=cost,
    )