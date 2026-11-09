"""Track objects and their life cycle (tentative -> confirmed -> lost -> deleted).

State machine:

    TENTATIVE --(confirm_hits hits)--------------------> CONFIRMED
    TENTATIVE --(more than max_tentative_misses misses)-> DELETED
    CONFIRMED --(no update for lost_after_time)---------> LOST
    LOST      --(measurement associated again)----------> CONFIRMED  (reacquisitions += 1)
    LOST      --(no update for delete_after_time)-------> DELETED
    any       --(position uncertainty above limit)------> DELETED

While a track is LOST (for example occluded) its filter keeps predicting, so its
gate grows and a reappearing object can be matched to the old track.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from estimation import EKF, UKF, ConstantVelocity3D, build_imm, initial_state
from perception.object_features import ColorDescriptor

# (initial position (3,), position covariance (3, 3), time [s]) -> EKF / UKF / IMM
FilterFactory = Callable[[np.ndarray, np.ndarray, float], object]


class TrackStatus(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    DELETED = "deleted"


# ------------------------------------------------------------------ filter factory
def make_filter_factory(
    kind: str = "ekf",
    use_imm: bool = True,
    velocity_std: float = 1.0,
    accel_std: float = 1.0,
    accel_std_2d: float = 0.5,
    accel_std_3d: float = 1.0,
    z_walk_std: float = 0.005,
    stay_probability: float = 0.98,
) -> FilterFactory:
    """Factory that builds the filter of a new track from its first measurement.

    kind: "ekf" or "ukf".
    use_imm: True -> IMM over the 2D and 3D constant-velocity models;
        False -> one filter with the 3D constant-velocity model (accel_std).
    velocity_std: initial velocity standard deviation [m/s] (the velocity is unknown).
    """
    kind = kind.lower()
    if kind not in ("ekf", "ukf"):
        raise ValueError("kind must be 'ekf' or 'ukf'")
    if velocity_std <= 0.0:
        raise ValueError("velocity_std must be positive")

    def factory(position, covariance, time: float):
        x0, p0 = initial_state(position, covariance, velocity_std)
        if use_imm:
            return build_imm(
                kind,
                x0,
                p0,
                accel_std_2d=accel_std_2d,
                accel_std_3d=accel_std_3d,
                z_walk_std=z_walk_std,
                stay_probability=stay_probability,
                time=time,
            )
        model = ConstantVelocity3D(accel_std)
        if kind == "ekf":
            return EKF(model, x0, p0, time)
        return UKF(model, x0, p0, time)

    return factory


# ------------------------------------------------------------------------- colour
def blend_colors(old: Optional[ColorDescriptor], new: Optional[ColorDescriptor], alpha: float):
    """Exponential smoothing of colour descriptors: (1 - alpha) * old + alpha * new.

    The hue is blended on the circle. A missing descriptor is ignored.
    """
    if new is None:
        return old
    if old is None:
        return new
    w_old, w_new = 1.0 - alpha, alpha

    chromaticity = w_old * np.asarray(old.chromaticity) + w_new * np.asarray(new.chromaticity)
    total = float(chromaticity.sum())
    if total > 1e-12:
        chromaticity = chromaticity / total

    hue_old, hue_new = old.hue_deg, new.hue_deg
    if np.isnan(hue_old):
        hue = hue_new
    elif np.isnan(hue_new):
        hue = hue_old
    else:
        a0, a1 = np.radians(hue_old), np.radians(hue_new)
        hue = float(
            np.degrees(
                np.arctan2(
                    w_old * np.sin(a0) + w_new * np.sin(a1),
                    w_old * np.cos(a0) + w_new * np.cos(a1),
                )
            )
            % 360.0
        )
    return ColorDescriptor(
        chromaticity=chromaticity,
        hue_deg=hue,
        saturation=w_old * old.saturation + w_new * new.saturation,
        value=w_old * old.value + w_new * new.value,
    )


# ------------------------------------------------------------------------- track
@dataclass(frozen=True, eq=False)
class TrackSample:
    """State of a track at the end of one frame (used for trajectories and evaluation)."""

    time: float
    position: np.ndarray  # (3,)
    velocity: np.ndarray  # (3,)
    status: TrackStatus
    updated: bool  # True if a measurement was applied in this frame
    motion_model: str


class Track:
    """One tracked object: a filter plus identity, status and appearance."""

    def __init__(self, track_id: int, filter, created_time: float, color=None) -> None:
        self.track_id = int(track_id)
        self.filter = filter
        self.status = TrackStatus.TENTATIVE
        self.created_time = float(created_time)
        self.last_update_time = float(created_time)
        self.confirmed_time: Optional[float] = None
        self.hits = 1  # frames with an associated measurement (creation counts as the first)
        self.consecutive_misses = 0
        self.total_misses = 0
        self.reacquisitions = 0
        self.updated = True  # a measurement was applied in the current frame
        self.delete_reason: Optional[str] = None  # "tentative_misses" | "lost_timeout" | "uncertain"
        self.color = color
        self.history: List[TrackSample] = []
        self._shape_votes: Dict[str, int] = {"sphere": 0, "other": 0}

    # ----------------------------------------------------------------- estimate
    @property
    def position(self) -> np.ndarray:
        return self.filter.x[:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.filter.x[3:]

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def covariance(self) -> np.ndarray:
        return self.filter.P

    @property
    def position_covariance(self) -> np.ndarray:
        return self.filter.P[:3, :3]

    @property
    def position_std(self) -> float:
        """Largest standard deviation of the position estimate [m]."""
        return float(np.sqrt(max(np.linalg.eigvalsh(self.position_covariance).max(), 0.0)))

    @property
    def motion_model(self) -> str:
        """Most likely motion model ("CV2D" / "CV3D" for the IMM, the model name otherwise)."""
        name = getattr(self.filter, "most_likely_model", None)
        return name if name is not None else self.filter.model.name

    @property
    def model_probabilities(self) -> Dict[str, float]:
        probabilities = getattr(self.filter, "probabilities_by_name", None)
        if probabilities is None:
            return {self.filter.model.name: 1.0}
        return dict(probabilities)

    @property
    def shape_hint(self) -> str:
        """"sphere" if most measurements looked like a sphere, otherwise "other"."""
        return "sphere" if self._shape_votes["sphere"] > self._shape_votes["other"] else "other"

    @property
    def age(self) -> int:
        """Number of frames the track has been alive (recorded samples)."""
        return len(self.history)

    # ------------------------------------------------------------------ helpers
    def gating_distance(self, z, r) -> float:
        """Squared Mahalanobis distance of a position measurement to the predicted state."""
        return float(self.filter.gating_distance(z, r))

    def time_since_update(self, time: float) -> float:
        return float(time) - self.last_update_time

    def add_shape_vote(self, measurement) -> None:
        hint = getattr(getattr(measurement, "features", None), "shape_hint", None)
        if hint == "sphere":
            self._shape_votes["sphere"] += 1
        elif hint is not None:
            self._shape_votes["other"] += 1

    def record(self, time: float) -> None:
        """Append the current state to the history."""
        self.history.append(
            TrackSample(
                time=float(time),
                position=self.position,
                velocity=self.velocity,
                status=self.status,
                updated=self.updated,
                motion_model=self.motion_model,
            )
        )

    def trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """(times (T,), positions (T, 3)) over the recorded history."""
        if not self.history:
            return np.zeros(0), np.zeros((0, 3))
        return (
            np.array([s.time for s in self.history]),
            np.array([s.position for s in self.history]),
        )

    def __repr__(self) -> str:
        return f"Track(id={self.track_id}, status={self.status.value}, hits={self.hits})"


# ------------------------------------------------------------------------ manager
@dataclass(frozen=True)
class TrackManagerConfig:
    """Life-cycle parameters.

    confirm_hits: hits needed to confirm a tentative track (creation counts as one).
    max_tentative_misses: a tentative track is deleted after MORE than this many
        consecutive misses.
    lost_after_time: a confirmed track becomes LOST after this long [s] without update.
    delete_after_time: a lost track is deleted after this long [s] without update.
    max_position_std: tracks whose position uncertainty [m] grows above this are deleted.
    color_alpha: smoothing factor of the track colour (0 = never change, 1 = latest only).
    """

    confirm_hits: int = 3
    max_tentative_misses: int = 2
    lost_after_time: float = 0.5
    delete_after_time: float = 3.0
    max_position_std: float = 1.0
    color_alpha: float = 0.3

    def __post_init__(self) -> None:
        if self.confirm_hits < 1:
            raise ValueError("confirm_hits must be >= 1")
        if self.max_tentative_misses < 0:
            raise ValueError("max_tentative_misses must be >= 0")
        if self.lost_after_time <= 0.0:
            raise ValueError("lost_after_time must be positive")
        if self.delete_after_time < self.lost_after_time:
            raise ValueError("delete_after_time must be >= lost_after_time")
        if self.max_position_std <= 0.0:
            raise ValueError("max_position_std must be positive")
        if not 0.0 <= self.color_alpha <= 1.0:
            raise ValueError("color_alpha must be in [0, 1]")


class TrackManager:
    """Owns the tracks and applies the life-cycle rules.

    Frame order (driven by ObjectTracker):
        predict(t) -> register_hit / register_miss ... -> create_track ... -> finalize(t)
    """

    def __init__(self, filter_factory: FilterFactory, config: Optional[TrackManagerConfig] = None) -> None:
        self._factory = filter_factory
        self.config = config or TrackManagerConfig()
        self._alive: List[Track] = []
        self._deleted: List[Track] = []
        self._next_id = 1

    # ------------------------------------------------------------------- queries
    @property
    def tracks(self) -> List[Track]:
        """Alive tracks (tentative, confirmed and lost), ordered by id."""
        return list(self._alive)

    @property
    def confirmed_tracks(self) -> List[Track]:
        return [t for t in self._alive if t.status is TrackStatus.CONFIRMED]

    @property
    def tentative_tracks(self) -> List[Track]:
        return [t for t in self._alive if t.status is TrackStatus.TENTATIVE]

    @property
    def lost_tracks(self) -> List[Track]:
        return [t for t in self._alive if t.status is TrackStatus.LOST]

    @property
    def deleted_tracks(self) -> List[Track]:
        """Archive of deleted tracks (their histories stay available for evaluation)."""
        return list(self._deleted)

    @property
    def num_created(self) -> int:
        return self._next_id - 1

    def get(self, track_id: int) -> Optional[Track]:
        """The alive track with this id, or None."""
        for track in self._alive:
            if track.track_id == track_id:
                return track
        return None

    # ----------------------------------------------------------------- life cycle
    def predict(self, time: float) -> None:
        """Propagate every alive track to `time` and clear the per-frame update flag."""
        time = float(time)
        for track in self._alive:
            dt = time - track.filter.time
            if dt < -1e-9:
                raise ValueError("cannot predict backwards in time")
            if dt > 1e-12:
                track.filter.predict_to(time)
            track.updated = False

    def create_track(self, measurement, time: float) -> Track:
        """Start a new track from a measurement. Ids are never reused."""
        filter_ = self._factory(
            np.asarray(measurement.position, dtype=float),
            np.asarray(measurement.covariance, dtype=float),
            float(time),
        )
        track = Track(self._next_id, filter_, time, measurement.color)
        self._next_id += 1
        track.add_shape_vote(measurement)
        if self.config.confirm_hits <= 1:
            track.status = TrackStatus.CONFIRMED
            track.confirmed_time = float(time)
        self._alive.append(track)
        return track

    def register_hit(self, track: Track, measurement, secondary: bool = False) -> None:
        """Update a track with a measurement.

        secondary=True is an additional measurement in a frame where the track already
        got its main one (for example a second sensor): the filter and the appearance
        are updated, but the hit count and the status are not changed.
        """
        track.filter.update_measurement(measurement)
        track.updated = True
        track.last_update_time = track.filter.time
        track.color = blend_colors(track.color, measurement.color, self.config.color_alpha)
        track.add_shape_vote(measurement)
        if secondary:
            return

        track.hits += 1
        track.consecutive_misses = 0
        if track.status is TrackStatus.TENTATIVE:
            if track.hits >= self.config.confirm_hits:
                track.status = TrackStatus.CONFIRMED
                track.confirmed_time = track.filter.time
        elif track.status is TrackStatus.LOST:
            track.status = TrackStatus.CONFIRMED
            track.reacquisitions += 1

    def register_miss(self, track: Track) -> None:
        """Count a frame without an associated measurement."""
        track.consecutive_misses += 1
        track.total_misses += 1

    def finalize(self, time: float) -> List[Track]:
        """Apply the status transitions and deletions. Returns the tracks deleted now."""
        cfg = self.config
        eps = 1e-9
        removed: List[Track] = []
        for track in list(self._alive):
            reason: Optional[str] = None
            since = float(time) - track.last_update_time

            if track.status is TrackStatus.TENTATIVE:
                if track.consecutive_misses > cfg.max_tentative_misses:
                    reason = "tentative_misses"
            elif track.status is TrackStatus.CONFIRMED:
                if since >= cfg.lost_after_time - eps:
                    track.status = TrackStatus.LOST
            if track.status is TrackStatus.LOST and since >= cfg.delete_after_time - eps:
                reason = "lost_timeout"
            if reason is None and track.position_std > cfg.max_position_std:
                reason = "uncertain"

            if reason is not None:
                track.status = TrackStatus.DELETED
                track.delete_reason = reason
                self._alive.remove(track)
                self._deleted.append(track)
                removed.append(track)
        return removed