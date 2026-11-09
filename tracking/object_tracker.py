"""Multi-object tracker: association, filtering and track management in one frame loop.

One call to update() processes one frame:
    1. predict all alive tracks to the frame time
    2. associate tracks and measurements (Mahalanobis gate + hue, Hungarian)
    3. update the matched tracks, count a miss for the unmatched ones
    4. use each leftover measurement as an additional update of a track that was
    already updated in this frame (a second sensor seeing the same object), or
    else start a new tentative track
    5. apply the life-cycle rules (confirm / lost / delete)
    6. record the state of every alive track

Ground truth is never used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .data_association import AssociationConfig, associate, association_cost
from .track_manager import (
    FilterFactory,
    Track,
    TrackManager,
    TrackManagerConfig,
    make_filter_factory,
)


@dataclass(frozen=True)
class TrackerConfig:
    """Tracker parameters.

    spawn_ambiguous: allow ambiguous measurements (probably several touching objects
        merged into one cluster) to start new tracks. Off by default: a merged cluster
        would create a ghost track between the objects.
    allow_secondary_updates: apply leftover measurements that fit an already updated
        track as additional updates instead of starting a duplicate track. This is what
        allows sequential (asynchronous) multi-sensor updates without prior fusion.
    """

    association: AssociationConfig = field(default_factory=AssociationConfig)
    manager: TrackManagerConfig = field(default_factory=TrackManagerConfig)
    spawn_ambiguous: bool = False
    allow_secondary_updates: bool = True


@dataclass(frozen=True)
class FrameResult:
    """What happened in the last frame (indices refer to the measurement list of the frame)."""

    time: float
    matches: List[Tuple[int, int]]  # (track id, measurement index): main updates
    secondary: List[Tuple[int, int]]  # (track id, measurement index): additional updates
    created: List[int]  # ids of the new tracks
    missed: List[int]  # ids of tracks without a main measurement
    deleted: List[int]  # ids of the tracks deleted in this frame
    dropped: List[int]  # measurement indices that were not used (ambiguous leftovers)


class ObjectTracker:
    """Multi-object tracker built on top of TrackManager."""

    def __init__(
        self,
        filter_factory: Optional[FilterFactory] = None,
        config: Optional[TrackerConfig] = None,
    ) -> None:
        self.config = config or TrackerConfig()
        self._factory = filter_factory or make_filter_factory()
        self.manager = TrackManager(self._factory, self.config.manager)
        self._time: Optional[float] = None
        self.last_result: Optional[FrameResult] = None

    # ------------------------------------------------------------------- queries
    @property
    def time(self) -> Optional[float]:
        """Time of the last processed frame (None before the first frame)."""
        return self._time

    @property
    def tracks(self) -> List[Track]:
        """Alive tracks (tentative, confirmed and lost)."""
        return self.manager.tracks

    @property
    def confirmed_tracks(self) -> List[Track]:
        return self.manager.confirmed_tracks

    @property
    def all_tracks(self) -> List[Track]:
        """Alive and deleted tracks ordered by id (for evaluation)."""
        return sorted(self.manager.tracks + self.manager.deleted_tracks, key=lambda t: t.track_id)

    def get(self, track_id: int) -> Optional[Track]:
        """The alive track with this id, or None."""
        return self.manager.get(track_id)

    def reset(self) -> None:
        """Forget all tracks and restart the id counter."""
        self.manager = TrackManager(self._factory, self.config.manager)
        self._time = None
        self.last_result = None

    # -------------------------------------------------------------------- update
    def update(self, measurements: Sequence, time: Optional[float] = None) -> List[Track]:
        """Process one frame and return the confirmed tracks.

        measurements: fusion.Measurement objects of this frame, treated as taken at `time`.
            Prefer one measurement per object (use MeasurementFusion.fuse_detections); extra
            measurements of an already updated object become additional updates.
        time: frame time [s]. Defaults to the latest measurement time; required when
            there are no measurements (the tracks then only coast).
        """
        meas = list(measurements)
        if time is None:
            if not meas:
                raise ValueError("time is required when there are no measurements")
            time = max(m.time for m in meas)
        time = float(time)
        if self._time is not None and time < self._time - 1e-12:
            raise ValueError("frames must be processed in time order")

        cfg = self.config
        manager = self.manager
        manager.predict(time)
        tracks = manager.tracks  # snapshot: indices of the association refer to this list
        assoc = associate(tracks, meas, cfg.association)

        matches: List[Tuple[int, int]] = []
        for i, j in assoc.matches:
            manager.register_hit(tracks[i], meas[j])
            matches.append((tracks[i].track_id, j))
        missed: List[int] = []
        for i in assoc.unmatched_tracks:
            manager.register_miss(tracks[i])
            missed.append(tracks[i].track_id)

        # Leftover measurements: better-supported (more points, not ambiguous) ones first.
        candidates: List[Track] = [tracks[i] for i, _ in assoc.matches]
        leftover = sorted(
            assoc.unmatched_measurements,
            key=lambda j: (meas[j].ambiguous, -meas[j].num_points, j),
        )
        secondary: List[Tuple[int, int]] = []
        created: List[int] = []
        dropped: List[int] = []
        for j in leftover:
            m = meas[j]
            if cfg.allow_secondary_updates:
                target = self._best_candidate(candidates, m)
                if target is not None:
                    manager.register_hit(target, m, secondary=True)
                    secondary.append((target.track_id, j))
                    continue
            if m.ambiguous and not cfg.spawn_ambiguous:
                dropped.append(j)
                continue
            track = manager.create_track(m, time)
            created.append(track.track_id)
            candidates.append(track)

        deleted = [t.track_id for t in manager.finalize(time)]
        for track in manager.tracks:
            track.record(time)

        self._time = time
        self.last_result = FrameResult(
            time=time,
            matches=matches,
            secondary=secondary,
            created=created,
            missed=missed,
            deleted=deleted,
            dropped=dropped,
        )
        return manager.confirmed_tracks

    def _best_candidate(self, candidates: Sequence[Track], measurement) -> Optional[Track]:
        """The candidate track with the lowest finite association cost, or None."""
        best: Optional[Track] = None
        best_cost = float("inf")
        for track in candidates:
            cost = association_cost(track, measurement, self.config.association)
            if cost < best_cost:
                best, best_cost = track, cost
        return best