"""Tracking metrics: MOTA, MOTP, IDF1, identity switches, fragmentation and coverage.

All metrics use the frame-by-frame matching of localization_metrics. A "detection" is a
confirmed track in a frame; every frame contains all N obstacles as ground-truth objects,
so an obstacle outside the sensors' view counts as a miss.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment

from .localization_metrics import CONFIRMED, DEFAULT_GATE, Matching, _finite


def _count_switches(track_ids: np.ndarray) -> int:
    """Number of times the matched track id of one obstacle changes."""
    return int(np.count_nonzero(np.diff(track_ids) != 0)) if track_ids.size > 1 else 0


def _count_fragments(tracked_frames: np.ndarray) -> int:
    """Number of interruptions: gaps between two tracked frames of one obstacle."""
    return int(np.count_nonzero(np.diff(tracked_frames) > 1)) if tracked_frames.size > 1 else 0


def tracking_metrics(
    gt_states, est_states, est_status, matching: Matching, gate: float = DEFAULT_GATE
) -> Dict[str, float]:
    """MOTA, MOTP, IDF1 and related counts of one run."""
    assign = matching.assign
    n_frames, n_obj = assign.shape
    status = np.asarray(est_status)
    confirmed = status == CONFIRMED
    gt_total = n_frames * n_obj
    matches = int((assign >= 0).sum())
    conf_total = int(confirmed.sum())
    misses = gt_total - matches
    false_positives = conf_total - matches

    id_switches = 0
    fragmentations = 0
    coverages: List[float] = []
    for n in range(n_obj):
        seq = assign[:, n]
        tracked = np.flatnonzero(seq >= 0)
        id_switches += _count_switches(seq[seq >= 0])
        fragmentations += _count_fragments(tracked)
        coverages.append(tracked.size / n_frames if n_frames else float("nan"))

    mota = 1.0 - (misses + false_positives + id_switches) / gt_total if gt_total else float("nan")
    dist = _finite(matching.distance)
    motp = float(np.mean(dist)) if dist.size else float("nan")

    # IDF1: global one-to-one matching of obstacles and track ids by overlap.
    idtp = 0
    n_tracks = status.shape[1]
    if n_obj and n_tracks and n_frames:
        gt = np.asarray(gt_states, dtype=float)
        est = np.asarray(est_states, dtype=float)
        overlap = np.zeros((n_obj, n_tracks))
        with np.errstate(invalid="ignore"):
            for n in range(n_obj):
                d = np.linalg.norm(est[:, :, :3] - gt[:, n, None, :3], axis=2)  # (T, K)
                overlap[n] = np.sum((d <= gate) & confirmed, axis=0)
        rows, cols = linear_sum_assignment(-overlap)
        idtp = int(overlap[rows, cols].sum())
    idfn = gt_total - idtp
    idfp = conf_total - idtp
    idf1 = 2.0 * idtp / (2.0 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) > 0 else float("nan")
    idp = idtp / conf_total if conf_total else float("nan")
    idr = idtp / gt_total if gt_total else float("nan")

    matched_columns = set(np.unique(assign[assign >= 0]).tolist())
    ever_confirmed = np.flatnonzero(confirmed.any(axis=0)) if n_tracks else np.zeros(0, dtype=int)
    ghost_tracks = sum(1 for k in ever_confirmed if int(k) not in matched_columns)

    cov = np.asarray(coverages, dtype=float)
    return {
        "mota": float(mota),
        "motp_m": motp,
        "idf1": float(idf1),
        "idp": float(idp),
        "idr": float(idr),
        "id_switches": int(id_switches),
        "fragmentations": int(fragmentations),
        "mean_coverage": float(np.nanmean(cov)) if cov.size else float("nan"),
        "mostly_tracked": int(np.sum(cov >= 0.8)),
        "mostly_lost": int(np.sum(cov <= 0.2)),
        "misses": int(misses),
        "false_positives": int(false_positives),
        "ghost_tracks": int(ghost_tracks),
        "tracks_ever_confirmed": int(ever_confirmed.size),
        "matches": matches,
    }


def per_object_tracking(gt_ids, matching: Matching) -> List[Dict[str, object]]:
    """Coverage and identity statistics of each obstacle."""
    n_frames = matching.assign.shape[0]
    rows: List[Dict[str, object]] = []
    for n, oid in enumerate(gt_ids):
        seq = matching.assign[:, n]
        ids = seq[seq >= 0]
        tracked = np.flatnonzero(seq >= 0)
        rows.append(
            {
                "obstacle_id": oid,
                "coverage": tracked.size / n_frames if n_frames else float("nan"),
                "id_switches": _count_switches(ids),
                "fragmentations": _count_fragments(tracked),
                "distinct_track_ids": int(np.unique(ids).size),
            }
        )
    return rows