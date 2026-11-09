"""Localization metrics.

Confirmed tracks are matched to the ground-truth obstacles frame by frame (CLEAR-MOT
style: an earlier match is kept while it stays inside the gate, the rest is assigned
with the Hungarian algorithm). Position / velocity errors are computed on matched pairs
only, so they measure accuracy; coverage and identity quality are measured by
tracking_metrics.

Inputs are the arrays saved by main.py:
    gt_states   (T, N, 6)  ground truth [px, py, pz, vx, vy, vz]
    est_states  (T, K, 6)  track estimates (NaN while a track is not alive)
    est_status  (T, K)     0 = not alive, 1 = tentative, 2 = confirmed, 3 = lost
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

CONFIRMED = 2  # status code of a confirmed track in the tracking log
DEFAULT_GATE = 0.5  # matching gate [m]


# ----------------------------------------------------------------------- statistics
def _finite(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def rmse(values) -> float:
    """Root mean square of the finite entries (NaN when there are none)."""
    f = _finite(values)
    return float(np.sqrt(np.mean(f**2))) if f.size else float("nan")


def mean_of(values) -> float:
    f = _finite(values)
    return float(np.mean(f)) if f.size else float("nan")


def median_of(values) -> float:
    f = _finite(values)
    return float(np.median(f)) if f.size else float("nan")


def percentile_of(values, q: float) -> float:
    f = _finite(values)
    return float(np.percentile(f, q)) if f.size else float("nan")


# --------------------------------------------------------------------- obstacle info
@dataclass(frozen=True)
class ObstacleInfo:
    obstacle_id: str
    geometry: str
    motion_dim: str  # "2D" or "3D"


def obstacle_info(ids: Sequence[str]) -> Dict[str, ObstacleInfo]:
    """Geometry and motion dimension of each obstacle id (from the default obstacle list)."""
    specs = {}
    try:
        from simulation.obstacle_manager import default_obstacles

        specs = {s.obstacle_id: s for s in default_obstacles()}
    except Exception:  # noqa: BLE001 - fall back to "unknown"
        pass
    out: Dict[str, ObstacleInfo] = {}
    for oid in ids:
        spec = specs.get(str(oid))
        out[str(oid)] = ObstacleInfo(
            str(oid),
            spec.geometry if spec is not None else "unknown",
            spec.motion_dim if spec is not None else "unknown",
        )
    return out


# ------------------------------------------------------------------------ matching
@dataclass(frozen=True, eq=False)
class Matching:
    """Frame-by-frame assignment of confirmed tracks to obstacles."""

    assign: np.ndarray  # (T, N) track column index, -1 = obstacle not tracked
    distance: np.ndarray  # (T, N) distance of the match [m], NaN when unmatched
    gate: float


def match_tracks_to_truth(
    gt_states, est_states, est_status, gate: float = DEFAULT_GATE, status: int = CONFIRMED
) -> Matching:
    """Match the tracks with the given status to the obstacles in every frame."""
    gt = np.asarray(gt_states, dtype=float)
    est = np.asarray(est_states, dtype=float)
    stat = np.asarray(est_status)
    if gt.ndim != 3 or gt.shape[2] < 3:
        raise ValueError("gt_states must have shape (T, N, >=3)")
    n_frames, n_obj = gt.shape[:2]
    assign = np.full((n_frames, n_obj), -1, dtype=int)
    distance = np.full((n_frames, n_obj), np.nan)
    last_track = np.full(n_obj, -1, dtype=int)

    for t in range(n_frames):
        alive = np.flatnonzero(stat[t] == status)
        if alive.size == 0:
            continue
        d = np.linalg.norm(gt[t, :, None, :3] - est[t][alive, :3][None, :, :], axis=2)  # (N, A)
        column = {int(k): j for j, k in enumerate(alive)}

        taken = set()
        free_obj: List[int] = []
        for n in range(n_obj):  # keep the previous match while it is still valid
            j = column.get(int(last_track[n]))
            if j is not None and d[n, j] <= gate and j not in taken:
                assign[t, n] = alive[j]
                distance[t, n] = d[n, j]
                taken.add(j)
            else:
                free_obj.append(n)

        free_trk = [j for j in range(alive.size) if j not in taken]
        if free_obj and free_trk:
            sub = d[np.ix_(free_obj, free_trk)]
            cost = np.where(sub <= gate, sub, 1e6)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if sub[r, c] <= gate:
                    assign[t, free_obj[r]] = alive[free_trk[c]]
                    distance[t, free_obj[r]] = sub[r, c]

        for n in range(n_obj):
            if assign[t, n] >= 0:
                last_track[n] = assign[t, n]
    return Matching(assign, distance, float(gate))


def _matched_indices(matching: Matching):
    tt, nn = np.nonzero(matching.assign >= 0)
    return tt, nn, matching.assign[tt, nn]


# ------------------------------------------------------------------------- errors
def _error_vectors(gt_states, est_states, matching: Matching, columns: slice) -> np.ndarray:
    gt = np.asarray(gt_states, dtype=float)
    est = np.asarray(est_states, dtype=float)
    out = np.full(matching.assign.shape + (3,), np.nan)
    tt, nn, kk = _matched_indices(matching)
    if tt.size:
        out[tt, nn] = est[tt, kk, columns] - gt[tt, nn, columns]
    return out


def position_error_vectors(gt_states, est_states, matching: Matching) -> np.ndarray:
    """(T, N, 3) estimate - truth, NaN where the obstacle is not tracked."""
    return _error_vectors(gt_states, est_states, matching, slice(0, 3))


def velocity_error_vectors(gt_states, est_states, matching: Matching) -> np.ndarray:
    return _error_vectors(gt_states, est_states, matching, slice(3, 6))


def position_error_norms(gt_states, est_states, matching: Matching) -> np.ndarray:
    """(T, N) Euclidean position error [m]."""
    return np.linalg.norm(position_error_vectors(gt_states, est_states, matching), axis=2)


def velocity_error_norms(gt_states, est_states, matching: Matching) -> np.ndarray:
    return np.linalg.norm(velocity_error_vectors(gt_states, est_states, matching), axis=2)


def localization_summary(gt_ids, gt_states, est_states, matching: Matching, infos) -> Dict[str, float]:
    """Overall and per-motion-class position / velocity errors."""
    pe = position_error_norms(gt_states, est_states, matching)
    ve = velocity_error_norms(gt_states, est_states, matching)
    out = {
        "pos_rmse_m": rmse(pe),
        "pos_mean_m": mean_of(pe),
        "pos_median_m": median_of(pe),
        "pos_p95_m": percentile_of(pe, 95.0),
        "vel_rmse_mps": rmse(ve),
    }
    for dim in ("2D", "3D"):
        cols = [n for n, oid in enumerate(gt_ids) if infos[oid].motion_dim == dim]
        out[f"pos_rmse_{dim.lower()}_m"] = rmse(pe[:, cols]) if cols else float("nan")
        out[f"vel_rmse_{dim.lower()}_mps"] = rmse(ve[:, cols]) if cols else float("nan")
    return out


def per_object_localization(gt_ids, gt_states, est_states, matching: Matching, infos) -> List[Dict[str, object]]:
    pos = position_error_vectors(gt_states, est_states, matching)
    vel = velocity_error_vectors(gt_states, est_states, matching)
    rows: List[Dict[str, object]] = []
    for n, oid in enumerate(gt_ids):
        p, v = pos[:, n, :], vel[:, n, :]
        pn, vn = np.linalg.norm(p, axis=1), np.linalg.norm(v, axis=1)
        rows.append(
            {
                "obstacle_id": oid,
                "geometry": infos[oid].geometry,
                "motion_dim": infos[oid].motion_dim,
                "matched_frames": int(np.isfinite(pn).sum()),
                "pos_rmse_m": rmse(pn),
                "pos_mean_m": mean_of(pn),
                "pos_p95_m": percentile_of(pn, 95.0),
                "rmse_x_m": rmse(p[:, 0]),
                "rmse_y_m": rmse(p[:, 1]),
                "rmse_z_m": rmse(p[:, 2]),
                "vel_rmse_mps": rmse(vn),
            }
        )
    return rows


def model_identification(gt_ids, est_p3d, matching: Matching, infos) -> Dict[str, float]:
    """IMM check: how often the most likely model matches the true motion dimension.

    NaN for runs without an IMM (a single CV3D filter always reports probability 1).
    """
    out = {"model_acc_2d": float("nan"), "model_acc_3d": float("nan")}
    p = np.asarray(est_p3d, dtype=float)
    finite = p[np.isfinite(p)]
    if finite.size == 0 or np.all(finite == 1.0):
        return out
    tt, nn, kk = _matched_indices(matching)
    if tt.size == 0:
        return out
    p3 = p[tt, kk]
    valid = np.isfinite(p3)
    predicted_3d = p3 > 0.5
    truth_3d = np.array([infos[gt_ids[n]].motion_dim == "3D" for n in nn])
    for key, mask in (("2d", ~truth_3d), ("3d", truth_3d)):
        sel = valid & mask
        if sel.any():
            out[f"model_acc_{key}"] = float(np.mean(predicted_3d[sel] == truth_3d[sel]))
    return out


def error_timeseries(times, gt_ids, track_ids, gt_states, est_states, matching: Matching) -> List[Dict[str, object]]:
    """One row per matched (frame, obstacle): errors and the matched track id."""
    pos = position_error_vectors(gt_states, est_states, matching)
    vel = velocity_error_vectors(gt_states, est_states, matching)
    tt, nn, kk = _matched_indices(matching)
    rows: List[Dict[str, object]] = []
    for t, n, k in zip(tt, nn, kk):
        rows.append(
            {
                "time_s": float(times[t]),
                "obstacle_id": gt_ids[n],
                "track_id": int(track_ids[k]),
                "pos_error_m": float(np.linalg.norm(pos[t, n])),
                "vel_error_mps": float(np.linalg.norm(vel[t, n])),
                "err_x_m": float(pos[t, n, 0]),
                "err_y_m": float(pos[t, n, 1]),
                "err_z_m": float(pos[t, n, 2]),
            }
        )
    return rows