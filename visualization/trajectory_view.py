"""Trajectory plots: tracks, ground truth, velocity vectors, errors and model probabilities.

Trajectory dictionaries map a key to an (T, 3) array of positions. Keys of estimated
trajectories are integer track ids (they select the colour); truth keys are obstacle names.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from matplotlib.patches import FancyArrowPatch

from .common import (
    as_points,
    get_axes,
    id_color,
    new_figure,
    predicted_path,
    status_color,
    to_unit_rgb,
    track_is_visible,
)
from .pointcloud_view import set_scene_limits


# ---------------------------------------------------------------------- data
def trajectories_from_tracks(tracks: Sequence, min_samples: int = 2) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """{track id: (times (T,), positions (T, 3))} from the histories of tracks.

    Tracks with fewer than min_samples recorded samples are left out.
    """
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for track in tracks:
        times, positions = track.trajectory()
        times = np.asarray(times, dtype=float)
        if len(times) >= min_samples:
            out[int(track.track_id)] = (times, np.asarray(positions, dtype=float).reshape(-1, 3))
    return out


def truth_trajectories(times, states, ids: Sequence[str]) -> Dict[str, np.ndarray]:
    """{obstacle id: positions (T, 3)} from a ground-truth log.

    states has shape (T, N, k) with k >= 3 (for example GroundTruth.to_array(), k = 6) and the
    obstacle order of `ids` (env.obstacles.ids).
    """
    t = np.asarray(times, dtype=float)
    arr = np.asarray(states, dtype=float)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("states must have shape (T, N, k) with k >= 3")
    if arr.shape[0] != len(t):
        raise ValueError("times and states must have the same length")
    if arr.shape[1] != len(ids):
        raise ValueError("ids must have one entry per obstacle")
    return {name: arr[:, i, :3].copy() for i, name in enumerate(ids)}


def position_errors(estimated, truth) -> np.ndarray:
    """Euclidean position error (T,) between two (T, 3) trajectories."""
    est = np.asarray(estimated, dtype=float)
    ref = np.asarray(truth, dtype=float)
    if est.shape != ref.shape or est.ndim != 2 or est.shape[1] != 3:
        raise ValueError("estimated and truth must both have shape (T, 3)")
    return np.linalg.norm(est - ref, axis=1)


def rmse(values) -> float:
    """Root mean square of the finite entries (NaN when there are none)."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite**2)))


# ----------------------------------------------------------------- top view
def plot_trajectories_xy(
    trajectories: Mapping[int, np.ndarray],
    truth: Optional[Mapping[str, np.ndarray]] = None,
    bounds=None,
    ax=None,
    show_start: bool = True,
    label_ids: bool = True,
    legend: bool = True,
):
    """Top view (X-Y) of estimated trajectories (coloured by id) and ground truth (dashed black).

    bounds = (bounds_min, bounds_max) sets the axis limits; only x and y are used.
    """
    ax = get_axes(ax)
    for track_id, positions in trajectories.items():
        pos = as_points(positions)
        if len(pos) == 0:
            continue
        color = to_unit_rgb(id_color(track_id))
        ax.plot(pos[:, 0], pos[:, 1], color=color, linewidth=1.5, label=f"track {track_id}")
        if show_start:
            ax.scatter([pos[0, 0]], [pos[0, 1]], color=color, marker="o", s=18)
        if label_ids:
            ax.text(pos[-1, 0], pos[-1, 1], f"ID{track_id}", color=color, fontsize=8)

    for name, positions in (truth or {}).items():
        pos = as_points(positions)
        if len(pos) == 0:
            continue
        ax.plot(pos[:, 0], pos[:, 1], color="k", linestyle="--", linewidth=1.0, label=f"truth {name}")

    if bounds is not None:
        lo, hi = (np.asarray(b, dtype=float) for b in bounds)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linewidth=0.3)
    if legend:
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7, loc="upper right")
    return ax


def plot_velocity_vectors_xy(
    tracks: Sequence,
    horizon: float = 1.0,
    ax=None,
    show_tentative: bool = False,
    show_lost: bool = False,
):
    """Predicted constant-velocity paths of tracks in the top view (dashed line plus arrowhead).

    Tracks that are (almost) not moving in the XY plane are skipped.
    """
    ax = get_axes(ax)
    for track in tracks:
        if not track_is_visible(track, show_tentative, show_lost):
            continue
        velocity = np.asarray(track.velocity, dtype=float)
        if np.linalg.norm(velocity[:2]) < 1e-6:
            continue
        color = to_unit_rgb(status_color(track))
        path = predicted_path(track.position, velocity, horizon, steps=10)
        ax.plot(path[:, 0], path[:, 1], color=color, linestyle="--", linewidth=1.0)
        ax.add_patch(
            FancyArrowPatch(
                posA=path[-2, :2], posB=path[-1, :2],
                arrowstyle="-|>", mutation_scale=10, color=color, linewidth=1.5,
            )
        )
    return ax


# --------------------------------------------------------------------- 3D view
def plot_trajectories_3d(
    trajectories: Mapping[int, np.ndarray],
    truth: Optional[Mapping[str, np.ndarray]] = None,
    bounds=None,
    ax=None,
    legend: bool = True,
):
    """3D plot of estimated trajectories (coloured by id) and ground truth (dashed black)."""
    ax = get_axes(ax, projection="3d")
    for track_id, positions in trajectories.items():
        pos = as_points(positions)
        if len(pos) == 0:
            continue
        ax.plot(
            pos[:, 0], pos[:, 1], pos[:, 2],
            color=to_unit_rgb(id_color(track_id)), linewidth=1.5, label=f"track {track_id}",
        )
    for name, positions in (truth or {}).items():
        pos = as_points(positions)
        if len(pos) == 0:
            continue
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color="k", linestyle="--", linewidth=1.0, label=f"truth {name}")

    if bounds is not None:
        set_scene_limits(ax, *bounds)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    if legend:
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7, loc="upper left")
    return ax


# --------------------------------------------------------------------- errors
def plot_estimate_vs_truth(estimated, truth, times=None, title: Optional[str] = None):
    """Figure comparing an estimated trajectory with its ground truth.

    Four stacked plots: x, y, z over time (truth dashed) and the position error.
    """
    est = np.asarray(estimated, dtype=float)
    ref = np.asarray(truth, dtype=float)
    errors = position_errors(est, ref)
    t = np.arange(len(est), dtype=float) if times is None else np.asarray(times, dtype=float)
    if t.shape != (len(est),):
        raise ValueError("times must have one entry per sample")

    fig = new_figure((7.0, 8.0))
    axes = fig.subplots(4, 1, sharex=True)
    for i, name in enumerate("xyz"):
        axes[i].plot(t, ref[:, i], color="k", linestyle="--", linewidth=1.0, label="truth")
        axes[i].plot(t, est[:, i], color="tab:blue", linewidth=1.2, label="estimate")
        axes[i].set_ylabel(f"{name} [m]")
        axes[i].grid(True, linewidth=0.3)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[3].plot(t, errors, color="tab:red", linewidth=1.2)
    axes[3].set_ylabel("error [m]")
    axes[3].set_xlabel("time [s]")
    axes[3].grid(True, linewidth=0.3)
    if title:
        fig.suptitle(title)
    return fig


def plot_position_error(times, errors: Mapping[str, np.ndarray], ax=None, title: str = "Position error"):
    """Position error over time for several labelled series (legend shows each RMSE).

    Series without any finite value are skipped; NaN entries leave gaps in the lines.
    """
    ax = get_axes(ax)
    t = np.asarray(times, dtype=float)
    for label, values in errors.items():
        e = np.asarray(values, dtype=float)
        if e.shape != t.shape:
            raise ValueError(f"series {label!r} must have the same length as times")
        value = rmse(e)
        if np.isnan(value):
            continue
        ax.plot(t, e, linewidth=1.2, label=f"{label} (RMSE {value:.3f} m)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position error [m]")
    ax.set_title(title, fontsize=9)
    ax.grid(True, linewidth=0.3)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7)
    return ax


def plot_model_probabilities(times, probabilities: Mapping[str, np.ndarray], ax=None):
    """Stacked IMM model probabilities over time (for example {"CV2D": ..., "CV3D": ...})."""
    names = list(probabilities)
    if not names:
        raise ValueError("probabilities must not be empty")
    t = np.asarray(times, dtype=float)
    arrays = []
    for name in names:
        values = np.asarray(probabilities[name], dtype=float)
        if values.shape != t.shape:
            raise ValueError(f"series {name!r} must have the same length as times")
        arrays.append(values)

    ax = get_axes(ax)
    ax.stackplot(t, *arrays, labels=names)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("model probability")
    ax.legend(fontsize=7, loc="upper right")
    return ax