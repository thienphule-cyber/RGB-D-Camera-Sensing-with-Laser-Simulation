"""3D views (matplotlib): point clouds, detections, boxes, tracks and the whole scene.

Every function draws into a given Axes3D (created when omitted) and returns it or the
artist it created. Point clouds are subsampled before plotting to keep rendering fast.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .common import (
    as_points,
    get_axes,
    recent_trail,
    status_color,
    status_name,
    to_unit_colors,
    to_unit_rgb,
    track_is_visible,
)

SENSOR_COLORS = {"rgbd": "tab:blue", "laser": "tab:red", "lidar": "tab:green"}


# ----------------------------------------------------------------------- clouds
def prepare_cloud(points, colors=None, max_points: Optional[int] = 5000, rng=None):
    """Clean a cloud for plotting. Returns (points (M, 3), colours (M, 3) in [0, 1] or None).

    Non-finite points are dropped, colours (uint8 0-255 or float 0-1) are kept aligned with
    the points, and clouds larger than max_points are randomly subsampled (reproducible:
    the default random generator has a fixed seed).
    """
    pts = as_points(points)
    cols = None
    if colors is not None:
        cols = np.asarray(colors)
        if cols.shape != pts.shape:
            raise ValueError("colors must have shape (N, 3), one row per point")
        cols = to_unit_colors(cols)

    keep = np.isfinite(pts).all(axis=1)
    pts = pts[keep]
    if cols is not None:
        cols = cols[keep]

    if max_points is not None and len(pts) > max_points:
        generator = rng if rng is not None else np.random.default_rng(0)
        index = np.sort(generator.choice(len(pts), size=int(max_points), replace=False))
        pts = pts[index]
        if cols is not None:
            cols = cols[index]
    return pts, cols


def plot_point_cloud(
    ax,
    points,
    colors=None,
    color="0.4",
    size: float = 2.0,
    max_points: Optional[int] = 5000,
    label: Optional[str] = None,
    alpha: float = 0.8,
    rng=None,
):
    """Scatter a point cloud. Returns the scatter artist, or None for an empty cloud."""
    pts, cols = prepare_cloud(points, colors, max_points, rng)
    if len(pts) == 0:
        return None
    kwargs = dict(s=size, alpha=alpha, depthshade=False, label=label)
    if cols is not None:
        kwargs["c"] = cols
    else:
        kwargs["color"] = color
    return ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], **kwargs)


# ------------------------------------------------------------------------ boxes
def _box_segments(lo, hi):
    """The 12 edges of an axis-aligned box as (corner a, corner b) pairs."""
    lo = np.asarray(lo, dtype=float).reshape(3)
    hi = np.asarray(hi, dtype=float).reshape(3)
    segments = []
    for i in range(8):
        for k in range(3):
            if not (i >> k) & 1:
                j = i | (1 << k)
                a = np.where(np.array([(i >> m) & 1 for m in range(3)], dtype=bool), hi, lo)
                b = np.where(np.array([(j >> m) & 1 for m in range(3)], dtype=bool), hi, lo)
                segments.append((a, b))
    return segments


def draw_box_3d(ax, bbox_min, bbox_max, color="tab:green", linewidth: float = 1.0, linestyle: str = "-"):
    """Draw the 12 edges of an axis-aligned box (also valid for flat boxes)."""
    lo = np.asarray(bbox_min, dtype=float).reshape(3)
    hi = np.asarray(bbox_max, dtype=float).reshape(3)
    if np.any(hi < lo):
        raise ValueError("bbox_max must be >= bbox_min on every axis")
    for a, b in _box_segments(lo, hi):
        ax.plot(
            [a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
            color=color, linewidth=linewidth, linestyle=linestyle,
        )
    return ax


def draw_arena_box(ax, bounds_min, bounds_max, color="0.6", linewidth: float = 0.8):
    """Draw the arena as a wireframe box."""
    return draw_box_3d(ax, bounds_min, bounds_max, color=color, linewidth=linewidth)


def set_scene_limits(ax, bounds_min, bounds_max):
    """Set the axis limits to the arena and make the box proportional to its size."""
    lo = np.asarray(bounds_min, dtype=float).reshape(3)
    hi = np.asarray(bounds_max, dtype=float).reshape(3)
    extent = hi - lo
    if np.any(extent <= 0.0):
        raise ValueError("bounds_max must be greater than bounds_min on every axis")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(float(v) for v in extent))
    return ax


# -------------------------------------------------------------------- detections
def plot_detections_3d(
    ax,
    detections: Sequence,
    color="k",
    marker: str = "x",
    size: float = 30.0,
    show_bbox: bool = False,
    label: Optional[str] = None,
):
    """Mark detections / measurements (objects with `position`; optionally features.bbox_min/max).

    Returns the scatter artist, or None when there are no detections.
    """
    if len(detections) == 0:
        return None
    positions = np.array([np.asarray(d.position, dtype=float) for d in detections]).reshape(-1, 3)
    artist = ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        color=color, marker=marker, s=size, depthshade=False, label=label,
    )
    if show_bbox:
        for d in detections:
            features = getattr(d, "features", None)
            if features is not None:
                draw_box_3d(ax, features.bbox_min, features.bbox_max, color=color, linewidth=0.8)
    return artist


# ------------------------------------------------------------------------ tracks
def plot_tracks_3d(
    ax,
    tracks: Sequence,
    velocity_horizon: float = 0.5,
    trail_seconds: Optional[float] = 3.0,
    show_tentative: bool = False,
    show_lost: bool = True,
    label_ids: bool = True,
    size: float = 40.0,
):
    """Draw tracks as points with a recent trail, a velocity segment (confirmed) and an id label."""
    for track in tracks:
        if not track_is_visible(track, show_tentative, show_lost):
            continue
        color = to_unit_rgb(status_color(track))
        p = np.asarray(track.position, dtype=float)
        ax.scatter([p[0]], [p[1]], [p[2]], color=color, s=size, marker="o", depthshade=False)

        trail = recent_trail(track, trail_seconds)
        if len(trail) >= 2:
            ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], color=color, linewidth=1.2)

        velocity = getattr(track, "velocity", None)
        if status_name(track) == "confirmed" and velocity is not None and velocity_horizon > 0.0:
            tip = p + np.asarray(velocity, dtype=float) * velocity_horizon
            if np.linalg.norm(tip - p) > 1e-6:
                ax.plot([p[0], tip[0]], [p[1], tip[1]], [p[2], tip[2]], color=color, linewidth=2.0)

        if label_ids:
            suffix = "?" if status_name(track) == "lost" else ""
            ax.text(p[0], p[1], p[2] + 0.12, f"ID{track.track_id}{suffix}", color=color, fontsize=8)
    return ax


# ------------------------------------------------------------------------- scene
def plot_scene_3d(
    clouds: Optional[Mapping[str, np.ndarray]] = None,
    tracks: Sequence = (),
    detections: Sequence = (),
    truth: Optional[Mapping[str, np.ndarray]] = None,
    bounds: Tuple[Sequence[float], Sequence[float]] = ((-2.5, -2.5, 0.0), (2.5, 2.5, 3.0)),
    sensor_positions: Optional[Mapping[str, Sequence[float]]] = None,
    ax=None,
    view: Tuple[float, float] = (25.0, -60.0),
    max_points: Optional[int] = 3000,
    trail_seconds: Optional[float] = 3.0,
    velocity_horizon: float = 0.5,
    show_tentative: bool = False,
    show_legend: bool = True,
):
    """Draw the full 3D scene: arena, sensor point clouds, measurements, ground truth, tracks.

    clouds: {sensor name: (N, 3) world points}. truth: {obstacle id: (3,) world position}.
    bounds: (bounds_min, bounds_max) of the arena. Returns the Axes3D.
    """
    ax = get_axes(ax, projection="3d")
    lo, hi = bounds
    draw_arena_box(ax, lo, hi)

    for name, points in (clouds or {}).items():
        plot_point_cloud(ax, points, color=SENSOR_COLORS.get(name, "0.4"), max_points=max_points, label=name)

    plot_detections_3d(ax, detections, color="k", marker="x", label="measurements")

    if truth:
        positions = as_points(list(truth.values()))
        ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            facecolors="none", edgecolors="k", s=70, depthshade=False, label="ground truth",
        )

    for name, position in (sensor_positions or {}).items():
        p = np.asarray(position, dtype=float).reshape(3)
        ax.scatter(
            [p[0]], [p[1]], [p[2]],
            color=SENSOR_COLORS.get(name, "0.3"), marker="^", s=40, depthshade=False,
        )
        ax.text(p[0], p[1], p[2] + 0.1, name, fontsize=7)

    plot_tracks_3d(
        ax, tracks,
        velocity_horizon=velocity_horizon,
        trail_seconds=trail_seconds,
        show_tentative=show_tentative,
    )

    set_scene_limits(ax, lo, hi)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    if show_legend:
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper left", fontsize=7)
    return ax