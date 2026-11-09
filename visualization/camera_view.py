"""Camera-view overlays: depth colouring and projected measurements / tracks on the RGB image.

Images are uint8 RGB arrays (H, W, 3). Every drawing function returns a modified copy and
never changes its input. World points are projected with the (calibrated) transform T_W_C
and the camera intrinsics (OpenCV optical frame: x right, y down, z forward).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from calibration.coordinate_frames import apply_transform, invert_transform
from sensors.rgb_camera import CameraIntrinsics

from .common import Color, as_points, status_color, status_name, track_is_visible

FONT = cv2.FONT_HERSHEY_SIMPLEX
MARKER_CROSS = cv2.MARKER_CROSS
MARKER_DIAMOND = cv2.MARKER_DIAMOND


# --------------------------------------------------------------------- helpers
def _check_image(image, intrinsics: Optional[CameraIntrinsics] = None) -> np.ndarray:
    """Validate an RGB uint8 image and return a contiguous copy."""
    img = np.asarray(image)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")
    if img.dtype != np.uint8:
        raise ValueError("image must be uint8")
    if intrinsics is not None and img.shape[:2] != (intrinsics.height, intrinsics.width):
        raise ValueError("image size does not match the camera intrinsics")
    return np.ascontiguousarray(img).copy()


def _pt(pixel) -> Tuple[int, int]:
    """Integer pixel for OpenCV drawing (clipped so far-away points cannot overflow)."""
    u = int(np.clip(round(float(pixel[0])), -10000, 10000))
    v = int(np.clip(round(float(pixel[1])), -10000, 10000))
    return u, v


def _put_label(image: np.ndarray, text: str, position, color, scale: float = 0.4) -> None:
    """Text with a dark outline so it stays readable on any background (in place)."""
    pos = (int(position[0]), int(position[1]))
    cv2.putText(image, text, pos, FONT, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, text, pos, FONT, scale, tuple(int(c) for c in color), 1, cv2.LINE_AA)


# ------------------------------------------------------------------ projection
def project_world_points(points_world, intrinsics: CameraIntrinsics, t_w_c, min_depth: float = 0.05):
    """Project world points into the image.

    Returns (pixels (N, 2) as [u, v], depth (N,) along the optical axis [m],
    inside (N,) bool: in front of the camera and within the image bounds).
    """
    pts = as_points(points_world)
    pts_cam = apply_transform(invert_transform(t_w_c), pts)
    depth = pts_cam[:, 2].copy()
    pixels = intrinsics.project(pts_cam)
    with np.errstate(invalid="ignore"):
        inside = (
            (depth > min_depth)
            & np.isfinite(pixels).all(axis=1)
            & (pixels[:, 0] >= 0.0)
            & (pixels[:, 0] <= intrinsics.width - 1)
            & (pixels[:, 1] >= 0.0)
            & (pixels[:, 1] <= intrinsics.height - 1)
        )
    return pixels, depth, inside


# ----------------------------------------------------------------------- depth
def colorize_depth(depth, min_depth: float = 0.1, max_depth: float = 8.0) -> np.ndarray:
    """Depth image (H, W) [m] -> RGB uint8 (H, W, 3). Near is warm, far is cool.

    NaN pixels and values outside [min_depth, max_depth] are black.
    """
    if not 0.0 <= min_depth < max_depth:
        raise ValueError("need 0 <= min_depth < max_depth")
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError("depth must be a 2D array")
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(d) & (d >= min_depth) & (d <= max_depth)

    safe = np.where(valid, d, min_depth)
    scaled = np.clip((safe - min_depth) / (max_depth - min_depth), 0.0, 1.0)
    gray = np.round((1.0 - scaled) * 255.0).astype(np.uint8)
    cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    bgr = cv2.applyColorMap(gray, cmap)
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    rgb[~valid] = 0
    return rgb


# ----------------------------------------------------------------------- text
def draw_text_lines(
    image,
    lines: Sequence[str],
    origin: Tuple[int, int] = (6, 16),
    scale: float = 0.45,
    color: Color = (255, 255, 255),
    line_height: int = 16,
) -> np.ndarray:
    """Draw several lines of text starting at `origin` (a copy is returned)."""
    out = _check_image(image)
    x, y = origin
    for i, line in enumerate(lines):
        _put_label(out, str(line), (x, y + i * line_height), color, scale)
    return out


# ------------------------------------------------------------------- overlays
def draw_measurements_on_image(
    image,
    positions,
    intrinsics: CameraIntrinsics,
    t_w_c,
    color: Color = (255, 255, 255),
    marker: int = MARKER_CROSS,
    size: int = 9,
    labels: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Draw a marker at the projection of every world position that is inside the image."""
    out = _check_image(image, intrinsics)
    pts = as_points(positions)
    if labels is not None and len(labels) != len(pts):
        raise ValueError("labels must have one entry per position")
    pixels, _, inside = project_world_points(pts, intrinsics, t_w_c)
    col = tuple(int(c) for c in color)
    for i in np.flatnonzero(inside):
        center = _pt(pixels[i])
        cv2.drawMarker(out, center, col, marker, size, 1)
        if labels is not None:
            _put_label(out, str(labels[i]), (center[0] + 6, center[1] - 6), col)
    return out


def draw_tracks_on_image(
    image,
    tracks: Sequence,
    intrinsics: CameraIntrinsics,
    t_w_c,
    velocity_horizon: float = 0.5,
    radius_m: float = 0.15,
    show_tentative: bool = False,
    show_lost: bool = True,
) -> np.ndarray:
    """Draw every visible track: a circle, its id and (confirmed tracks) a velocity arrow.

    Confirmed tracks get a thick circle in their id colour, tentative tracks a thin grey one,
    lost tracks a dimmed one with "?" after the id. Tracks outside the image or behind the
    camera are skipped. The arrow shows the position after `velocity_horizon` seconds.
    """
    out = _check_image(image, intrinsics)
    for track in tracks:
        if not track_is_visible(track, show_tentative, show_lost):
            continue
        position = np.asarray(track.position, dtype=float)
        pixels, depth, inside = project_world_points(position, intrinsics, t_w_c)
        if not inside[0]:
            continue

        name = status_name(track)
        color = status_color(track)
        center = _pt(pixels[0])
        radius = int(np.clip(radius_m * intrinsics.fx / depth[0], 6, 80))
        cv2.circle(out, center, radius, color, 2 if name == "confirmed" else 1)
        text = f"ID{track.track_id}" + ("?" if name == "lost" else "")
        _put_label(out, text, (center[0] + radius + 4, center[1] - radius), color)

        velocity = getattr(track, "velocity", None)
        if name == "confirmed" and velocity is not None and velocity_horizon > 0.0:
            step = np.asarray(velocity, dtype=float) * velocity_horizon
            if np.linalg.norm(step) > 1e-6:
                end_pixels, end_depth, _ = project_world_points(position + step, intrinsics, t_w_c)
                if end_depth[0] > 0.05 and np.isfinite(end_pixels[0]).all():
                    cv2.arrowedLine(out, center, _pt(end_pixels[0]), color, 1, cv2.LINE_8, 0, 0.25)
    return out