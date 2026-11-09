"""Shared helpers of the visualization package: colours, figures and small geometry utilities.

Figures are created with matplotlib.figure.Figure (never pyplot), so the package needs no
GUI backend, keeps no global figure state and works on headless machines.

A "track" is any object with track_id, status (a TrackStatus or its string value) and
position (3,); velocity (3,), speed, position_std, motion_model and trajectory() are
used when available. tracking.Track provides all of them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

Color = Tuple[int, int, int]

# Distinct colours (RGB, 0-255), cycled by track id.
PALETTE: Tuple[Color, ...] = (
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
)
TENTATIVE_COLOR: Color = (150, 150, 150)


# --------------------------------------------------------------------- colours
def id_color(track_id: int) -> Color:
    """Stable RGB colour (0-255) of an id; ids 1, 2, ... walk through the palette and wrap."""
    return PALETTE[(int(track_id) - 1) % len(PALETTE)]


def dim_color(color, factor: float = 0.5) -> Color:
    """The colour scaled toward black by `factor` (0-1)."""
    return tuple(int(round(c * factor)) for c in color)  # type: ignore[return-value]


def to_unit_rgb(color) -> Tuple[float, float, float]:
    """Convert an RGB colour in 0-255 to matplotlib's 0-1 floats."""
    return tuple(float(c) / 255.0 for c in color)  # type: ignore[return-value]


def to_unit_colors(colors) -> np.ndarray:
    """(N, 3) colours as floats in [0, 1]; integer arrays are taken as 0-255."""
    arr = np.asarray(colors)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float64) / 255.0
    return np.clip(arr.astype(np.float64), 0.0, 1.0)


def status_name(track) -> str:
    """Status of a track as a lower-case string ("unknown" when the track has none)."""
    status = getattr(track, "status", None)
    if status is None:
        return "unknown"
    return str(getattr(status, "value", status))


def track_is_visible(track, show_tentative: bool = False, show_lost: bool = True) -> bool:
    """Whether a track should be drawn: confirmed always, tentative and lost on request.

    Deleted tracks and tracks without a status are never drawn.
    """
    name = status_name(track)
    if name == "confirmed":
        return True
    if name == "tentative":
        return bool(show_tentative)
    if name == "lost":
        return bool(show_lost)
    return False


def status_color(track) -> Color:
    """Draw colour of a track: its id colour, grey while tentative, dimmed while lost."""
    name = status_name(track)
    base = id_color(track.track_id)
    if name == "tentative":
        return TENTATIVE_COLOR
    if name == "lost":
        return dim_color(base, 0.55)
    return base


# -------------------------------------------------------------------- geometry
def as_points(points) -> np.ndarray:
    """Convert to a float (N, 3) array (an empty input gives shape (0, 3))."""
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 3))
    return arr.reshape(-1, 3)


def predicted_path(position, velocity, horizon: float = 1.0, steps: int = 10) -> np.ndarray:
    """Constant-velocity path (steps + 1, 3) from `position` over `horizon` seconds."""
    if horizon < 0.0:
        raise ValueError("horizon must be >= 0")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    p = np.asarray(position, dtype=float).reshape(3)
    v = np.asarray(velocity, dtype=float).reshape(3)
    t = np.linspace(0.0, horizon, steps + 1)
    return p[None, :] + t[:, None] * v[None, :]


def recent_trail(track, seconds: Optional[float]) -> np.ndarray:
    """Positions (M, 3) of the track samples in the last `seconds` of its history.

    seconds=None returns the whole history. Objects without trajectory() give an empty array.
    """
    getter = getattr(track, "trajectory", None)
    if getter is None:
        return np.zeros((0, 3))
    times, positions = getter()
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    if len(times) == 0:
        return np.zeros((0, 3))
    if seconds is None:
        return positions
    return positions[times >= times[-1] - float(seconds) - 1e-9]


# --------------------------------------------------------------------- figures
def new_figure(figsize: Tuple[float, float] = (8.0, 6.0), dpi: int = 100) -> Figure:
    """A standalone Figure (not registered with pyplot)."""
    return Figure(figsize=figsize, dpi=dpi)


def get_axes(ax=None, projection: Optional[str] = None, figsize: Tuple[float, float] = (6.0, 5.0)):
    """Return `ax`, or create a new figure with one axes (projection "3d" for 3D plots)."""
    if ax is not None:
        return ax
    return new_figure(figsize).add_subplot(111, projection=projection)


def figure_to_array(fig: Figure) -> np.ndarray:
    """Render a figure to an RGB uint8 array (height, width, 3)."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return rgba[..., :3].copy()


def save_figure(fig: Figure, path: Union[str, Path]) -> Path:
    """Save a figure; the format follows the file extension. Parent folders are created."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    return out