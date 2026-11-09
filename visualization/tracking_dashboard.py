"""Tracking dashboard: camera view, top view, 3D scene and a status table in one figure.

    +-------------------------+-------------------------+
    | camera view + overlays  | top view (X-Y), trails  |
    +-------------------------+-------------------------+
    | 3D scene                | status table            |
    +-------------------------+-------------------------+

One DashboardFrame describes one time step. TrackingDashboard renders it to an RGB
array, a PNG file, or (frame sequence) a video. Ground truth is only drawn as a
reference marker; it is never used by the tracker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from matplotlib.patches import Rectangle

from sensors.rgb_camera import CameraIntrinsics

from .camera_view import (
    MARKER_CROSS,
    MARKER_DIAMOND,
    draw_measurements_on_image,
    draw_tracks_on_image,
)
from .common import (
    as_points,
    figure_to_array,
    new_figure,
    recent_trail,
    save_figure,
    status_color,
    status_name,
    to_unit_rgb,
    track_is_visible,
)
from .pointcloud_view import SENSOR_COLORS, plot_scene_3d
from .trajectory_view import plot_velocity_vectors_xy

STATUS_ORDER = ("confirmed", "tentative", "lost")


@dataclass
class DashboardFrame:
    """Everything shown for one time step.

    time: frame time [s].
    image: RGB uint8 camera image (H, W, 3) matching the dashboard intrinsics, or None.
    tracks: alive tracks (all statuses; deleted ones are ignored).
    measurements: objects with `position` (fusion.Measurement or perception.Detection).
    clouds: {sensor name: (N, 3) world points} of the current scans.
    truth: {obstacle id: (3,) world position} for reference markers, or None.
    """

    time: float
    image: Optional[np.ndarray] = None
    tracks: Sequence = ()
    measurements: Sequence = ()
    clouds: Mapping[str, np.ndarray] = field(default_factory=dict)
    truth: Optional[Mapping[str, np.ndarray]] = None


@dataclass(frozen=True)
class DashboardConfig:
    """Layout and drawing options (the rendered size is figsize * dpi pixels)."""

    figsize: Tuple[float, float] = (12.0, 8.0)
    dpi: int = 80
    trail_seconds: float = 3.0
    velocity_horizon: float = 1.0
    show_tentative: bool = False
    max_cloud_points: int = 3000
    view_elev: float = 25.0
    view_azim: float = -60.0
    title: str = "Multi-object tracking"

    def __post_init__(self) -> None:
        if self.figsize[0] <= 0.0 or self.figsize[1] <= 0.0:
            raise ValueError("figsize must be positive")
        if self.dpi < 1:
            raise ValueError("dpi must be >= 1")
        if self.trail_seconds < 0.0:
            raise ValueError("trail_seconds must be >= 0")
        if self.velocity_horizon < 0.0:
            raise ValueError("velocity_horizon must be >= 0")
        if self.max_cloud_points < 1:
            raise ValueError("max_cloud_points must be >= 1")


def _num(value, width: int, digits: int) -> str:
    """Right-aligned number, or "-" for missing / non-finite values."""
    value = float(value)
    if not np.isfinite(value):
        return "-".rjust(width)
    return f"{value:{width}.{digits}f}"


def _speed(track) -> float:
    speed = getattr(track, "speed", None)
    if speed is None:
        velocity = getattr(track, "velocity", None)
        return float(np.linalg.norm(velocity)) if velocity is not None else float("nan")
    return float(speed)


class TrackingDashboard:
    """Renders DashboardFrames.

    Args:
        intrinsics: intrinsics of the camera that produced the images.
        t_w_c: 4x4 transform T_W_C (camera frame -> world frame), from the calibration.
        bounds_min, bounds_max: arena limits in the world frame.
        sensor_positions: {sensor name: world position} shown as triangles.
        config: layout options.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        t_w_c,
        bounds_min: Sequence[float] = (-2.5, -2.5, 0.0),
        bounds_max: Sequence[float] = (2.5, 2.5, 3.0),
        sensor_positions: Optional[Mapping[str, Sequence[float]]] = None,
        config: Optional[DashboardConfig] = None,
    ) -> None:
        transform = np.asarray(t_w_c, dtype=float)
        if transform.shape != (4, 4):
            raise ValueError("t_w_c must be a 4x4 transform")
        lo = np.asarray(bounds_min, dtype=float).reshape(3)
        hi = np.asarray(bounds_max, dtype=float).reshape(3)
        if np.any(hi <= lo):
            raise ValueError("bounds_max must be greater than bounds_min on every axis")
        self.intrinsics = intrinsics
        self.t_w_c = transform
        self.bounds_min = lo
        self.bounds_max = hi
        self.sensor_positions = {
            name: np.asarray(p, dtype=float).reshape(3) for name, p in (sensor_positions or {}).items()
        }
        self.config = config or DashboardConfig()

    # ------------------------------------------------------------------- text
    def info_lines(self, frame: DashboardFrame, max_rows: int = 12) -> List[str]:
        """Text of the status panel: time, track counts and one row per track."""
        tracks = [t for t in frame.tracks if status_name(t) != "deleted"]
        counts = {name: sum(status_name(t) == name for t in tracks) for name in STATUS_ORDER}
        lines = [
            f"time: {frame.time:.2f} s",
            f"tracks: {counts['confirmed']} confirmed, {counts['tentative']} tentative, {counts['lost']} lost",
            f"measurements: {len(frame.measurements)}",
            "",
            " ID  status     model  v[m/s]  std[cm]",
        ]
        rows = sorted(tracks, key=lambda t: t.track_id)
        for track in rows[:max_rows]:
            std_cm = float(getattr(track, "position_std", float("nan"))) * 100.0
            lines.append(
                f"{int(track.track_id):>3d}  {status_name(track):<9s}  "
                f"{str(getattr(track, 'motion_model', '-')):<5s}  "
                f"{_num(_speed(track), 6, 2)}  {_num(std_cm, 7, 1)}"
            )
        if len(rows) > max_rows:
            lines.append(f"... +{len(rows) - max_rows} more")
        return lines

    # ------------------------------------------------------------------ panels
    def _draw_camera(self, ax, frame: DashboardFrame) -> None:
        cfg = self.config
        if frame.image is None:
            ax.text(0.5, 0.5, "no image", ha="center", va="center", transform=ax.transAxes)
        else:
            img = frame.image
            if len(frame.measurements) > 0:
                positions = [np.asarray(m.position, dtype=float) for m in frame.measurements]
                img = draw_measurements_on_image(
                    img, positions, self.intrinsics, self.t_w_c, color=(255, 255, 255), marker=MARKER_CROSS
                )
            if frame.truth:
                img = draw_measurements_on_image(
                    img,
                    list(frame.truth.values()),
                    self.intrinsics,
                    self.t_w_c,
                    color=(255, 255, 0),
                    marker=MARKER_DIAMOND,
                    size=12,
                    labels=list(frame.truth.keys()),
                )
            img = draw_tracks_on_image(
                img,
                frame.tracks,
                self.intrinsics,
                self.t_w_c,
                velocity_horizon=cfg.velocity_horizon,
                show_tentative=cfg.show_tentative,
            )
            ax.imshow(img)
        ax.set_axis_off()
        ax.set_title("Camera view", fontsize=9)

    def _draw_top_down(self, ax, frame: DashboardFrame) -> None:
        cfg = self.config
        lo, hi = self.bounds_min, self.bounds_max
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal", adjustable="box")
        ax.add_patch(Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1], fill=False, edgecolor="0.5"))
        ax.grid(True, linewidth=0.3)
        ax.tick_params(labelsize=7)

        for name, p in self.sensor_positions.items():
            ax.scatter([p[0]], [p[1]], marker="^", s=30, color=SENSOR_COLORS.get(name, "0.3"))

        if len(frame.measurements) > 0:
            pts = np.array([np.asarray(m.position, dtype=float) for m in frame.measurements]).reshape(-1, 3)
            ax.scatter(pts[:, 0], pts[:, 1], marker="x", s=20, color="0.4")
        if frame.truth:
            pts = as_points(list(frame.truth.values()))
            ax.scatter(pts[:, 0], pts[:, 1], facecolors="none", edgecolors="k", s=60)

        for track in frame.tracks:
            if not track_is_visible(track, cfg.show_tentative):
                continue
            color = to_unit_rgb(status_color(track))
            trail = recent_trail(track, cfg.trail_seconds)
            if len(trail) >= 2:
                ax.plot(trail[:, 0], trail[:, 1], color=color, linewidth=1.2)
            p = np.asarray(track.position, dtype=float)
            ax.scatter([p[0]], [p[1]], color=color, s=28)
            ax.text(p[0] + 0.05, p[1] + 0.05, f"ID{track.track_id}", color=color, fontsize=8)

        plot_velocity_vectors_xy(
            frame.tracks, horizon=cfg.velocity_horizon, ax=ax, show_tentative=cfg.show_tentative
        )
        ax.set_title("Top view (X-Y)", fontsize=9)

    def _draw_scene_3d(self, ax, frame: DashboardFrame) -> None:
        cfg = self.config
        plot_scene_3d(
            clouds=frame.clouds,
            tracks=frame.tracks,
            detections=frame.measurements,
            truth=frame.truth,
            bounds=(self.bounds_min, self.bounds_max),
            sensor_positions=self.sensor_positions,
            ax=ax,
            view=(cfg.view_elev, cfg.view_azim),
            max_points=cfg.max_cloud_points,
            trail_seconds=cfg.trail_seconds,
            velocity_horizon=cfg.velocity_horizon,
            show_tentative=cfg.show_tentative,
            show_legend=False,
        )
        ax.set_title("3D scene", fontsize=9)
        ax.tick_params(labelsize=6)

    def _draw_info(self, ax, frame: DashboardFrame) -> None:
        ax.set_axis_off()
        ax.text(
            0.0, 1.0, "\n".join(self.info_lines(frame)),
            va="top", ha="left", family="monospace", fontsize=8, transform=ax.transAxes,
        )

    # ----------------------------------------------------------------- output
    def figure(self, frame: DashboardFrame):
        """Build the matplotlib Figure of one frame."""
        cfg = self.config
        fig = new_figure(cfg.figsize, cfg.dpi)
        grid = fig.add_gridspec(2, 2)
        ax_camera = fig.add_subplot(grid[0, 0])
        ax_top = fig.add_subplot(grid[0, 1])
        ax_3d = fig.add_subplot(grid[1, 0], projection="3d")
        ax_info = fig.add_subplot(grid[1, 1])

        self._draw_camera(ax_camera, frame)
        self._draw_top_down(ax_top, frame)
        self._draw_scene_3d(ax_3d, frame)
        self._draw_info(ax_info, frame)

        fig.suptitle(f"{cfg.title} - t = {frame.time:.2f} s", fontsize=11)
        fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.06, wspace=0.12, hspace=0.2)
        return fig

    def render(self, frame: DashboardFrame) -> np.ndarray:
        """Render one frame to an RGB uint8 array of shape (figsize[1] * dpi, figsize[0] * dpi, 3)."""
        return figure_to_array(self.figure(frame))

    def save_frame(self, frame: DashboardFrame, path: Union[str, Path]) -> Path:
        """Save one frame as an image file (the format follows the extension)."""
        return save_figure(self.figure(frame), path)

    def save_frames(
        self, frames: Iterable[DashboardFrame], directory: Union[str, Path], prefix: str = "frame"
    ) -> List[Path]:
        """Save a sequence of frames as PNG files <prefix>_0000.png, <prefix>_0001.png, ..."""
        folder = Path(directory)
        return [self.save_frame(frame, folder / f"{prefix}_{i:04d}.png") for i, frame in enumerate(frames)]

    def write_video(
        self,
        path: Union[str, Path],
        frames: Iterable[DashboardFrame],
        fps: float = 10.0,
        fourcc: str = "mp4v",
    ) -> Path:
        """Write the frames to a video file with OpenCV.

        "mp4v" works for .mp4 on most installations; if the writer cannot be opened, try
        fourcc="MJPG" with a .avi file.
        """
        if fps <= 0.0:
            raise ValueError("fps must be positive")
        if len(fourcc) != 4:
            raise ValueError("fourcc must have 4 characters")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        writer = None
        count = 0
        try:
            for frame in frames:
                rgb = self.render(frame)
                if writer is None:
                    height, width = rgb.shape[:2]
                    writer = cv2.VideoWriter(
                        str(out), cv2.VideoWriter_fourcc(*fourcc), float(fps), (width, height)
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"could not open a video writer for {out} with fourcc {fourcc!r}"
                        )
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                count += 1
        finally:
            if writer is not None:
                writer.release()
        if count == 0:
            raise ValueError("no frames to write")
        return out