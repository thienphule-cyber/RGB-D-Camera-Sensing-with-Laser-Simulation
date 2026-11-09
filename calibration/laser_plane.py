"""Laser plane calibration and laser triangulation.

Calibration procedure (checkerboard at several poses):
    1. Estimate the board plane in the camera frame from the image corners (PnP).
    2. Detect the laser stripe pixels on the board.
    3. Back-project every stripe pixel to a camera ray and intersect it with the
    board plane: the result is a 3D point that lies on the laser plane.
    4. Fit the laser plane  n . x + d = 0  to the points of all board poses.

Triangulation with the calibrated plane (ray r = K^-1 [u, v, 1]^T):
    P = -d / (n . r) * r
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from sensors.laser_scanner import LaserScanner
from sensors.rgb_camera import CameraIntrinsics, RGBCamera, ground_truth_camera_pose_opencv

from .camera_intrinsic import observe_board
from .coordinate_frames import Plane, fit_plane, invert_transform, make_transform
from .target_scene import CalibrationScene

Pose = tuple


def pixels_to_rays(pixels, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Camera rays (N, 3) with z = 1 for undistorted pixels (N, 2)."""
    uv = np.asarray(pixels, dtype=float).reshape(-1, 2)
    x = (uv[:, 0] - intrinsics.cx) / intrinsics.fx
    y = (uv[:, 1] - intrinsics.cy) / intrinsics.fy
    return np.column_stack([x, y, np.ones(len(uv))])


def triangulate_laser_pixels(pixels, intrinsics: CameraIntrinsics, plane: Plane) -> np.ndarray:
    """3D points (N, 3) in the camera frame for laser stripe pixels (NaN if invalid).

    A laser plane that passes through the camera centre (d = 0) is degenerate and
    gives only NaN: the scanner needs a baseline between laser and camera.
    """
    return plane.intersect_rays(pixels_to_rays(pixels, intrinsics))


@dataclass
class LaserView:
    """Data of one board pose: the board plane (camera frame) and the stripe pixels on it."""

    board_plane: Plane
    stripe_pixels: np.ndarray  # (M, 2)


@dataclass
class LaserPlaneResult:
    """Output of the laser plane calibration."""

    plane: Plane  # in the camera frame, facing the camera origin (d >= 0)
    rms: float  # RMS distance of the calibration points to the plane [m]
    num_points: int
    num_views: int


def calibrate_laser_plane(
    views: Sequence[LaserView],
    intrinsics: CameraIntrinsics,
    robust_iterations: int = 2,
) -> LaserPlaneResult:
    """Estimate the laser plane in the camera frame from several board poses."""
    chunks: List[np.ndarray] = []
    used_views = 0
    for view in views:
        rays = pixels_to_rays(view.stripe_pixels, intrinsics)
        points = view.board_plane.intersect_rays(rays)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) > 0:
            chunks.append(points)
            used_views += 1
    if not chunks:
        raise ValueError("no usable stripe points")
    points = np.vstack(chunks)
    if len(points) < 10:
        raise ValueError("need at least 10 stripe points")

    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    if singular[1] < 1e-3 * singular[0]:
        raise ValueError(
            "stripe points are (nearly) collinear; use more varied board poses"
        )

    plane = fit_plane(points, robust_iterations=robust_iterations)
    rms = float(np.sqrt(np.mean(plane.signed_distance(points) ** 2)))
    return LaserPlaneResult(plane=plane, rms=rms, num_points=len(points), num_views=used_views)


# -------------------------------------------------------------- MuJoCo helpers
def ground_truth_laser_plane(scene: CalibrationScene, scanner: LaserScanner, yaw_deg: float = 0.0) -> Plane:
    """True laser plane in the camera frame (evaluation only)."""
    normal_w, point_w = scanner.plane_world(yaw_deg)
    plane_world = Plane(normal_w, -float(normal_w @ point_w))
    cam_pos, rot_wc = ground_truth_camera_pose_opencv(scene)
    t_c_w = invert_transform(make_transform(rot_wc, cam_pos))
    return plane_world.transformed(t_c_w).facing_origin()


def simulate_stripe_pixels(
    scene: CalibrationScene,
    scanner: LaserScanner,
    intrinsics: CameraIntrinsics,
    yaw_deg: float = 0.0,
    pixel_noise_std: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Pixels of the laser stripe on the board, as a stripe detector would report them.

    MuJoCo does not render the laser, so the laser spots are ray cast, restricted
    to the board (the stand-in for detecting the stripe inside the board region of
    the image) and projected through the camera. Returns (M, 2) pixels.
    """
    rng = rng or np.random.default_rng()
    scan = scanner.scan_line(yaw_deg)
    on_board = np.isin(scan.geom_ids[scan.valid], scene.board_geom_ids)
    points_laser = scan.points()[on_board]
    if len(points_laser) == 0:
        return np.zeros((0, 2))

    laser_pos, laser_rot = scanner.ground_truth_pose()
    points_world = points_laser @ laser_rot.T + laser_pos
    cam_pos, rot_wc = ground_truth_camera_pose_opencv(scene)
    points_cam = (points_world - cam_pos) @ rot_wc  # R^T (p - t) in row form
    pixels = intrinsics.project(points_cam)
    if pixel_noise_std > 0.0:
        pixels = pixels + rng.normal(0.0, pixel_noise_std, pixels.shape)
    return pixels


def collect_laser_views(
    scene: CalibrationScene,
    camera: RGBCamera,
    scanner: LaserScanner,
    poses: Sequence[Pose],
    intrinsics: Optional[CameraIntrinsics] = None,
    yaw_deg: float = 0.0,
    pixel_noise_std: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    min_pixels: int = 10,
) -> List[LaserView]:
    """Observe the board and the laser stripe for each pose (skips failed poses)."""
    intrinsics = intrinsics or camera.intrinsics
    views: List[LaserView] = []
    for pose in poses:
        observation = observe_board(scene, camera, pose, intrinsics)
        if observation is None:
            continue
        pixels = simulate_stripe_pixels(scene, scanner, intrinsics, yaw_deg, pixel_noise_std, rng)
        if len(pixels) < min_pixels:
            continue
        views.append(LaserView(observation.plane, pixels))
    return views


# ------------------------------------------------------------------ evaluation
@dataclass(frozen=True)
class LaserPlaneError:
    normal_angle_deg: float  # angle between estimated and true plane normals
    distance_error: float  # |d_est - d_true| [m]


def evaluate_laser_plane(estimated: Plane, truth: Plane) -> LaserPlaneError:
    """Compare two laser planes expressed in the same frame."""
    est = estimated.facing_origin()
    ref = truth.facing_origin()
    return LaserPlaneError(
        normal_angle_deg=est.angle_to(ref),
        distance_error=abs(est.d - ref.d),
    )