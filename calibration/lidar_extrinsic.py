"""LiDAR-to-camera extrinsic calibration from checkerboard planes.

Each board pose gives one plane in the camera frame (from the image, PnP) and one
plane in the LiDAR frame (fitted to the LiDAR points on the board). With planes
n . x + d = 0 expressed in both frames and p_c = R p_r + t:
    n_c = R n_r                 (normals rotate)
    n_c . t = d_r - d_c         (offsets give the translation)
The rotation is solved in closed form (Kabsch / Wahba) and the translation by
linear least squares; the result is then refined by minimizing the point-to-plane
distances of all LiDAR points (robust Huber loss).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from sensors.lidar import Lidar
from sensors.rgb_camera import CameraIntrinsics, RGBCamera

from .camera_intrinsic import observe_board
from .coordinate_frames import (
    Plane,
    fit_plane,
    make_transform,
    split_transform,
    transform_error,
)
from .target_scene import CalibrationScene

Pose = tuple


@dataclass
class PlaneCorrespondence:
    """One board pose seen by both sensors."""

    camera_plane: Plane  # board plane in the camera frame
    lidar_points: np.ndarray  # (N, 3) LiDAR points on the board, LiDAR frame

    def __post_init__(self) -> None:
        pts = np.asarray(self.lidar_points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("lidar_points must have shape (N, 3)")
        self.lidar_points = pts


def estimate_extrinsic_closed_form(
    camera_planes: Sequence[Plane], lidar_planes: Sequence[Plane]
) -> np.ndarray:
    """Closed-form T_C_R (LiDAR -> camera) from matching planes (both facing the sensors)."""
    if len(camera_planes) != len(lidar_planes):
        raise ValueError("camera_planes and lidar_planes must have the same length")
    if len(camera_planes) < 3:
        raise ValueError("need at least 3 plane correspondences")

    n_c = np.array([p.normal for p in camera_planes])
    n_r = np.array([p.normal for p in lidar_planes])
    singular = np.linalg.svd(n_c, compute_uv=False)
    if singular[-1] < 0.05 * singular[0]:
        raise ValueError(
            "plane normals are (nearly) coplanar; use board poses with more varied orientations"
        )

    # Rotation: minimize sum |n_c - R n_r|^2 (Kabsch).
    h = n_r.T @ n_c
    u, _, vt = np.linalg.svd(h)
    correction = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    rotation = vt.T @ correction @ u.T

    # Translation: n_c . t = d_r - d_c for every plane.
    d_c = np.array([p.d for p in camera_planes])
    d_r = np.array([p.d for p in lidar_planes])
    translation = np.linalg.lstsq(n_c, d_r - d_c, rcond=None)[0]
    return make_transform(rotation, translation)


def point_to_plane_residuals(
    transform_c_r, camera_planes: Sequence[Plane], lidar_points: Sequence[np.ndarray]
) -> np.ndarray:
    """Signed distances [m] of all LiDAR points (mapped to the camera frame) to their board planes."""
    rot, trans = split_transform(transform_c_r)
    chunks = []
    for plane, points in zip(camera_planes, lidar_points):
        pts_c = np.asarray(points, dtype=float) @ rot.T + trans
        chunks.append(plane.signed_distance(pts_c))
    return np.concatenate(chunks)


def refine_point_to_plane(
    transform_init,
    camera_planes: Sequence[Plane],
    lidar_points: Sequence[np.ndarray],
    f_scale: float = 0.01,
):
    """Refine T_C_R by robust point-to-plane least squares. Returns (T_C_R, rms [m])."""
    pts = [np.asarray(p, dtype=float) for p in lidar_points]
    all_points = np.vstack(pts)
    normals = np.vstack([np.repeat(pl.normal[None, :], len(p), axis=0) for pl, p in zip(camera_planes, pts)])
    offsets = np.concatenate([np.full(len(p), pl.d) for pl, p in zip(camera_planes, pts)])

    rot0, trans0 = split_transform(transform_init)
    x0 = np.concatenate([Rotation.from_matrix(rot0).as_rotvec(), trans0])

    def residual(x: np.ndarray) -> np.ndarray:
        rot = Rotation.from_rotvec(x[:3]).as_matrix()
        pts_c = all_points @ rot.T + x[3:]
        return np.einsum("ij,ij->i", normals, pts_c) + offsets

    solution = least_squares(residual, x0, loss="huber", f_scale=f_scale, method="trf")
    rot = Rotation.from_rotvec(solution.x[:3]).as_matrix()
    transform = make_transform(rot, solution.x[3:])
    rms = float(np.sqrt(np.mean(residual(solution.x) ** 2)))
    return transform, rms


@dataclass
class LidarExtrinsicResult:
    """Output of the LiDAR-to-camera calibration."""

    transform: np.ndarray  # T_C_R (4, 4), maps LiDAR points into the camera frame
    closed_form_transform: np.ndarray  # estimate before refinement
    rms: float  # point-to-plane RMS after calibration [m]
    num_views: int
    num_points: int

    @property
    def rotation(self) -> np.ndarray:
        return self.transform[:3, :3].copy()

    @property
    def translation(self) -> np.ndarray:
        return self.transform[:3, 3].copy()


def calibrate_lidar_extrinsic(
    correspondences: Sequence[PlaneCorrespondence], refine: bool = True
) -> LidarExtrinsicResult:
    """Estimate T_C_R from board observations seen by both camera and LiDAR."""
    if len(correspondences) < 3:
        raise ValueError("need at least 3 board poses")
    camera_planes = [c.camera_plane.facing_origin() for c in correspondences]
    lidar_points = [c.lidar_points for c in correspondences]
    lidar_planes = [fit_plane(p, robust_iterations=2) for p in lidar_points]

    closed_form = estimate_extrinsic_closed_form(camera_planes, lidar_planes)
    if refine:
        transform, rms = refine_point_to_plane(closed_form, camera_planes, lidar_points)
    else:
        transform = closed_form
        residuals = point_to_plane_residuals(transform, camera_planes, lidar_points)
        rms = float(np.sqrt(np.mean(residuals**2)))
    return LidarExtrinsicResult(
        transform=transform,
        closed_form_transform=closed_form,
        rms=rms,
        num_views=len(correspondences),
        num_points=int(sum(len(p) for p in lidar_points)),
    )


# -------------------------------------------------------------- MuJoCo helpers
def collect_lidar_views(
    scene: CalibrationScene,
    camera: RGBCamera,
    lidar: Lidar,
    poses: Sequence[Pose],
    intrinsics: Optional[CameraIntrinsics] = None,
    min_points: int = 100,
) -> List[PlaneCorrespondence]:
    """Observe the board with camera and LiDAR for each pose (skips failed poses).

    The camera plane comes from the image (PnP). The LiDAR points on the board are
    selected with the simulation geom ids, which stands in for the operator
    selecting the target region in the point cloud.
    """
    intrinsics = intrinsics or camera.intrinsics
    views: List[PlaneCorrespondence] = []
    for pose in poses:
        observation = observe_board(scene, camera, pose, intrinsics)
        if observation is None:
            continue
        scan = lidar.scan()
        on_board = np.isin(scan.geom_ids[scan.valid], scene.board_geom_ids)
        points = scan.points()[on_board]
        if len(points) < min_points:
            continue
        views.append(PlaneCorrespondence(observation.plane, points))
    return views


@dataclass(frozen=True)
class ExtrinsicError:
    rotation_deg: float
    translation_m: float


def evaluate_extrinsic(estimated, truth) -> ExtrinsicError:
    """Rotation and translation error of an estimated transform against the true one."""
    rotation_deg, translation_m = transform_error(estimated, truth)
    return ExtrinsicError(rotation_deg, translation_m)