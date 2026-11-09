"""Camera intrinsic calibration (Zhang's method via OpenCV) and checkerboard observation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from sensors.rgb_camera import CameraIntrinsics, RGBCamera

from .coordinate_frames import Plane, board_plane
from .target_scene import CalibrationScene, checkerboard_object_points

Pose = Tuple[np.ndarray, np.ndarray]


# ------------------------------------------------------------- synthetic views
def project_board_corners(
    intrinsics: CameraIntrinsics,
    distortion: Optional[Sequence[float]],
    pattern_size: Tuple[int, int],
    square_size: float,
    poses: Sequence[Pose],
    noise_std: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """Synthetic corner observations: project the board corners for each pose.

    distortion = (k1, k2, p1, p2, k3) or None. Views with a corner outside the
    image are dropped. Returns a list of (N, 2) pixel arrays.
    """
    rng = rng or np.random.default_rng()
    objp = checkerboard_object_points(pattern_size, square_size)
    dist = np.zeros(5) if distortion is None else np.asarray(distortion, dtype=float).reshape(-1)
    views: List[np.ndarray] = []
    for rotation, translation in poses:
        rvec, _ = cv2.Rodrigues(np.asarray(rotation, dtype=float))
        pts, _ = cv2.projectPoints(
            objp, rvec, np.asarray(translation, dtype=float).reshape(3, 1), intrinsics.matrix, dist
        )
        pts = pts.reshape(-1, 2)
        if noise_std > 0.0:
            pts = pts + rng.normal(0.0, noise_std, pts.shape)
        inside = (
            (pts[:, 0] >= 0.0) & (pts[:, 0] <= intrinsics.width - 1)
            & (pts[:, 1] >= 0.0) & (pts[:, 1] <= intrinsics.height - 1)
        )
        if inside.all():
            views.append(pts)
    return views


# ------------------------------------------------------------ corner detection
def detect_corners(image: np.ndarray, pattern_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Detect checkerboard inner corners. Returns (N, 2) float64 pixels or None."""
    img = np.asarray(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    pattern = (int(pattern_size[0]), int(pattern_size[1]))

    corners = None
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found, found_corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
        if found:
            corners = found_corners
    if corners is None:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, found_corners = cv2.findChessboardCorners(gray, pattern, flags)
        if not found:
            return None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
        corners = cv2.cornerSubPix(gray, found_corners, (5, 5), (-1, -1), criteria)
    return np.asarray(corners, dtype=np.float64).reshape(-1, 2)


def capture_board_corners(
    scene: CalibrationScene, camera: RGBCamera, pose: Pose
) -> Optional[np.ndarray]:
    """Place the board at `pose`, render an image and detect its corners."""
    scene.set_board_pose(*pose)
    return detect_corners(camera.capture(), scene.pattern_size)


def collect_board_corners(
    scene: CalibrationScene, camera: RGBCamera, poses: Sequence[Pose]
) -> List[np.ndarray]:
    """Corner detections for every pose where the board was found."""
    views = (capture_board_corners(scene, camera, pose) for pose in poses)
    return [corners for corners in views if corners is not None]


# -------------------------------------------------------- intrinsic calibration
@dataclass
class IntrinsicCalibrationResult:
    """Output of Zhang calibration."""

    intrinsics: CameraIntrinsics
    distortion: np.ndarray  # (5,) = [k1, k2, p1, p2, k3]
    rms: float  # overall RMS reprojection error [px]
    per_view_rms: np.ndarray  # (n_views,) [px]
    poses: List[Pose]  # estimated board poses (R_cb, t_cb), one per view

    @property
    def camera_matrix(self) -> np.ndarray:
        return self.intrinsics.matrix


def calibrate_camera(
    corner_views: Sequence[np.ndarray],
    pattern_size: Tuple[int, int],
    square_size: float,
    image_size: Tuple[int, int],
    estimate_distortion: bool = True,
) -> IntrinsicCalibrationResult:
    """Zhang calibration from checkerboard corner detections.

    image_size = (width, height). With estimate_distortion=False the distortion
    coefficients are fixed to zero (the MuJoCo camera has no lens distortion).
    """
    if len(corner_views) < 3:
        raise ValueError("need at least 3 views for calibration")
    width, height = int(image_size[0]), int(image_size[1])
    objp = checkerboard_object_points(pattern_size, square_size)
    n_points = len(objp)
    img_points = []
    for corners in corner_views:
        arr = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        if len(arr) != n_points:
            raise ValueError(f"each view needs {n_points} corners, got {len(arr)}")
        img_points.append(arr.astype(np.float32).reshape(-1, 1, 2))
    obj_points = [objp.astype(np.float32)] * len(img_points)

    if estimate_distortion:
        flags = 0
    else:
        flags = (
            cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
        )
    criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-12)
    rms, mat, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (width, height), None, None, flags=flags, criteria=criteria
    )

    per_view = []
    poses: List[Pose] = []
    for img, rvec, tvec in zip(img_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, mat, dist)
        diff = projected.reshape(-1, 2) - img.reshape(-1, 2).astype(np.float64)
        per_view.append(float(np.sqrt(np.mean(np.sum(diff**2, axis=1)))))
        rot, _ = cv2.Rodrigues(rvec)
        poses.append((rot, tvec.reshape(3).copy()))

    intrinsics = CameraIntrinsics(
        fx=float(mat[0, 0]), fy=float(mat[1, 1]),
        cx=float(mat[0, 2]), cy=float(mat[1, 2]),
        width=width, height=height,
    )
    return IntrinsicCalibrationResult(
        intrinsics=intrinsics,
        distortion=np.asarray(dist, dtype=float).reshape(-1)[:5].copy(),
        rms=float(rms),
        per_view_rms=np.asarray(per_view),
        poses=poses,
    )


@dataclass(frozen=True)
class IntrinsicError:
    """Calibration errors against ground truth."""

    fx_rel: float  # relative focal length error (x)
    fy_rel: float  # relative focal length error (y)
    principal_point_px: float  # distance between estimated and true principal point [px]
    distortion_max_abs: float  # max |estimated - true| over the 5 distortion coefficients
    reprojection_rms: float  # RMS reprojection error of the calibration [px]


def evaluate_intrinsics(
    result: IntrinsicCalibrationResult,
    truth: CameraIntrinsics,
    truth_distortion: Optional[Sequence[float]] = None,
) -> IntrinsicError:
    """Compare a calibration result with the true intrinsics (and distortion)."""
    est = result.intrinsics
    true_dist = np.zeros(5) if truth_distortion is None else np.asarray(truth_distortion, float)
    return IntrinsicError(
        fx_rel=abs(est.fx - truth.fx) / truth.fx,
        fy_rel=abs(est.fy - truth.fy) / truth.fy,
        principal_point_px=float(np.hypot(est.cx - truth.cx, est.cy - truth.cy)),
        distortion_max_abs=float(np.max(np.abs(result.distortion - true_dist.reshape(-1)[:5]))),
        reprojection_rms=result.rms,
    )


# ------------------------------------------------------------- board pose (PnP)
@dataclass
class BoardObservation:
    """Checkerboard pose estimated from one image."""

    corners: np.ndarray  # (N, 2) detected corners [px]
    rotation: np.ndarray  # R_cb, board -> camera (OpenCV frame)
    translation: np.ndarray  # t_cb
    reprojection_rms: float  # [px]

    @property
    def plane(self) -> Plane:
        """Board plane in the camera frame, normal facing the camera."""
        return board_plane(self.rotation, self.translation)


def estimate_board_pose(
    corners: np.ndarray,
    intrinsics: CameraIntrinsics,
    pattern_size: Tuple[int, int],
    square_size: float,
    distortion: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Board pose (R_cb, t_cb) and reprojection RMS [px] from detected corners (PnP)."""
    objp = checkerboard_object_points(pattern_size, square_size)
    img = np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2)
    if len(img) != len(objp):
        raise ValueError(f"expected {len(objp)} corners, got {len(img)}")
    dist = np.zeros(5) if distortion is None else np.asarray(distortion, dtype=float).reshape(-1)
    mat = intrinsics.matrix

    ok, rvec, tvec = cv2.solvePnP(objp, img, mat, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    rvec, tvec = cv2.solvePnPRefineLM(objp, img, mat, dist, rvec, tvec)
    projected, _ = cv2.projectPoints(objp, rvec, tvec, mat, dist)
    diff = projected.reshape(-1, 2) - img.reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
    rot, _ = cv2.Rodrigues(rvec)
    return rot, tvec.reshape(3).copy(), rms


def observe_board(
    scene: CalibrationScene,
    camera: RGBCamera,
    pose: Pose,
    intrinsics: Optional[CameraIntrinsics] = None,
    distortion: Optional[Sequence[float]] = None,
    max_rms_px: float = 1.0,
) -> Optional[BoardObservation]:
    """Place the board, render, detect corners and estimate its pose from the image.

    Returns None when the board is not detected or the pose fit is poor. The pose
    is estimated from the image only (no ground truth), using `intrinsics`
    (default: the nominal camera intrinsics).
    """
    intrinsics = intrinsics or camera.intrinsics
    corners = capture_board_corners(scene, camera, pose)
    if corners is None:
        return None
    rotation, translation, rms = estimate_board_pose(
        corners, intrinsics, scene.pattern_size, scene.square_size, distortion
    )
    if translation[2] <= 0.0 or rms > max_rms_px:
        return None
    return BoardObservation(corners, rotation, translation, rms)