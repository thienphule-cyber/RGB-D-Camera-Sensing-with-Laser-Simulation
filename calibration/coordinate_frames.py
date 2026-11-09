"""Coordinate frames, rigid transforms and plane geometry for calibration.

Frames (see the design document):
    W  world (MuJoCo global frame)
    B  sensor assembly (mounting body)
    C  RGB-D camera, OpenCV optical frame (x right, y down, z forward)
    L  laser scanner (x forward, y left, z up)
    R  LiDAR (x forward, y left, z up)
RGB and depth share one optical centre in this simulation, so the depth frame D
is identical to C.

Convention: T_a_b is a 4x4 homogeneous matrix mapping points expressed in
frame b into frame a, i.e. p_a = T_a_b @ p_b.

Planes are stored as n . x + d = 0 with a unit normal. Planes returned by this
module face the sensor origin (d >= 0), so d is the distance from the origin.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import mujoco
import numpy as np

from sensors.common import FORWARD_Y_MOUNT
from sensors.rgb_camera import ground_truth_camera_pose_opencv
from simulation.environment import Environment


# ------------------------------------------------------------------ transforms
def make_transform(rotation, translation) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a 3x3 rotation and a translation (3,)."""
    rot = np.asarray(rotation, dtype=float)
    if rot.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got shape {rot.shape}")
    trans = np.asarray(translation, dtype=float).reshape(3)
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = trans
    return transform


def split_transform(transform) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rotation (3, 3), translation (3,)) of a 4x4 transform."""
    mat = np.asarray(transform, dtype=float)
    if mat.shape != (4, 4):
        raise ValueError(f"transform must be 4x4, got shape {mat.shape}")
    return mat[:3, :3].copy(), mat[:3, 3].copy()


def invert_transform(transform) -> np.ndarray:
    """Inverse of a rigid transform (exploits R^-1 = R^T)."""
    rot, trans = split_transform(transform)
    return make_transform(rot.T, -rot.T @ trans)


def compose_transforms(*transforms) -> np.ndarray:
    """Matrix product T1 @ T2 @ ... (identity when called with no arguments)."""
    result = np.eye(4)
    for transform in transforms:
        result = result @ np.asarray(transform, dtype=float)
    return result


def apply_transform(transform, points) -> np.ndarray:
    """Apply a 4x4 transform to an (N, 3) array of points."""
    rot, trans = split_transform(transform)
    return np.asarray(points, dtype=float) @ rot.T + trans


def rotation_angle_deg(rotation_a, rotation_b) -> float:
    """Geodesic angle [deg] between two rotation matrices."""
    rel = np.asarray(rotation_a, dtype=float).T @ np.asarray(rotation_b, dtype=float)
    cos_angle = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def transform_error(estimated, truth) -> Tuple[float, float]:
    """(rotation error [deg], translation error [m]) between two 4x4 transforms."""
    rot_e, trans_e = split_transform(estimated)
    rot_t, trans_t = split_transform(truth)
    return rotation_angle_deg(rot_e, rot_t), float(np.linalg.norm(trans_e - trans_t))


# ------------------------------------------------------------------ frame tree
class FrameTree:
    """Graph of named frames connected by rigid transforms.

    add_transform("W", "C", T_W_C) stores the edge in both directions, and
    get_transform(target, source) chains edges to return T_target_source.
    """

    def __init__(self) -> None:
        self._transforms: Dict[Tuple[str, str], np.ndarray] = {}
        self._neighbors: Dict[str, List[str]] = {}

    @property
    def frames(self) -> List[str]:
        return sorted(self._neighbors)

    def add_transform(self, parent: str, child: str, transform_parent_child) -> None:
        """Register T_parent_child (maps child coordinates into parent coordinates)."""
        if parent == child:
            raise ValueError("parent and child must be different frames")
        transform = np.asarray(transform_parent_child, dtype=float)
        if transform.shape != (4, 4):
            raise ValueError("transform must be 4x4")
        self._transforms[(parent, child)] = transform.copy()
        self._transforms[(child, parent)] = invert_transform(transform)
        for a, b in ((parent, child), (child, parent)):
            neighbors = self._neighbors.setdefault(a, [])
            if b not in neighbors:
                neighbors.append(b)

    def get_transform(self, target: str, source: str) -> np.ndarray:
        """Return T_target_source (p_target = T_target_source @ p_source)."""
        for name in (target, source):
            if name not in self._neighbors:
                raise KeyError(f"unknown frame {name!r}")
        if target == source:
            return np.eye(4)

        parents: Dict[str, str] = {source: ""}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            if node == target:
                break
            for neighbor in self._neighbors[node]:
                if neighbor not in parents:
                    parents[neighbor] = node
                    queue.append(neighbor)
        if target not in parents:
            raise ValueError(f"no transform path between {source!r} and {target!r}")

        # Walk back from the target; the leftmost factor is the last hop.
        result = np.eye(4)
        node = target
        while node != source:
            previous = parents[node]
            result = result @ self._transforms[(node, previous)]
            node = previous
        return result

    def transform_points(self, points, target: str, source: str) -> np.ndarray:
        """Express (N, 3) points given in `source` in the `target` frame."""
        return apply_transform(self.get_transform(target, source), points)


def ground_truth_frame_tree(env: Environment) -> FrameTree:
    """Ground-truth frame tree W, B, C, L, R read from the MuJoCo model.

    Use for evaluating calibration only, never inside the estimation pipeline.
    """
    model, data = env.model, env.data
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "sensor_assembly")
    if body_id < 0:
        raise RuntimeError("body 'sensor_assembly' not found")
    t_w_b = make_transform(data.xmat[body_id].reshape(3, 3), data.xpos[body_id])

    cam_pos, cam_rot = ground_truth_camera_pose_opencv(env)
    t_w_c = make_transform(cam_rot, cam_pos)

    lidar_pos, lidar_rot = env.get_site_pose(env.config.lidar_site_name)
    t_w_r = make_transform(lidar_rot @ FORWARD_Y_MOUNT, lidar_pos)

    laser_pos, laser_rot = env.get_site_pose(env.config.laser_site_name)
    t_w_l = make_transform(laser_rot @ FORWARD_Y_MOUNT, laser_pos)

    t_b_w = invert_transform(t_w_b)
    tree = FrameTree()
    tree.add_transform("W", "B", t_w_b)
    tree.add_transform("B", "C", t_b_w @ t_w_c)
    tree.add_transform("B", "L", t_b_w @ t_w_l)
    tree.add_transform("B", "R", t_b_w @ t_w_r)
    return tree


# ---------------------------------------------------------------------- planes
@dataclass(frozen=True, eq=False)
class Plane:
    """Plane n . x + d = 0 (the constructor normalizes n to unit length)."""

    normal: np.ndarray
    d: float

    def __post_init__(self) -> None:
        n = np.asarray(self.normal, dtype=float).reshape(3)
        norm = float(np.linalg.norm(n))
        if norm < 1e-12:
            raise ValueError("plane normal must be non-zero")
        object.__setattr__(self, "normal", n / norm)
        object.__setattr__(self, "d", float(self.d) / norm)

    def signed_distance(self, points) -> np.ndarray:
        """Signed distance [m] of (N, 3) points to the plane."""
        return np.asarray(points, dtype=float).reshape(-1, 3) @ self.normal + self.d

    def facing_origin(self) -> "Plane":
        """Same plane with the normal flipped so that d >= 0."""
        if self.d < 0.0:
            return Plane(-self.normal, -self.d)
        return self

    def angle_to(self, other: "Plane") -> float:
        """Angle [deg] between the two planes (normal orientation is ignored)."""
        cos_angle = np.clip(abs(float(self.normal @ other.normal)), 0.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def transformed(self, transform_new_old) -> "Plane":
        """Express the plane in a new frame, given T_new_old."""
        rot, trans = split_transform(transform_new_old)
        normal = rot @ self.normal
        return Plane(normal, self.d - float(normal @ trans))

    def intersect_rays(self, directions, origin=(0.0, 0.0, 0.0)) -> np.ndarray:
        """Intersect rays origin + lambda * direction with the plane.

        Returns (N, 3) points; rows are NaN when the ray is parallel to the plane
        or the intersection lies behind the origin.
        """
        dirs = np.asarray(directions, dtype=float).reshape(-1, 3)
        orig = np.asarray(origin, dtype=float).reshape(3)
        denom = dirs @ self.normal
        numer = -(float(self.normal @ orig) + self.d)
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = numer / denom
        lam[(np.abs(denom) < 1e-9) | ~(lam > 0.0)] = np.nan
        return orig + lam[:, None] * dirs


def fit_plane(points, robust_iterations: int = 0, inlier_sigma: float = 3.0) -> Plane:
    """Least-squares plane through (N, 3) points (SVD), facing the origin.

    With robust_iterations > 0 the fit is repeated on the points whose residual is
    within inlier_sigma * (robust sigma from the median residual), which removes
    outliers such as board edges or stray returns.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) < 3:
        raise ValueError("need at least 3 points to fit a plane")
    mask = np.ones(len(pts), dtype=bool)
    normal = np.array([0.0, 0.0, 1.0])
    d = 0.0
    for iteration in range(robust_iterations + 1):
        centroid = pts[mask].mean(axis=0)
        _, _, vt = np.linalg.svd(pts[mask] - centroid, full_matrices=False)
        normal = vt[-1]
        d = -float(normal @ centroid)
        if iteration == robust_iterations:
            break
        residuals = np.abs(pts @ normal + d)
        sigma = 1.4826 * float(np.median(residuals[mask]))
        new_mask = residuals <= max(inlier_sigma * sigma, 1e-6)
        if int(new_mask.sum()) < 3:
            break
        mask = new_mask
    return Plane(normal, d).facing_origin()


def board_plane(rotation, translation) -> Plane:
    """Plane of a checkerboard in the camera frame, normal facing the camera.

    (rotation, translation) is the board pose T_cam_board in the OpenCV
    convention: the board frame z axis points away from the camera.
    """
    rot = np.asarray(rotation, dtype=float)
    z_axis = rot[:, 2]
    return Plane(-z_axis, float(z_axis @ np.asarray(translation, dtype=float))).facing_origin()