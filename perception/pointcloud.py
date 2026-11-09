"""Point cloud utilities: conversion to the world frame, filtering and downsampling.

All clouds are (N, 3) float arrays. Functions named *_world return points in the
world frame W, given the (calibrated) transform of the sensor frame.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from calibration.coordinate_frames import apply_transform
from sensors.laser_scanner import LaserScan
from sensors.lidar import LidarScan
from sensors.rgb_camera import CameraIntrinsics


def finite_points(points) -> np.ndarray:
    """Return the rows of an (N, 3) array whose coordinates are all finite."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    return pts[np.isfinite(pts).all(axis=1)]


def voxel_downsample(points, voxel_size: float) -> np.ndarray:
    """Replace all points inside each cubic voxel by their centroid.

    Voxels are returned in lexicographic order of their integer index.
    """
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")
    pts = finite_points(points)
    if len(pts) == 0:
        return pts
    keys = np.floor(pts / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    sums = np.zeros((len(counts), 3))
    np.add.at(sums, inverse, pts)
    return sums / counts[:, None]


def above_ground_mask(points, ground_z: float = 0.0, margin: float = 0.04) -> np.ndarray:
    """Boolean mask of the points higher than ground_z + margin (world frame, z up).

    The margin absorbs depth noise and the thin layer where objects touch the floor.
    """
    if margin < 0.0:
        raise ValueError("margin must be >= 0")
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    return pts[:, 2] > ground_z + margin


def crop_mask(points, bounds_min, bounds_max) -> np.ndarray:
    """Boolean mask of the points inside the axis-aligned box [bounds_min, bounds_max]."""
    lo = np.asarray(bounds_min, dtype=float).reshape(3)
    hi = np.asarray(bounds_max, dtype=float).reshape(3)
    if np.any(hi < lo):
        raise ValueError("bounds_max must be >= bounds_min on every axis")
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    return np.all((pts >= lo) & (pts <= hi), axis=1)


def depth_to_world_cloud(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    t_w_c,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project a depth image to a world-frame cloud.

    Args:
        depth: (H, W) depth along the optical axis [m]; NaN or <= 0 means invalid.
        intrinsics: camera intrinsics (OpenCV optical frame: x right, y down, z forward).
        t_w_c: 4x4 transform T_W_C (camera frame -> world frame).
        stride: keep every `stride`-th row and column (1 = full resolution).

    Returns:
        (points (N, 3) in the world frame, pixels (N, 2) as integer [row, col]).
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError("depth must be a 2D array")
    height, width = depth.shape
    if (width, height) != (intrinsics.width, intrinsics.height):
        raise ValueError("depth image size does not match the intrinsics")

    rows = np.arange(0, height, stride)
    cols = np.arange(0, width, stride)
    grid_c, grid_r = np.meshgrid(cols, rows)
    z = depth[grid_r, grid_c].astype(np.float64)
    valid = np.isfinite(z) & (z > 0.0)

    z = z[valid]
    u = grid_c[valid].astype(np.float64)
    v = grid_r[valid].astype(np.float64)
    cam = np.column_stack(
        [
            (u - intrinsics.cx) * z / intrinsics.fx,
            (v - intrinsics.cy) * z / intrinsics.fy,
            z,
        ]
    )
    pixels = np.column_stack([grid_r[valid], grid_c[valid]])
    return apply_transform(t_w_c, cam), pixels


def colors_at_pixels(image: np.ndarray, pixels) -> np.ndarray:
    """RGB values (N, 3) of an (H, W, 3) image at integer pixels (N, 2) = [row, col]."""
    img = np.asarray(image)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")
    pix = np.asarray(pixels, dtype=int).reshape(-1, 2)
    return img[pix[:, 0], pix[:, 1]]


def lidar_points_world(scan: LidarScan, t_w_r) -> np.ndarray:
    """Valid LiDAR returns of a scan as (M, 3) world points (t_w_r = T_W_R)."""
    return apply_transform(t_w_r, scan.points())


def laser_points_world(scan: LaserScan, t_w_l) -> np.ndarray:
    """Valid laser spots of a scan as (M, 3) world points (t_w_l = T_W_L)."""
    return apply_transform(t_w_l, scan.points())