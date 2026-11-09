"""Object detectors: point clouds -> list of world-frame object detections.

Pipeline (the same for every sensor):
    world points -> drop non-finite -> drop ground -> crop to the arena
                -> Euclidean clustering -> per-cluster features -> Detection

A Detection reports what was measured (position, shape features, colour). The
measurement covariance is assigned later by the fusion module, not here.
Ground truth is never used by the detectors.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import numpy as np

from sensors.laser_scanner import LaserScan
from sensors.lidar import LidarScan
from sensors.rgb_camera import CameraIntrinsics

from .object_features import (
    ColorDescriptor,
    ObjectFeatures,
    color_descriptor,
    compute_object_features,
)
from .pointcloud import (
    above_ground_mask,
    colors_at_pixels,
    crop_mask,
    depth_to_world_cloud,
    laser_points_world,
    lidar_points_world,
)
from .segmentation import ClusterConfig, cluster_points

CENTER_MODES = ("centroid", "sphere")


@dataclass
class DetectorConfig:
    """Detector parameters.

    center_mode:
        "centroid": position = mean of the visible surface points. This is biased
            toward the sensor (about half the object depth along the line of sight).
        "sphere": position = fitted sphere centre when the cluster looks like a
            sphere, otherwise the centroid.
    max_extent: clusters with a bounding-box side above this [m] are flagged
        ambiguous (probably several touching objects merged into one cluster).
    """

    sensor: str
    cluster: ClusterConfig
    ground_z: float = 0.0
    ground_margin: float = 0.04
    bounds_min: Tuple[float, float, float] = (-2.5, -2.5, 0.0)
    bounds_max: Tuple[float, float, float] = (2.5, 2.5, 3.0)
    max_extent: float = 0.5
    center_mode: str = "centroid"

    def __post_init__(self) -> None:
        if not self.sensor:
            raise ValueError("sensor name must not be empty")
        if self.center_mode not in CENTER_MODES:
            raise ValueError(f"center_mode must be one of {CENTER_MODES}")
        if self.ground_margin < 0.0:
            raise ValueError("ground_margin must be >= 0")
        if self.max_extent <= 0.0:
            raise ValueError("max_extent must be positive")
        lo = np.asarray(self.bounds_min, dtype=float)
        hi = np.asarray(self.bounds_max, dtype=float)
        if lo.shape != (3,) or hi.shape != (3,):
            raise ValueError("bounds must have 3 elements")
        if np.any(hi < lo):
            raise ValueError("bounds_max must be >= bounds_min on every axis")


def default_detector_config(sensor: str, **overrides) -> DetectorConfig:
    """Default configuration for "rgbd", "laser" or "lidar", with optional field overrides."""
    key = sensor.lower()
    if key == "rgbd":
        cluster = ClusterConfig(radius=0.05, min_points=12)  # dense cloud
    elif key in ("laser", "lidar"):
        cluster = ClusterConfig(radius=0.15, min_points=3)  # sparse cloud
    else:
        raise ValueError(f"unknown sensor {sensor!r}; use 'rgbd', 'laser' or 'lidar'")
    config = DetectorConfig(sensor=key, cluster=cluster)
    return replace(config, **overrides) if overrides else config


@dataclass(frozen=True, eq=False)
class Detection:
    """One detected object in the world frame."""

    sensor: str
    time: float
    position: np.ndarray  # (3,) measured position (see DetectorConfig.center_mode)
    features: ObjectFeatures
    color: Optional[ColorDescriptor]  # None for sensors without colour (LiDAR, laser)
    ambiguous: bool  # True if the cluster is larger than max_extent
    points: np.ndarray  # (K, 3) world points of the cluster

    @property
    def num_points(self) -> int:
        return int(len(self.points))


class ClusterDetector:
    """Detector working on a world-frame point cloud."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config

    def detect_points(self, points_world, time: float = 0.0, colors=None) -> List[Detection]:
        """Detect objects in an (N, 3) world cloud, with optional (N, 3) point colours.

        Detections are ordered by decreasing number of points.
        """
        cfg = self.config
        pts = np.asarray(points_world, dtype=float).reshape(-1, 3)
        cols = None
        if colors is not None:
            cols = np.asarray(colors).reshape(-1, 3)
            if len(cols) != len(pts):
                raise ValueError("colors must have one row per point")

        keep = (
            np.isfinite(pts).all(axis=1)
            & above_ground_mask(pts, cfg.ground_z, cfg.ground_margin)
            & crop_mask(pts, cfg.bounds_min, cfg.bounds_max)
        )
        pts = pts[keep]
        if cols is not None:
            cols = cols[keep]
        if len(pts) < cfg.cluster.min_points:
            return []

        detections: List[Detection] = []
        for indices in cluster_points(pts, cfg.cluster):
            cluster = pts[indices]
            features = compute_object_features(cluster)
            position = features.centroid.copy()
            if cfg.center_mode == "sphere" and features.sphere is not None:
                position = features.sphere.center.copy()
            color = color_descriptor(cols[indices]) if cols is not None else None
            detections.append(
                Detection(
                    sensor=cfg.sensor,
                    time=float(time),
                    position=position,
                    features=features,
                    color=color,
                    ambiguous=bool(np.max(features.dimensions) > cfg.max_extent),
                    points=cluster,
                )
            )
        return detections


def _check_transform(transform, name: str) -> np.ndarray:
    mat = np.asarray(transform, dtype=float)
    if mat.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 transform")
    return mat


class RGBDDetector(ClusterDetector):
    """RGB-D detector: depth image -> world cloud, RGB gives the colour descriptor."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        t_w_c,
        config: Optional[DetectorConfig] = None,
        stride: int = 2,
    ) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        super().__init__(config or default_detector_config("rgbd"))
        self.intrinsics = intrinsics
        self.t_w_c = _check_transform(t_w_c, "t_w_c")
        self.stride = int(stride)

    def detect(self, rgb: Optional[np.ndarray], depth: np.ndarray, time: float = 0.0) -> List[Detection]:
        """Detect objects from an RGB image (H, W, 3) (or None) and a depth image (H, W)."""
        depth = np.asarray(depth)
        if rgb is not None and np.asarray(rgb).shape[:2] != depth.shape:
            raise ValueError("rgb and depth images must have the same height and width")
        points, pixels = depth_to_world_cloud(depth, self.intrinsics, self.t_w_c, self.stride)
        colors = colors_at_pixels(rgb, pixels) if rgb is not None else None
        return self.detect_points(points, time, colors)


class LidarDetector(ClusterDetector):
    """LiDAR detector: scan -> world cloud (using T_W_R) -> clusters."""

    def __init__(self, t_w_r, config: Optional[DetectorConfig] = None) -> None:
        super().__init__(config or default_detector_config("lidar"))
        self.t_w_r = _check_transform(t_w_r, "t_w_r")

    def detect(self, scan: LidarScan) -> List[Detection]:
        return self.detect_points(lidar_points_world(scan, self.t_w_r), scan.time)


class LaserDetector(ClusterDetector):
    """Laser line scanner detector: scan -> world cloud (using T_W_L) -> clusters."""

    def __init__(self, t_w_l, config: Optional[DetectorConfig] = None) -> None:
        super().__init__(config or default_detector_config("laser"))
        self.t_w_l = _check_transform(t_w_l, "t_w_l")

    def detect(self, scan: LaserScan) -> List[Detection]:
        return self.detect_points(laser_points_world(scan, self.t_w_l), scan.time)