"""Perception package: point clouds, clustering, object features and detectors."""

from .object_detection import (
    ClusterDetector,
    Detection,
    DetectorConfig,
    LaserDetector,
    LidarDetector,
    RGBDDetector,
    default_detector_config,
)
from .object_features import (
    ColorDescriptor,
    ObjectFeatures,
    SphereFit,
    color_descriptor,
    compute_object_features,
    fit_sphere,
    hue_difference_deg,
    is_sphere_like,
    rgb_to_hsv,
)
from .pointcloud import (
    above_ground_mask,
    colors_at_pixels,
    crop_mask,
    depth_to_world_cloud,
    finite_points,
    laser_points_world,
    lidar_points_world,
    voxel_downsample,
)
from .segmentation import ClusterConfig, cluster_labels, cluster_points, split_clusters

__all__ = [
    "ClusterDetector",
    "Detection",
    "DetectorConfig",
    "LaserDetector",
    "LidarDetector",
    "RGBDDetector",
    "default_detector_config",
    "ColorDescriptor",
    "ObjectFeatures",
    "SphereFit",
    "color_descriptor",
    "compute_object_features",
    "fit_sphere",
    "hue_difference_deg",
    "is_sphere_like",
    "rgb_to_hsv",
    "above_ground_mask",
    "colors_at_pixels",
    "crop_mask",
    "depth_to_world_cloud",
    "finite_points",
    "laser_points_world",
    "lidar_points_world",
    "voxel_downsample",
    "ClusterConfig",
    "cluster_labels",
    "cluster_points",
    "split_clusters",
]