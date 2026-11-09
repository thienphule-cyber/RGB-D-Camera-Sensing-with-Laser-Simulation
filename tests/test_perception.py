"""Unit tests for the perception package."""

import numpy as np
import pytest

from calibration.coordinate_frames import (
    apply_transform,
    ground_truth_frame_tree,
    make_transform,
)
from perception.object_detection import (
    ClusterDetector,
    DetectorConfig,
    LaserDetector,
    LidarDetector,
    RGBDDetector,
    default_detector_config,
)
from perception.object_features import (
    SphereFit,
    color_descriptor,
    compute_object_features,
    fit_sphere,
    hue_difference_deg,
    is_sphere_like,
    rgb_to_hsv,
)
from perception.pointcloud import (
    above_ground_mask,
    colors_at_pixels,
    crop_mask,
    depth_to_world_cloud,
    finite_points,
    laser_points_world,
    lidar_points_world,
    voxel_downsample,
)
from perception.segmentation import (
    ClusterConfig,
    cluster_labels,
    cluster_points,
    split_clusters,
)
from sensors.depth_camera import DepthCamera
from sensors.laser_scanner import LaserConfig, LaserScan, LaserScanner
from sensors.lidar import Lidar, LidarConfig, LidarScan
from sensors.rgb_camera import CameraIntrinsics, RGBCamera
from simulation.environment import Environment
from simulation.motion_generator import LinearBounceTrajectory
from simulation.obstacle_manager import ObstacleSpec


# ------------------------------------------------------------------ helpers
def make_env(spheres):
    """Environment with stationary spheres: [(id, center, radius[, rgba]), ...]."""
    specs = []
    for entry in spheres:
        oid, center, radius = entry[:3]
        rgba = entry[3] if len(entry) > 3 else (0.9, 0.1, 0.1, 1.0)
        traj = LinearBounceTrajectory(center, (0.0, 0.0, 0.0), center, center)
        specs.append(ObstacleSpec(oid, "sphere", (radius,), "3D", traj, rgba=rgba))
    return Environment(obstacles=specs)


def make_cap(center, radius, n=3000, half_angle_deg=80.0, toward=(0.0, -1.0, 0.0)):
    """Points on the part of a sphere facing `toward` (a stand-in for the visible surface)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + np.sqrt(5.0)) * i
    dirs = np.column_stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    )
    axis = np.asarray(toward, dtype=float)
    axis = axis / np.linalg.norm(axis)
    keep = dirs @ axis >= np.cos(np.radians(half_angle_deg))
    return np.asarray(center, dtype=float) + radius * dirs[keep]


def floor_grid(half=1.5, step=0.05):
    axis = np.arange(-half, half + 1e-9, step)
    gx, gy = np.meshgrid(axis, axis)
    return np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])


def nearest(detections, point):
    return min(detections, key=lambda d: np.linalg.norm(d.position - np.asarray(point)))


# ---------------------------------------------------------------- point cloud
class TestPointCloud:
    def test_finite_points_drops_nan_and_inf(self):
        pts = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [1.0, np.inf, 0.0], [1.0, 2.0, 3.0]])
        out = finite_points(pts)
        assert np.allclose(out, [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])

    def test_voxel_downsample_merges_points_in_a_voxel(self):
        pts = np.array([[0.01, 0.01, 0.01], [0.03, 0.03, 0.03], [0.55, 0.55, 0.55]])
        out = voxel_downsample(pts, 0.1)
        assert out.shape == (2, 3)
        assert np.allclose(out[0], [0.02, 0.02, 0.02])
        assert np.allclose(out[1], [0.55, 0.55, 0.55])

    def test_voxel_downsample_reduces_dense_cloud(self):
        rng = np.random.default_rng(0)
        pts = rng.uniform(0.0, 1.0, (5000, 3))
        out = voxel_downsample(pts, 0.1)
        assert len(out) <= 1000
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_voxel_downsample_invalid_size_raises(self):
        with pytest.raises(ValueError):
            voxel_downsample(np.zeros((3, 3)), 0.0)

    def test_voxel_downsample_empty(self):
        assert voxel_downsample(np.zeros((0, 3)), 0.1).shape == (0, 3)

    def test_above_ground_mask(self):
        pts = np.array([[0, 0, 0.0], [0, 0, 0.03], [0, 0, 0.05], [0, 0, 1.0]])
        assert np.array_equal(above_ground_mask(pts, 0.0, 0.04), [False, False, True, True])
        assert np.array_equal(above_ground_mask(pts, 0.1, 0.04), [False, False, False, True])

    def test_above_ground_mask_negative_margin_raises(self):
        with pytest.raises(ValueError):
            above_ground_mask(np.zeros((1, 3)), margin=-0.1)

    def test_crop_mask(self):
        pts = np.array([[0.0, 0.0, 1.0], [3.0, 0.0, 1.0], [0.0, 0.0, 5.0], [-1.0, 1.0, 0.5]])
        mask = crop_mask(pts, (-2.0, -2.0, 0.0), (2.0, 2.0, 3.0))
        assert np.array_equal(mask, [True, False, False, True])

    def test_crop_mask_invalid_bounds_raise(self):
        with pytest.raises(ValueError):
            crop_mask(np.zeros((1, 3)), (1.0, 0.0, 0.0), (0.0, 1.0, 1.0))

    def test_depth_to_world_cloud_identity(self):
        intr = CameraIntrinsics.from_fovy(64, 48, 60.0)
        depth = np.full((48, 64), 2.0, dtype=np.float32)
        pts, pix = depth_to_world_cloud(depth, intr, np.eye(4))
        assert pts.shape == (48 * 64, 3)
        assert np.allclose(pts[:, 2], 2.0)
        uv = intr.project(pts)  # projecting back must return the source pixels
        assert np.allclose(uv[:, 0], pix[:, 1]) and np.allclose(uv[:, 1], pix[:, 0])

    def test_depth_to_world_cloud_applies_transform(self):
        intr = CameraIntrinsics.from_fovy(64, 48, 60.0)
        depth = np.full((48, 64), 2.0, dtype=np.float32)
        transform = make_transform(np.eye(3), [1.0, 2.0, 3.0])
        pts, _ = depth_to_world_cloud(depth, intr, transform)
        assert np.allclose(pts[:, 2], 5.0)
        assert np.allclose(pts.mean(axis=0)[:2], [1.0 + (31.5 - intr.cx) * 2.0 / intr.fx * 0.0, 2.0], atol=0.05)

    def test_depth_to_world_cloud_stride_and_invalid_pixels(self):
        intr = CameraIntrinsics.from_fovy(64, 48, 60.0)
        depth = np.full((48, 64), 2.0, dtype=np.float32)
        pts, _ = depth_to_world_cloud(depth, intr, np.eye(4), stride=4)
        assert len(pts) == 12 * 16
        depth[10, 10] = np.nan
        depth[20, 20] = 0.0
        pts, pix = depth_to_world_cloud(depth, intr, np.eye(4))
        assert len(pts) == 48 * 64 - 2
        assert not any((p == [10, 10]).all() for p in pix)

    def test_depth_to_world_cloud_rejects_bad_input(self):
        intr = CameraIntrinsics.from_fovy(64, 48, 60.0)
        with pytest.raises(ValueError):
            depth_to_world_cloud(np.zeros((10, 10)), intr, np.eye(4))
        with pytest.raises(ValueError):
            depth_to_world_cloud(np.zeros((48, 64)), intr, np.eye(4), stride=0)
        with pytest.raises(ValueError):
            depth_to_world_cloud(np.zeros(5), intr, np.eye(4))

    def test_colors_at_pixels(self):
        image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        out = colors_at_pixels(image, [[1, 2], [0, 0]])
        assert np.array_equal(out[0], image[1, 2]) and np.array_equal(out[1], image[0, 0])

    def test_colors_at_pixels_rejects_gray_image(self):
        with pytest.raises(ValueError):
            colors_at_pixels(np.zeros((4, 5)), [[0, 0]])

    def test_lidar_points_world(self):
        azimuth = np.radians([-10.0, 0.0, 10.0])
        elevation = np.radians([0.0, 10.0])
        ranges = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, 1.5]])
        scan = LidarScan(0.5, azimuth, elevation, ranges, np.full((2, 3), -1))
        transform = make_transform(np.eye(3), [0.0, 0.0, 1.0])
        out = lidar_points_world(scan, transform)
        assert out.shape == (4, 3)
        assert np.allclose(out, scan.points() + [0.0, 0.0, 1.0])

    def test_laser_points_world(self):
        yaw = np.radians([0.0, 10.0])
        fan = np.radians([-5.0, 0.0, 5.0])
        ranges = np.array([[1.0, 2.0, np.nan], [np.nan, 1.5, 1.2]])
        scan = LaserScan(0.0, yaw, fan, ranges, np.full((2, 3), -1))
        transform = make_transform(np.eye(3), [1.0, 0.0, 0.0])
        out = laser_points_world(scan, transform)
        assert out.shape == (4, 3)
        assert np.allclose(out, scan.points() + [1.0, 0.0, 0.0])


# --------------------------------------------------------------- segmentation
class TestSegmentation:
    def test_two_blobs_and_a_noise_point(self):
        rng = np.random.default_rng(0)
        blob_a = np.array([0.0, 0.0, 1.0]) + rng.uniform(-0.05, 0.05, (200, 3))
        blob_b = np.array([1.0, 0.0, 1.0]) + rng.uniform(-0.05, 0.05, (100, 3))
        noise = np.array([[3.0, 3.0, 1.0]])
        points = np.vstack([blob_a, blob_b, noise])
        labels = cluster_labels(points, radius=0.08, min_points=5)
        assert np.all(labels[:200] == 0)  # larger cluster gets label 0
        assert np.all(labels[200:300] == 1)
        assert labels[300] == -1

    def test_radius_decides_connectivity(self):
        radius = 0.1
        close = np.column_stack([np.arange(10) * 0.9 * radius, np.zeros(10), np.zeros(10)])
        far = np.column_stack([np.arange(10) * 1.5 * radius, np.zeros(10), np.zeros(10)])
        assert len(np.unique(cluster_labels(close, radius))) == 1
        assert len(np.unique(cluster_labels(far, radius))) == 10

    def test_min_points_filters_small_clusters(self):
        pts = np.array([[0, 0, 0], [0.01, 0, 0], [1, 0, 0], [1.01, 0, 0], [1.02, 0, 0]], dtype=float)
        assert np.array_equal(cluster_labels(pts, 0.05, min_points=3), [-1, -1, 0, 0, 0])
        assert np.array_equal(cluster_labels(pts, 0.05, min_points=1), [1, 1, 0, 0, 0])

    def test_empty_and_single_point(self):
        assert cluster_labels(np.zeros((0, 3)), 0.1).shape == (0,)
        assert np.array_equal(cluster_labels([[0.0, 0.0, 0.0]], 0.1), [0])
        assert np.array_equal(cluster_labels([[0.0, 0.0, 0.0]], 0.1, min_points=2), [-1])

    def test_invalid_arguments_raise(self):
        pts = np.zeros((3, 3))
        with pytest.raises(ValueError):
            cluster_labels(pts, 0.0)
        with pytest.raises(ValueError):
            cluster_labels(pts, 0.1, min_points=0)
        with pytest.raises(ValueError):
            cluster_labels([[np.nan, 0.0, 0.0]], 0.1)

    def test_split_clusters(self):
        groups = split_clusters(np.array([0, 0, 1, -1, 1]))
        assert [g.tolist() for g in groups] == [[0, 1], [2, 4]]
        assert split_clusters(np.array([-1, -1])) == []
        assert split_clusters(np.array([], dtype=int)) == []

    def test_cluster_points_returns_index_arrays(self):
        rng = np.random.default_rng(1)
        a = rng.uniform(-0.05, 0.05, (60, 3))
        b = np.array([2.0, 0.0, 0.0]) + rng.uniform(-0.05, 0.05, (30, 3))
        groups = cluster_points(np.vstack([a, b]), ClusterConfig(radius=0.08, min_points=5))
        assert len(groups) == 2
        assert len(groups[0]) == 60 and len(groups[1]) == 30
        assert np.all(groups[1] >= 60)

    def test_cluster_config_validation(self):
        with pytest.raises(ValueError):
            ClusterConfig(radius=0.0)
        with pytest.raises(ValueError):
            ClusterConfig(min_points=2)


# ------------------------------------------------------------ sphere & features
class TestSphereFit:
    def test_exact_cap_recovers_center_and_radius(self):
        center = np.array([0.3, 1.0, 0.8])
        fit = fit_sphere(make_cap(center, 0.15))
        assert fit is not None
        assert np.allclose(fit.center, center, atol=1e-6)
        assert fit.radius == pytest.approx(0.15, abs=1e-6)
        assert fit.rms < 1e-6
        assert fit.inlier_ratio == 1.0

    def test_noisy_cap(self):
        rng = np.random.default_rng(0)
        center = np.array([0.3, 1.0, 0.8])
        pts = make_cap(center, 0.15, n=2000)
        pts = pts + rng.normal(0.0, 0.001, pts.shape)
        fit = fit_sphere(pts)
        assert np.linalg.norm(fit.center - center) < 0.003
        assert fit.radius == pytest.approx(0.15, abs=0.003)

    def test_outliers_are_rejected(self):
        center = np.array([0.0, 1.0, 1.0])
        pts = make_cap(center, 0.15)
        rng = np.random.default_rng(3)
        directions = rng.normal(size=(10, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        outliers = center + directions * rng.uniform(0.23, 0.30, (10, 1))  # 8-15 cm off the surface
        fit = fit_sphere(np.vstack([pts, outliers]))
        assert np.allclose(fit.center, center, atol=1e-6)
        assert fit.inlier_ratio < 1.0

    def test_degenerate_inputs_return_none(self):
        assert fit_sphere(np.zeros((3, 3))) is None  # too few points
        gx, gz = np.meshgrid(np.linspace(-0.15, 0.15, 15), np.linspace(-0.15, 0.15, 15))
        plane = np.column_stack([gx.ravel(), np.full(gx.size, 2.0), gz.ravel()])
        assert fit_sphere(plane) is None  # coplanar
        line = np.column_stack([np.linspace(0, 1, 30), np.zeros(30), np.zeros(30)])
        assert fit_sphere(line) is None  # collinear

    def test_is_sphere_like_thresholds(self):
        good = SphereFit(np.zeros(3), 0.15, 0.001, 0.3, 1.0)
        assert is_sphere_like(good)
        assert not is_sphere_like(SphereFit(np.zeros(3), 2.0, 0.0, 3.0, 1.0))  # too large
        assert not is_sphere_like(SphereFit(np.zeros(3), 0.01, 0.0, 0.1, 1.0))  # too small
        assert not is_sphere_like(SphereFit(np.zeros(3), 0.15, 0.05, 0.3, 1.0))  # poor fit
        assert not is_sphere_like(SphereFit(np.zeros(3), 0.15, 0.001, 0.05, 1.0))  # tiny patch
        assert not is_sphere_like(SphereFit(np.zeros(3), 0.15, 0.001, 0.3, 0.3))  # many outliers


class TestObjectFeatures:
    def test_basic_geometry_of_a_box_face(self):
        gx, gz = np.meshgrid(np.linspace(-0.1, 0.1, 21), np.linspace(0.0, 0.3, 31))
        face = np.column_stack([gx.ravel() + 0.5, np.full(gx.size, 1.0), gz.ravel()])
        feats = compute_object_features(face)
        assert feats.num_points == 21 * 31
        assert np.allclose(feats.centroid, [0.5, 1.0, 0.15], atol=1e-9)
        assert np.allclose(feats.dimensions, [0.2, 0.0, 0.3], atol=1e-9)
        assert np.allclose(feats.bbox_center, [0.5, 1.0, 0.15], atol=1e-9)
        assert feats.covariance.shape == (3, 3)

    def test_flat_patch_is_planar_and_not_a_sphere(self):
        gx, gz = np.meshgrid(np.linspace(-0.15, 0.15, 21), np.linspace(-0.15, 0.15, 21))
        patch = np.column_stack([gx.ravel(), np.full(gx.size, 2.0), gz.ravel()])
        feats = compute_object_features(patch)
        assert feats.planarity > 0.95
        assert feats.linearity < 0.05
        assert feats.sphere is None and feats.shape_hint == "other"

    def test_noisy_flat_patch_is_not_a_sphere(self):
        rng = np.random.default_rng(5)
        gx, gz = np.meshgrid(np.linspace(-0.15, 0.15, 21), np.linspace(-0.15, 0.15, 21))
        patch = np.column_stack([gx.ravel(), np.full(gx.size, 2.0), gz.ravel()])
        patch = patch + rng.normal(0.0, 0.001, patch.shape)
        assert compute_object_features(patch).shape_hint == "other"

    def test_line_is_linear(self):
        line = np.column_stack([np.linspace(-1, 1, 50), np.zeros(50), np.zeros(50)])
        feats = compute_object_features(line)
        assert feats.linearity > 0.99

    def test_sphere_cap_gets_sphere_hint_and_center(self):
        center = np.array([0.4, 1.0, 1.2])
        feats = compute_object_features(make_cap(center, 0.15))
        assert feats.shape_hint == "sphere"
        assert np.allclose(feats.sphere.center, center, atol=1e-6)
        assert feats.centroid[1] < center[1]  # visible-surface centroid is biased toward the sensor
        assert feats.sphericity > 0.0

    def test_sphere_fit_can_be_disabled(self):
        feats = compute_object_features(make_cap([0, 1, 1], 0.15), fit_sphere_model=False)
        assert feats.sphere is None

    def test_yaw_of_a_line_in_the_xy_plane(self):
        t = np.linspace(-1.0, 1.0, 50)
        for angle_deg, expected_deg in ((30.0, 30.0), (120.0, -60.0), (0.0, 0.0)):
            a = np.radians(angle_deg)
            pts = np.column_stack([t * np.cos(a), t * np.sin(a), np.zeros(50)])
            yaw = compute_object_features(pts).yaw
            assert yaw == pytest.approx(np.radians(expected_deg), abs=1e-6)
            assert -np.pi / 2 <= yaw < np.pi / 2

    def test_needs_three_points(self):
        with pytest.raises(ValueError):
            compute_object_features(np.zeros((2, 3)))


class TestColor:
    def test_rgb_to_hsv_primary_colors(self):
        colors = np.array(
            [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
             [0, 255, 255], [255, 0, 255], [128, 128, 128]], dtype=np.uint8
        )
        hsv = rgb_to_hsv(colors)
        assert np.allclose(hsv[:, 0], [0, 120, 240, 60, 180, 300, 0])
        assert np.allclose(hsv[:, 1], [1, 1, 1, 1, 1, 1, 0])
        assert np.allclose(hsv[:, 2], [1, 1, 1, 1, 1, 1, 128 / 255])

    def test_rgb_to_hsv_accepts_float_input(self):
        hsv = rgb_to_hsv(np.array([[0.0, 0.0, 1.0]]))
        assert np.allclose(hsv[0], [240.0, 1.0, 1.0])

    def test_hue_difference(self):
        assert hue_difference_deg(350.0, 10.0) == pytest.approx(20.0)
        assert hue_difference_deg(10.0, 350.0) == pytest.approx(20.0)
        assert hue_difference_deg(0.0, 180.0) == pytest.approx(180.0)
        assert hue_difference_deg(90.0, 90.0) == pytest.approx(0.0)

    def test_descriptor_of_a_red_cluster(self):
        colors = np.tile(np.array([[200, 30, 30]], dtype=np.uint8), (50, 1))
        desc = color_descriptor(colors)
        assert hue_difference_deg(desc.hue_deg, 0.0) < 1e-6
        assert desc.chromaticity.sum() == pytest.approx(1.0)
        assert desc.chromaticity[0] > 0.6
        assert desc.saturation == pytest.approx(170 / 200)
        assert desc.value == pytest.approx(200 / 255)

    def test_circular_mean_hue_across_zero(self):
        colors = np.array([[255, 43, 0], [255, 0, 43]], dtype=np.uint8)  # about +10 and -10 degrees
        desc = color_descriptor(colors)
        assert hue_difference_deg(desc.hue_deg, 0.0) < 1e-6

    def test_gray_has_no_hue(self):
        desc = color_descriptor(np.full((10, 3), 128, dtype=np.uint8))
        assert np.isnan(desc.hue_deg)
        assert desc.saturation == 0.0
        assert np.allclose(desc.chromaticity, 1.0 / 3.0)

    def test_empty_colors_raise(self):
        with pytest.raises(ValueError):
            color_descriptor(np.zeros((0, 3), dtype=np.uint8))


# ----------------------------------------------------------- synthetic detector
CENTER_A, RADIUS_A = np.array([-0.5, 0.5, 0.9]), 0.15
CENTER_B, RADIUS_B = np.array([0.6, 1.0, 1.2]), 0.10


@pytest.fixture(scope="module")
def synthetic_cloud():
    return np.vstack(
        [floor_grid(), make_cap(CENTER_A, RADIUS_A), make_cap(CENTER_B, RADIUS_B)]
    )


class TestClusterDetector:
    def test_detects_two_spheres_and_ignores_the_floor(self, synthetic_cloud):
        detector = ClusterDetector(default_detector_config("rgbd", center_mode="sphere"))
        detections = detector.detect_points(synthetic_cloud, time=1.5)
        assert len(detections) == 2
        assert all(d.time == 1.5 and d.sensor == "rgbd" for d in detections)
        assert np.allclose(nearest(detections, CENTER_A).position, CENTER_A, atol=1e-6)
        assert np.allclose(nearest(detections, CENTER_B).position, CENTER_B, atol=1e-6)
        assert all(d.features.shape_hint == "sphere" for d in detections)
        assert all(not d.ambiguous for d in detections)

    def test_centroid_mode_is_biased_toward_the_sensor(self, synthetic_cloud):
        sphere = ClusterDetector(default_detector_config("rgbd", center_mode="sphere"))
        centroid = ClusterDetector(default_detector_config("rgbd", center_mode="centroid"))
        err_sphere = np.linalg.norm(nearest(sphere.detect_points(synthetic_cloud), CENTER_A).position - CENTER_A)
        det_c = nearest(centroid.detect_points(synthetic_cloud), CENTER_A)
        err_centroid = np.linalg.norm(det_c.position - CENTER_A)
        assert err_sphere < 1e-6
        assert err_centroid > 0.05
        assert det_c.position[1] < CENTER_A[1]

    def test_floor_only_gives_no_detections(self):
        detector = ClusterDetector(default_detector_config("rgbd"))
        assert detector.detect_points(floor_grid()) == []

    def test_empty_input(self):
        detector = ClusterDetector(default_detector_config("lidar"))
        assert detector.detect_points(np.zeros((0, 3))) == []

    def test_points_outside_the_arena_are_cropped(self):
        detector = ClusterDetector(default_detector_config("rgbd"))
        assert detector.detect_points(make_cap([4.0, 1.0, 1.0], 0.15)) == []
        assert detector.detect_points(make_cap([0.0, 1.0, 4.0], 0.15)) == []

    def test_small_clusters_are_rejected(self):
        detector = ClusterDetector(default_detector_config("rgbd"))  # min_points = 12
        few = np.array([0.0, 1.0, 1.0]) + np.random.default_rng(0).uniform(-0.01, 0.01, (8, 3))
        assert detector.detect_points(few) == []

    def test_nan_points_are_ignored(self, synthetic_cloud):
        detector = ClusterDetector(default_detector_config("rgbd", center_mode="sphere"))
        cloud = np.vstack([synthetic_cloud, np.full((5, 3), np.nan)])
        assert len(detector.detect_points(cloud)) == 2

    def test_detection_holds_the_cluster_points(self, synthetic_cloud):
        detector = ClusterDetector(default_detector_config("rgbd"))
        det = nearest(detector.detect_points(synthetic_cloud), CENTER_A)
        assert det.points.shape == (det.num_points, 3)
        assert det.features.num_points == det.num_points
        assert det.color is None

    def test_color_descriptor_is_attached(self, synthetic_cloud):
        colors = np.full((len(synthetic_cloud), 3), 128, dtype=np.uint8)
        n_floor = len(floor_grid())
        n_a = len(make_cap(CENTER_A, RADIUS_A))
        colors[n_floor:n_floor + n_a] = (200, 30, 30)  # sphere A red
        colors[n_floor + n_a:] = (30, 30, 200)  # sphere B blue
        detector = ClusterDetector(default_detector_config("rgbd", center_mode="sphere"))
        detections = detector.detect_points(synthetic_cloud, colors=colors)
        assert hue_difference_deg(nearest(detections, CENTER_A).color.hue_deg, 0.0) < 1.0
        assert hue_difference_deg(nearest(detections, CENTER_B).color.hue_deg, 240.0) < 1.0

    def test_colors_length_mismatch_raises(self, synthetic_cloud):
        detector = ClusterDetector(default_detector_config("rgbd"))
        with pytest.raises(ValueError):
            detector.detect_points(synthetic_cloud, colors=np.zeros((3, 3), dtype=np.uint8))

    def test_ambiguous_flag_for_oversized_clusters(self):
        big = make_cap([0.0, 1.0, 1.0], 0.2)  # about 0.39 m wide
        strict = ClusterDetector(default_detector_config("rgbd", max_extent=0.3))
        default = ClusterDetector(default_detector_config("rgbd"))
        assert strict.detect_points(big)[0].ambiguous is True
        assert default.detect_points(big)[0].ambiguous is False


class TestDetectorConfig:
    def test_default_configs(self):
        assert default_detector_config("rgbd").cluster.radius == pytest.approx(0.05)
        assert default_detector_config("LiDAR").sensor == "lidar"
        assert default_detector_config("laser").cluster.min_points == 3

    def test_overrides_are_applied_and_validated(self):
        cfg = default_detector_config("lidar", center_mode="sphere", max_extent=0.9)
        assert cfg.center_mode == "sphere" and cfg.max_extent == 0.9
        with pytest.raises(ValueError):
            default_detector_config("lidar", center_mode="median")

    def test_unknown_sensor_raises(self):
        with pytest.raises(ValueError):
            default_detector_config("sonar")

    def test_invalid_config_values_raise(self):
        cluster = ClusterConfig()
        with pytest.raises(ValueError):
            DetectorConfig(sensor="", cluster=cluster)
        with pytest.raises(ValueError):
            DetectorConfig(sensor="x", cluster=cluster, ground_margin=-1.0)
        with pytest.raises(ValueError):
            DetectorConfig(sensor="x", cluster=cluster, max_extent=0.0)
        with pytest.raises(ValueError):
            DetectorConfig(sensor="x", cluster=cluster, bounds_min=(1, 0, 0), bounds_max=(0, 1, 1))

    def test_invalid_transforms_and_stride_raise(self):
        intr = CameraIntrinsics.from_fovy(64, 48, 60.0)
        with pytest.raises(ValueError):
            RGBDDetector(intr, np.eye(3))
        with pytest.raises(ValueError):
            RGBDDetector(intr, np.eye(4), stride=0)
        with pytest.raises(ValueError):
            LidarDetector(np.eye(3))
        with pytest.raises(ValueError):
            LaserDetector(np.zeros((2, 2)))

    def test_rgbd_detect_rejects_mismatched_image_sizes(self):
        intr = CameraIntrinsics.from_fovy(64, 48, 60.0)
        detector = RGBDDetector(intr, np.eye(4))
        with pytest.raises(ValueError):
            detector.detect(np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((48, 64), dtype=np.float32))


# ------------------------------------------------- detectors on simulated sensors
class TestLidarDetector:
    @pytest.fixture
    def scene(self):
        spheres = [("S001", (-1.0, 0.0, 1.62), 0.2), ("S002", (1.0, 0.5, 1.2), 0.15)]
        env = make_env(spheres)
        lidar = Lidar(
            env,
            LidarConfig(azimuth_range_deg=(-45.0, 45.0), elevation_range_deg=(-25.0, 25.0), resolution_deg=0.5),
        )
        t_w_r = ground_truth_frame_tree(env).get_transform("W", "R")
        return env, lidar, t_w_r, spheres

    def test_detects_both_spheres_at_their_true_centers(self, scene):
        env, lidar, t_w_r, spheres = scene
        detector = LidarDetector(t_w_r, default_detector_config("lidar", center_mode="sphere"))
        scan = lidar.scan()
        detections = detector.detect(scan)
        assert len(detections) == 2
        for _, center, _ in spheres:
            det = nearest(detections, center)
            assert np.linalg.norm(det.position - np.asarray(center)) < 2e-3
            assert det.features.shape_hint == "sphere"
            assert det.sensor == "lidar" and det.time == scan.time
            assert det.color is None

    def test_centroid_mode_error_is_below_the_radius(self, scene):
        env, lidar, t_w_r, spheres = scene
        detector = LidarDetector(t_w_r, default_detector_config("lidar", center_mode="centroid"))
        detections = detector.detect(lidar.scan())
        for _, center, radius in spheres:
            assert np.linalg.norm(nearest(detections, center).position - np.asarray(center)) < radius

    def test_empty_scene_has_no_detections(self):
        env = Environment(obstacles=[])
        lidar = Lidar(env, LidarConfig(azimuth_range_deg=(-45.0, 45.0), elevation_range_deg=(-25.0, 25.0)))
        t_w_r = ground_truth_frame_tree(env).get_transform("W", "R")
        assert LidarDetector(t_w_r).detect(lidar.scan()) == []


class TestLaserDetector:
    def test_detects_a_sphere_at_its_true_center(self):
        center = (0.0, 0.0, 1.38)
        env = make_env([("S001", center, 0.2)])
        scanner = LaserScanner(env, LaserConfig(check_camera_visibility=False))
        scan = scanner.scan_sweep(np.arange(-15.0, 15.01, 1.0))
        t_w_l = ground_truth_frame_tree(env).get_transform("W", "L")
        detector = LaserDetector(t_w_l, default_detector_config("laser", center_mode="sphere"))
        detections = detector.detect(scan)
        assert len(detections) == 1
        assert np.linalg.norm(detections[0].position - np.asarray(center)) < 3e-3
        assert detections[0].sensor == "laser"


class TestRGBDDetector:
    @pytest.fixture
    def render_scene(self):
        spheres = [
            ("S001", (0.4, 0.0, 1.8), 0.2, (0.9, 0.1, 0.1, 1.0)),  # red
            ("S002", (-0.8, 0.8, 0.9), 0.15, (0.1, 0.2, 0.9, 1.0)),  # blue
        ]
        env = make_env(spheres)
        try:
            rgb = RGBCamera(env)
            depth = DepthCamera(env)
        except Exception as exc:  # no OpenGL context available
            pytest.skip(f"offscreen rendering unavailable: {exc}")
        yield env, rgb, depth, spheres
        rgb.close()
        depth.close()

    def test_detects_both_spheres_with_position_and_color(self, render_scene):
        env, rgb, depth, spheres = render_scene
        t_w_c = ground_truth_frame_tree(env).get_transform("W", "C")
        detector = RGBDDetector(
            depth.intrinsics, t_w_c, default_detector_config("rgbd", center_mode="sphere")
        )
        detections = detector.detect(rgb.capture(), depth.capture(), time=env.time)
        assert len(detections) == 2

        red = nearest(detections, spheres[0][1])
        blue = nearest(detections, spheres[1][1])
        assert red is not blue
        assert np.linalg.norm(red.position - np.asarray(spheres[0][1])) < 0.03
        assert np.linalg.norm(blue.position - np.asarray(spheres[1][1])) < 0.03
        assert hue_difference_deg(red.color.hue_deg, 0.0) < 30.0
        assert hue_difference_deg(blue.color.hue_deg, 232.5) < 30.0
        assert all(d.sensor == "rgbd" for d in detections)

    def test_detection_without_rgb_has_no_color(self, render_scene):
        env, _, depth, _ = render_scene
        t_w_c = ground_truth_frame_tree(env).get_transform("W", "C")
        detector = RGBDDetector(depth.intrinsics, t_w_c)
        detections = detector.detect(None, depth.capture())
        assert len(detections) == 2
        assert all(d.color is None for d in detections)