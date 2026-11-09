"""Unit tests for the sensors package."""

import mujoco
import numpy as np
import pytest

from sensors.common import (
    FORWARD_Y_MOUNT,
    MUJOCO_TO_OPENCV,
    RayCaster,
    angle_grid_deg,
    spherical_to_cartesian,
    transform_points,
)
from sensors.depth_camera import DepthCamera, depth_to_points
from sensors.laser_scanner import LaserConfig, LaserScanner
from sensors.lidar import Lidar, LidarConfig
from sensors.rgb_camera import (
    CameraIntrinsics,
    RGBCamera,
    ground_truth_camera_pose_opencv,
)
from simulation.environment import Environment
from simulation.motion_generator import LinearBounceTrajectory
from simulation.obstacle_manager import ObstacleSpec

SPHERE_RADIUS = 0.2
SPHERE_CENTER = np.array([0.4, 0.0, 1.8])  # off-axis target for the camera tests


# ------------------------------------------------------------------ helpers
def make_env(spheres):
    """Environment with stationary spheres: spheres = [(id, center, radius), ...]."""
    specs = []
    for oid, center, radius in spheres:
        traj = LinearBounceTrajectory(center, (0.0, 0.0, 0.0), center, center)
        specs.append(
            ObstacleSpec(oid, "sphere", (radius,), "3D", traj, rgba=(0.9, 0.1, 0.1, 1.0))
        )
    return Environment(obstacles=specs)


def geom_names(env, geom_ids):
    return [
        mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) for g in geom_ids
    ]


def project_world_point(env, intrinsics, point):
    """Pixel [u, v] of a world point using the ground-truth camera pose."""
    position, rotation = ground_truth_camera_pose_opencv(env)
    p_cam = (np.asarray(point, dtype=float) - position) @ rotation
    return intrinsics.project(p_cam)[0]


def is_red(pixel):
    r, g, b = (int(x) for x in pixel)
    return r > b + 30 and r > g + 30


# ----------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def default_env():
    return Environment()


@pytest.fixture(scope="module")
def lidar_env():
    # Sphere at the LiDAR site height (1.5 + 0.12 = 1.62 m).
    return make_env([("S001", (0.0, 0.0, 1.62), SPHERE_RADIUS)])


@pytest.fixture(scope="module")
def laser_env():
    # Sphere at the laser site height (1.5 - 0.12 = 1.38 m).
    return make_env([("S001", (0.0, 0.0, 1.38), SPHERE_RADIUS)])


@pytest.fixture(scope="module")
def render_env():
    env = make_env([("S001", tuple(SPHERE_CENTER), SPHERE_RADIUS)])
    try:
        rgb = RGBCamera(env)
        depth = DepthCamera(env)
    except Exception as exc:  # no OpenGL context available
        pytest.skip(f"offscreen rendering unavailable: {exc}")
    yield env, rgb, depth
    rgb.close()
    depth.close()


# ------------------------------------------------------------------- common
class TestCommon:
    def test_angle_grid_includes_endpoints(self):
        grid = angle_grid_deg(-90.0, 90.0, 1.0)
        assert len(grid) == 181
        assert grid[0] == -90.0 and grid[-1] == 90.0

    def test_angle_grid_single_value(self):
        assert np.allclose(angle_grid_deg(0.0, 0.0, 1.0), [0.0])

    def test_angle_grid_invalid(self):
        with pytest.raises(ValueError):
            angle_grid_deg(0.0, 1.0, 0.0)
        with pytest.raises(ValueError):
            angle_grid_deg(1.0, 0.0, 0.5)

    def test_spherical_to_cartesian_axes(self):
        assert np.allclose(spherical_to_cartesian(2.0, 0.0, 0.0), [2.0, 0.0, 0.0])
        assert np.allclose(spherical_to_cartesian(2.0, np.pi / 2, 0.0), [0.0, 2.0, 0.0], atol=1e-12)
        assert np.allclose(spherical_to_cartesian(2.0, 0.0, np.pi / 2), [0.0, 0.0, 2.0], atol=1e-12)

    def test_opencv_conversion_is_an_involution(self):
        assert np.allclose(MUJOCO_TO_OPENCV @ MUJOCO_TO_OPENCV, np.eye(3))

    def test_forward_mount_is_a_proper_rotation(self):
        assert np.allclose(FORWARD_Y_MOUNT.T @ FORWARD_Y_MOUNT, np.eye(3))
        assert np.linalg.det(FORWARD_Y_MOUNT) == pytest.approx(1.0)
        assert np.allclose(FORWARD_Y_MOUNT @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])

    def test_transform_points(self):
        rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # +90 deg about z
        out = transform_points([[1.0, 0.0, 0.0]], rot, [1.0, 2.0, 3.0])
        assert np.allclose(out, [[1.0, 3.0, 3.0]])


class TestRayCaster:
    def test_hits_sphere_at_expected_distance(self, laser_env):
        caster = RayCaster(laser_env.model, laser_env.data)
        dist, gid = caster.cast([0.0, -3.0, 1.38], [0.0, 1.0, 0.0])
        assert dist == pytest.approx(2.8, abs=1e-6)
        assert geom_names(laser_env, [gid]) == ["S001_geom"]

    def test_miss_returns_minus_one(self, laser_env):
        caster = RayCaster(laser_env.model, laser_env.data)
        dist, gid = caster.cast([0.0, -3.0, 1.38], [0.0, 0.0, 1.0])
        assert dist == -1.0
        assert gid == -1

    def test_hits_floor(self, laser_env):
        caster = RayCaster(laser_env.model, laser_env.data)
        dist, gid = caster.cast([0.0, 0.0, 1.0], [0.0, 0.0, -1.0])
        assert dist == pytest.approx(1.0, abs=1e-6)
        assert geom_names(laser_env, [gid]) == ["floor"]

    def test_cast_many_matches_cast(self, laser_env):
        caster = RayCaster(laser_env.model, laser_env.data)
        origin = np.array([0.0, -3.0, 1.38])
        dirs = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        dist, gid = caster.cast_many(origin, dirs)
        assert dist[0] == pytest.approx(2.8, abs=1e-6)
        assert dist[1] == -1.0
        assert gid[1] == -1


# ---------------------------------------------------------------- intrinsics
class TestCameraIntrinsics:
    def test_from_fovy(self):
        intr = CameraIntrinsics.from_fovy(640, 480, 60.0)
        assert intr.fy == pytest.approx(240.0 / np.tan(np.radians(30.0)))
        assert intr.fx == pytest.approx(intr.fy)
        assert intr.cx == pytest.approx(319.5)
        assert intr.cy == pytest.approx(239.5)
        assert intr.matrix.shape == (3, 3)
        assert intr.matrix[2, 2] == 1.0

    def test_project_deproject_roundtrip(self):
        intr = CameraIntrinsics.from_fovy(640, 480, 60.0)
        rng = np.random.default_rng(0)
        pts = np.column_stack(
            [rng.uniform(-1, 1, 50), rng.uniform(-1, 1, 50), rng.uniform(1, 5, 50)]
        )
        uv = intr.project(pts)
        assert np.allclose(intr.deproject(uv, pts[:, 2]), pts)

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            CameraIntrinsics.from_fovy(0, 480, 60.0)
        with pytest.raises(ValueError):
            CameraIntrinsics.from_fovy(640, 480, 180.0)

    def test_ground_truth_pose_axes_and_projection(self, default_env):
        position, rotation = ground_truth_camera_pose_opencv(default_env)
        assert np.allclose(position, [0.0, -3.0, 1.5])
        assert np.allclose(rotation[:, 0], [1.0, 0.0, 0.0])  # image right -> world +X
        assert np.allclose(rotation[:, 1], [0.0, 0.0, -1.0])  # image down -> world -Z
        assert np.allclose(rotation[:, 2], [0.0, 1.0, 0.0])  # forward -> world +Y
        assert np.linalg.det(rotation) == pytest.approx(1.0)

        intr = CameraIntrinsics.from_fovy(640, 480, 60.0)
        straight = project_world_point(default_env, intr, [0.0, 0.0, 1.5])
        assert np.allclose(straight, [intr.cx, intr.cy])
        right = project_world_point(default_env, intr, [0.5, 0.0, 1.5])
        assert right[0] == pytest.approx(intr.fx * 0.5 / 3.0 + intr.cx)
        above = project_world_point(default_env, intr, [0.0, 0.0, 2.0])
        assert above[1] == pytest.approx(intr.cy - intr.fy * 0.5 / 3.0)  # smaller v = higher


# ---------------------------------------------------------------- RGB camera
class TestRGBCamera:
    def test_capture_shape_and_dtype(self, render_env):
        _, rgb, _ = render_env
        image = rgb.capture()
        assert image.shape == (480, 640, 3)
        assert image.dtype == np.uint8

    def test_target_appears_at_projected_pixel_with_correct_orientation(self, render_env):
        env, rgb, _ = render_env
        image = rgb.capture()
        u, v = project_world_point(env, rgb.intrinsics, SPHERE_CENTER)
        row, col = int(round(v)), int(round(u))
        assert is_red(image[row, col])
        # Mirrored positions (flipped vertically / horizontally) must not be red.
        assert not is_red(image[int(round(479 - v)), col])
        assert not is_red(image[row, int(round(639 - u))])

    def test_static_scene_gives_identical_frames(self, render_env):
        _, rgb, _ = render_env
        assert np.array_equal(rgb.capture(), rgb.capture())

    def test_noise_changes_image_and_is_seeded(self, render_env):
        env, rgb, _ = render_env
        clean = rgb.capture()
        with RGBCamera(env, noise_std=8.0, seed=1) as cam:
            first = cam.capture()
        with RGBCamera(env, noise_std=8.0, seed=1) as cam:
            second = cam.capture()
        assert not np.array_equal(first, clean)
        assert np.array_equal(first, second)

    def test_moving_obstacles_change_the_image(self, render_env):
        env = Environment()
        with RGBCamera(env) as cam:
            before = cam.capture()
            for _ in range(500):
                env.step()
            after = cam.capture()
        assert not np.array_equal(before, after)

    def test_negative_noise_raises(self, default_env):
        with pytest.raises(ValueError):
            RGBCamera(default_env, noise_std=-1.0)


# -------------------------------------------------------------- depth camera
class TestDepthCamera:
    def test_capture_shape_and_dtype(self, render_env):
        _, _, depth = render_env
        img = depth.capture()
        assert img.shape == (480, 640)
        assert img.dtype == np.float32

    def test_background_is_nan(self, render_env):
        _, _, depth = render_env
        assert np.isnan(depth.capture()[0, 0])

    def test_depth_at_target_matches_geometry(self, render_env):
        env, _, depth = render_env
        img = depth.capture()
        u, v = project_world_point(env, depth.intrinsics, SPHERE_CENTER)
        row, col = int(round(v)), int(round(u))

        position, _ = env.get_camera_pose()
        to_center = SPHERE_CENTER - position
        dist = np.linalg.norm(to_center)
        # z-depth of the sphere surface along the ray through its center.
        expected = (dist - SPHERE_RADIUS) * (3.0 / dist)  # camera-to-sphere offset along y is 3 m
        assert img[row, col] == pytest.approx(expected, abs=0.01)

    def test_point_cloud_lies_on_sphere_surface(self, render_env):
        env, _, depth = render_env
        img = depth.capture()
        u, v = project_world_point(env, depth.intrinsics, SPHERE_CENTER)
        row, col = int(round(v)), int(round(u))
        window = (slice(row - 12, row + 13), slice(col - 12, col + 13))

        points = depth_to_points(img, depth.intrinsics)[window].reshape(-1, 3)
        position, rotation = ground_truth_camera_pose_opencv(env)
        world = transform_points(points, rotation, position)
        radial = np.linalg.norm(world - SPHERE_CENTER, axis=1)
        assert np.all(np.isfinite(radial))
        assert np.allclose(radial, SPHERE_RADIUS, atol=0.01)

    def test_capture_point_cloud_returns_valid_points(self, render_env):
        _, _, depth = render_env
        cloud = depth.capture_point_cloud()
        assert cloud.ndim == 2 and cloud.shape[1] == 3
        assert len(cloud) > 0
        assert np.all(np.isfinite(cloud))

    def test_depth_to_points_rejects_wrong_size(self):
        intr = CameraIntrinsics.from_fovy(640, 480, 60.0)
        with pytest.raises(ValueError):
            depth_to_points(np.zeros((10, 10), dtype=np.float32), intr)

    def test_noise_has_expected_spread(self, render_env):
        env, _, depth = render_env
        clean = depth.capture()
        with DepthCamera(env, noise_std=0.01, seed=3) as noisy_cam:
            noisy = noisy_cam.capture()
        diff = noisy - clean
        assert np.sum(np.isfinite(diff)) > 1000
        assert np.nanstd(diff) == pytest.approx(0.01, abs=0.002)

    def test_dropout_removes_all_pixels(self, render_env):
        env, _, _ = render_env
        with DepthCamera(env, dropout_prob=1.0) as cam:
            assert np.all(np.isnan(cam.capture()))

    def test_invalid_configuration_raises(self, default_env):
        with pytest.raises(ValueError):
            DepthCamera(default_env, min_depth=5.0, max_depth=1.0)
        with pytest.raises(ValueError):
            DepthCamera(default_env, dropout_prob=1.5)


# --------------------------------------------------------------------- LiDAR
class TestLidar:
    def test_ground_truth_pose_axes(self, default_env):
        position, rotation = Lidar(default_env).ground_truth_pose()
        assert np.allclose(position, [0.0, -3.0, 1.62])
        assert np.allclose(rotation @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])  # forward -> world +Y
        assert np.allclose(rotation @ [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0])  # left -> world -X
        assert np.allclose(rotation @ [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])  # up -> world +Z

    def test_default_scan_grid_shape(self, default_env):
        lidar = Lidar(default_env)
        scan = lidar.scan()
        assert scan.ranges.shape == (61, 181)
        assert scan.geom_ids.shape == (61, 181)
        assert lidar.num_beams == 61 * 181

    def test_center_beam_range_matches_geometry(self, lidar_env):
        cfg = LidarConfig(azimuth_range_deg=(-2, 2), elevation_range_deg=(-2, 2), resolution_deg=1.0)
        scan = Lidar(lidar_env, cfg).scan()
        assert scan.ranges[2, 2] == pytest.approx(2.8, abs=1e-6)

    def test_points_lie_on_sphere_in_world_frame(self, lidar_env):
        cfg = LidarConfig(azimuth_range_deg=(-2, 2), elevation_range_deg=(-2, 2), resolution_deg=0.5)
        lidar = Lidar(lidar_env, cfg)
        scan = lidar.scan()
        assert scan.valid.all()  # sphere subtends about +-3.8 degrees, so every beam hits it
        position, rotation = lidar.ground_truth_pose()
        world = transform_points(scan.points(), rotation, position)
        center = np.array([0.0, 0.0, 1.62])
        assert np.allclose(np.linalg.norm(world - center, axis=1), SPHERE_RADIUS, atol=1e-6)
        assert set(geom_names(lidar_env, scan.geom_ids[scan.valid])) == {"S001_geom"}

    def test_polar_and_cartesian_are_consistent(self, lidar_env):
        scan = Lidar(lidar_env, LidarConfig(azimuth_range_deg=(-2, 2), elevation_range_deg=(-2, 2))).scan()
        polar = scan.polar()
        assert np.allclose(spherical_to_cartesian(polar[:, 0], polar[:, 1], polar[:, 2]), scan.points())

    def test_scan_of_default_scene_hits_obstacles_within_limits(self, default_env):
        default_env.reset()
        lidar = Lidar(default_env)
        scan = lidar.scan()
        valid_ranges = scan.ranges[scan.valid]
        assert len(valid_ranges) > 0
        assert np.all(valid_ranges >= lidar.config.min_range)
        assert np.all(valid_ranges <= lidar.config.max_range)
        names = set(geom_names(default_env, np.unique(scan.geom_ids[scan.valid])))
        assert any(name.startswith("O00") for name in names)

    def test_noise_is_seeded_and_has_expected_spread(self, lidar_env):
        clean = Lidar(lidar_env, LidarConfig()).scan().ranges
        noisy_a = Lidar(lidar_env, LidarConfig(range_noise_std=0.02), seed=5).scan().ranges
        noisy_b = Lidar(lidar_env, LidarConfig(range_noise_std=0.02), seed=5).scan().ranges
        noisy_c = Lidar(lidar_env, LidarConfig(range_noise_std=0.02), seed=6).scan().ranges
        assert np.array_equal(noisy_a, noisy_b, equal_nan=True)
        assert not np.array_equal(noisy_a, noisy_c, equal_nan=True)
        diff = noisy_a - clean
        assert np.sum(np.isfinite(diff)) > 50
        assert 0.01 < np.nanstd(diff) < 0.04

    def test_dropout_removes_all_returns(self, lidar_env):
        cfg = LidarConfig(azimuth_range_deg=(-2, 2), elevation_range_deg=(-2, 2), dropout_prob=1.0)
        scan = Lidar(lidar_env, cfg).scan()
        assert not scan.valid.any()
        assert np.all(scan.geom_ids == -1)

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError):
            LidarConfig(resolution_deg=0.0)
        with pytest.raises(ValueError):
            LidarConfig(min_range=5.0, max_range=1.0)
        with pytest.raises(ValueError):
            LidarConfig(azimuth_range_deg=(10.0, -10.0))
        with pytest.raises(ValueError):
            LidarConfig(elevation_range_deg=(-100.0, 10.0))
        with pytest.raises(ValueError):
            LidarConfig(dropout_prob=2.0)


# ------------------------------------------------------------ laser scanner
SINGLE_LINE = dict(
    fan_range_deg=(-1.0, 1.0),
    fan_step_deg=1.0,
    sweep_range_deg=(0.0, 0.0),
    sweep_step_deg=1.0,
)


class TestLaserScanner:
    def test_ground_truth_pose_axes(self, default_env):
        position, rotation = LaserScanner(default_env).ground_truth_pose()
        assert np.allclose(position, [0.0, -3.0, 1.38])
        assert np.allclose(rotation @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])

    def test_default_sweep_shape(self, default_env):
        scanner = LaserScanner(default_env, LaserConfig(check_camera_visibility=False))
        scan = scanner.scan_sweep()
        assert scan.ranges.shape == (61, 61)
        assert len(scan.yaw) == 61 and len(scan.fan) == 61

    def test_center_spot_range_matches_geometry(self, laser_env):
        scan = LaserScanner(laser_env, LaserConfig(**SINGLE_LINE)).scan_line(0.0)
        assert scan.ranges.shape == (1, 3)
        assert scan.ranges[0, 1] == pytest.approx(2.8, abs=1e-6)

    def test_spots_lie_on_the_laser_plane(self, default_env):
        scanner = LaserScanner(default_env, LaserConfig(check_camera_visibility=False))
        scan = scanner.scan_line(20.0)
        points_laser = scan.points()
        assert len(points_laser) > 0
        position, rotation = scanner.ground_truth_pose()
        points_world = transform_points(points_laser, rotation, position)
        normal, point = scanner.plane_world(20.0)
        assert np.linalg.norm(normal) == pytest.approx(1.0)
        assert np.all(np.abs((points_world - point) @ normal) < 1e-6)

    def test_plane_normal_at_zero_yaw_is_perpendicular_to_forward_and_up(self, default_env):
        normal, _ = LaserScanner(default_env).plane_world(0.0)
        assert np.allclose(normal, [-1.0, 0.0, 0.0])

    def test_visibility_check_drops_spot_hidden_from_camera(self):
        # Target A is lit by the laser; a small sphere B sits right in front of the
        # camera and blocks its view of A, but not the laser ray (B is 0.12 m higher).
        env = make_env(
            [
                ("S001", (0.0, 0.0, 1.38), SPHERE_RADIUS),
                ("S002", (0.0, -2.0, 1.5), 0.05),
            ]
        )
        without_check = LaserScanner(env, LaserConfig(check_camera_visibility=False, **SINGLE_LINE))
        with_check = LaserScanner(env, LaserConfig(check_camera_visibility=True, **SINGLE_LINE))
        assert without_check.scan_line(0.0).ranges[0, 1] == pytest.approx(2.8, abs=1e-6)
        assert np.isnan(with_check.scan_line(0.0).ranges[0, 1])

    def test_noise_is_seeded_and_has_expected_spread(self, default_env):
        base = dict(check_camera_visibility=False)
        clean = LaserScanner(default_env, LaserConfig(**base)).scan_sweep().ranges
        noisy_a = LaserScanner(default_env, LaserConfig(noise_std=0.01, **base), seed=3).scan_sweep().ranges
        noisy_b = LaserScanner(default_env, LaserConfig(noise_std=0.01, **base), seed=3).scan_sweep().ranges
        assert np.array_equal(noisy_a, noisy_b, equal_nan=True)
        diff = noisy_a - clean
        assert np.sum(np.isfinite(diff)) > 50
        assert 0.005 < np.nanstd(diff) < 0.02

    def test_distance_dependent_noise_grows_with_range(self, default_env):
        base = dict(check_camera_visibility=False)
        clean = LaserScanner(default_env, LaserConfig(**base)).scan_sweep().ranges
        cfg = LaserConfig(noise_std_per_m2=0.002, **base)
        noisy = LaserScanner(default_env, cfg, seed=4).scan_sweep().ranges
        diff = np.abs(noisy - clean)
        near = np.isfinite(diff) & (clean < 3.0)
        far = np.isfinite(diff) & (clean >= 3.5)
        assert near.sum() > 20 and far.sum() > 20
        assert diff[far].mean() > diff[near].mean()

    def test_dropout_removes_all_spots(self, laser_env):
        scan = LaserScanner(laser_env, LaserConfig(dropout_prob=1.0, **SINGLE_LINE)).scan_line(0.0)
        assert not scan.valid.any()
        assert np.all(scan.geom_ids == -1)

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError):
            LaserConfig(fan_step_deg=0.0)
        with pytest.raises(ValueError):
            LaserConfig(min_range=2.0, max_range=1.0)
        with pytest.raises(ValueError):
            LaserConfig(sweep_range_deg=(10.0, -10.0))
        with pytest.raises(ValueError):
            LaserConfig(noise_std=-0.1)