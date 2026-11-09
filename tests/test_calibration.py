"""Unit tests for the calibration package."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibration.camera_intrinsic import (
    calibrate_camera,
    capture_board_corners,
    collect_board_corners,
    evaluate_intrinsics,
    observe_board,
    project_board_corners,
)
from calibration.coordinate_frames import (
    FrameTree,
    Plane,
    apply_transform,
    board_plane,
    compose_transforms,
    fit_plane,
    ground_truth_frame_tree,
    invert_transform,
    make_transform,
    rotation_angle_deg,
    split_transform,
    transform_error,
)
from calibration.laser_plane import (
    LaserView,
    calibrate_laser_plane,
    collect_laser_views,
    evaluate_laser_plane,
    ground_truth_laser_plane,
    pixels_to_rays,
    triangulate_laser_pixels,
)
from calibration.lidar_extrinsic import (
    PlaneCorrespondence,
    calibrate_lidar_extrinsic,
    collect_lidar_views,
    estimate_extrinsic_closed_form,
    evaluate_extrinsic,
    point_to_plane_residuals,
)
from calibration.target_scene import (
    CalibrationScene,
    checkerboard_object_points,
    sample_board_poses,
)
from sensors.laser_scanner import LaserConfig, LaserScanner
from sensors.lidar import Lidar, LidarConfig
from sensors.rgb_camera import CameraIntrinsics, RGBCamera
from simulation.environment import Environment, EnvironmentConfig

TRUTH = CameraIntrinsics(fx=450.0, fy=455.0, cx=322.0, cy=238.0, width=640, height=480)
PATTERN = (8, 5)
SQUARE = 0.10
HALF_EXTENT = (0.55, 0.40)  # outer board half size for pattern (8, 5) and 0.10 m squares
LASER_OFFSET = (0.25, 0.0, -0.12)


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def default_env():
    return Environment()


@pytest.fixture(scope="module")
def scene():
    return CalibrationScene(EnvironmentConfig(laser_offset=LASER_OFFSET))


@pytest.fixture(scope="module")
def rig(scene):
    try:
        camera = RGBCamera(scene)
    except Exception as exc:  # no OpenGL context available
        pytest.skip(f"offscreen rendering unavailable: {exc}")
    yield scene, camera
    camera.close()


# ---------------------------------------------------------- coordinate frames
class TestTransforms:
    def test_make_split_roundtrip(self):
        rot = Rotation.from_rotvec([0.2, -0.1, 0.4]).as_matrix()
        transform = make_transform(rot, [1.0, 2.0, 3.0])
        rot2, trans2 = split_transform(transform)
        assert np.allclose(rot2, rot) and np.allclose(trans2, [1.0, 2.0, 3.0])
        assert np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0])

    def test_invert_and_compose(self):
        transform = make_transform(Rotation.from_rotvec([0.3, 0.2, -0.5]).as_matrix(), [0.5, -1.0, 2.0])
        assert np.allclose(compose_transforms(transform, invert_transform(transform)), np.eye(4))
        assert np.allclose(compose_transforms(), np.eye(4))

    def test_apply_transform(self):
        rot = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
        out = apply_transform(make_transform(rot, [1.0, 2.0, 3.0]), [[1.0, 0.0, 0.0]])
        assert np.allclose(out, [[1.0, 3.0, 3.0]])

    def test_rotation_angle(self):
        rot = Rotation.from_euler("z", 30.0, degrees=True).as_matrix()
        assert rotation_angle_deg(np.eye(3), rot) == pytest.approx(30.0)
        assert rotation_angle_deg(rot, rot) == pytest.approx(0.0, abs=1e-6)

    def test_transform_error(self):
        a = make_transform(np.eye(3), [0.0, 0.0, 0.0])
        b = make_transform(Rotation.from_euler("x", 5.0, degrees=True).as_matrix(), [0.0, 0.3, 0.4])
        rot_err, trans_err = transform_error(a, b)
        assert rot_err == pytest.approx(5.0)
        assert trans_err == pytest.approx(0.5)

    def test_invalid_shapes_raise(self):
        with pytest.raises(ValueError):
            make_transform(np.eye(2), [0, 0, 0])
        with pytest.raises(ValueError):
            split_transform(np.eye(3))


class TestFrameTree:
    def test_chain_and_inverse(self):
        t_ab = make_transform(Rotation.from_euler("z", 40.0, degrees=True).as_matrix(), [1.0, 0.0, 0.0])
        t_bc = make_transform(Rotation.from_euler("x", -20.0, degrees=True).as_matrix(), [0.0, 2.0, 0.5])
        tree = FrameTree()
        tree.add_transform("A", "B", t_ab)
        tree.add_transform("B", "C", t_bc)
        assert np.allclose(tree.get_transform("A", "C"), t_ab @ t_bc)
        assert np.allclose(tree.get_transform("C", "A"), invert_transform(t_ab @ t_bc))
        assert np.allclose(tree.get_transform("B", "B"), np.eye(4))
        assert tree.frames == ["A", "B", "C"]

    def test_transform_points(self):
        tree = FrameTree()
        tree.add_transform("A", "B", make_transform(np.eye(3), [1.0, 2.0, 3.0]))
        assert np.allclose(tree.transform_points([[0.0, 0.0, 0.0]], "A", "B"), [[1.0, 2.0, 3.0]])

    def test_unknown_and_disconnected_frames(self):
        tree = FrameTree()
        tree.add_transform("A", "B", np.eye(4))
        tree.add_transform("C", "D", np.eye(4))
        with pytest.raises(KeyError):
            tree.get_transform("A", "Z")
        with pytest.raises(ValueError):
            tree.get_transform("A", "C")

    def test_same_parent_and_child_raises(self):
        with pytest.raises(ValueError):
            FrameTree().add_transform("A", "A", np.eye(4))


class TestGroundTruthFrameTree:
    def test_default_environment_frames(self, default_env):
        default_env.reset()
        tree = ground_truth_frame_tree(default_env)
        assert set(tree.frames) == {"W", "B", "C", "L", "R"}
        t_w_b = tree.get_transform("W", "B")
        t_b_c = tree.get_transform("B", "C")
        assert np.allclose(t_w_b[:3, 3], [0.0, -3.0, 1.5])
        assert np.allclose(tree.get_transform("W", "C"), t_w_b @ t_b_c)

    def test_lidar_in_camera_frame(self, default_env):
        t_c_r = ground_truth_frame_tree(default_env).get_transform("C", "R")
        assert np.allclose(t_c_r[:3, 3], [0.0, -0.12, 0.0])  # LiDAR sits 12 cm above the camera
        rot = t_c_r[:3, :3]
        assert np.allclose(rot @ [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])  # LiDAR forward -> camera +z
        assert np.allclose(rot @ [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0])  # LiDAR left -> camera -x
        assert np.allclose(rot @ [0.0, 0.0, 1.0], [0.0, -1.0, 0.0])  # LiDAR up -> camera -y

    def test_laser_offset_appears_in_frame_tree(self):
        env = Environment(EnvironmentConfig(laser_offset=LASER_OFFSET))
        t_c_l = ground_truth_frame_tree(env).get_transform("C", "L")
        assert np.allclose(t_c_l[:3, 3], [0.25, 0.12, 0.0])

    def test_default_laser_has_no_baseline(self, default_env):
        t_c_l = ground_truth_frame_tree(default_env).get_transform("C", "L")
        assert np.allclose(t_c_l[:3, 3], [0.0, 0.12, 0.0])


# --------------------------------------------------------------------- planes
class TestPlane:
    def test_normalization_and_facing_origin(self):
        plane = Plane((0.0, 0.0, 2.0), -4.0)  # z = 2 -> normal (0,0,1), d = -2
        assert np.allclose(plane.normal, [0.0, 0.0, 1.0])
        assert plane.d == pytest.approx(-2.0)
        facing = plane.facing_origin()
        assert np.allclose(facing.normal, [0.0, 0.0, -1.0])
        assert facing.d == pytest.approx(2.0)

    def test_zero_normal_raises(self):
        with pytest.raises(ValueError):
            Plane((0.0, 0.0, 0.0), 1.0)

    def test_signed_distance(self):
        plane = Plane((0.0, 0.0, -1.0), 2.0)  # the plane z = 2
        assert np.allclose(plane.signed_distance([[0, 0, 2], [0, 0, 3]]), [0.0, -1.0])

    def test_transformed_keeps_points_on_plane(self):
        plane = Plane((0.3, -0.5, -1.0), 1.7)
        points = np.array([[0.0, 0.0, 1.7], [1.0, 0.0, 1.7 + 0.3 / 1.0]])
        # Build points exactly on the plane.
        xy = np.array([[0.2, 0.1], [-0.4, 0.5], [0.7, -0.3]])
        z = -(plane.normal[0] * xy[:, 0] + plane.normal[1] * xy[:, 1] + plane.d) / plane.normal[2]
        points = np.column_stack([xy, z])
        assert np.allclose(plane.signed_distance(points), 0.0, atol=1e-12)
        transform = make_transform(Rotation.from_rotvec([0.2, 0.3, -0.1]).as_matrix(), [0.5, -0.2, 0.1])
        moved = plane.transformed(transform)
        assert np.allclose(moved.signed_distance(apply_transform(transform, points)), 0.0, atol=1e-12)

    def test_angle_to(self):
        a = Plane((0.0, 0.0, 1.0), 1.0)
        b = Plane((np.sin(np.radians(7.0)), 0.0, np.cos(np.radians(7.0))), 1.0)
        assert a.angle_to(b) == pytest.approx(7.0)
        assert a.angle_to(Plane((0.0, 0.0, -1.0), -1.0)) == pytest.approx(0.0, abs=1e-6)

    def test_intersect_rays(self):
        plane = Plane((0.0, 0.0, -1.0), 2.0)  # z = 2
        out = plane.intersect_rays([[0.0, 0.0, 1.0], [0.5, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        assert np.allclose(out[0], [0.0, 0.0, 2.0])
        assert np.allclose(out[1], [1.0, 0.0, 2.0])
        assert np.all(np.isnan(out[2]))  # parallel to the plane
        assert np.all(np.isnan(out[3]))  # intersection behind the origin

    def test_fit_plane_exact_and_facing_origin(self):
        rng = np.random.default_rng(0)
        true = Plane((0.2, -0.3, -1.0), 1.5)
        xy = rng.uniform(-1, 1, (200, 2))
        z = -(true.normal[0] * xy[:, 0] + true.normal[1] * xy[:, 1] + true.d) / true.normal[2]
        plane = fit_plane(np.column_stack([xy, z]))
        assert plane.angle_to(true) < 1e-6
        assert plane.d == pytest.approx(true.d, abs=1e-9)
        assert plane.d >= 0.0

    def test_fit_plane_robust_rejects_outliers(self):
        rng = np.random.default_rng(1)
        xy = rng.uniform(-1, 1, (300, 2))
        inliers = np.column_stack([xy, np.full(300, 1.0)])  # the plane z = 1
        outliers = np.column_stack([rng.uniform(-1, 1, (15, 2)), 1.0 + rng.uniform(0.2, 0.4, 15)])
        points = np.vstack([inliers, outliers])
        truth = Plane((0.0, 0.0, -1.0), 1.0)
        plain = fit_plane(points)
        robust = fit_plane(points, robust_iterations=3)
        assert robust.angle_to(truth) < 1e-3
        assert robust.d == pytest.approx(1.0, abs=1e-6)
        assert plain.angle_to(truth) > robust.angle_to(truth)

    def test_fit_plane_needs_three_points(self):
        with pytest.raises(ValueError):
            fit_plane([[0, 0, 0], [1, 0, 0]])

    def test_board_plane(self):
        plane = board_plane(np.eye(3), [0.0, 0.0, 2.0])
        assert np.allclose(plane.normal, [0.0, 0.0, -1.0])
        assert plane.d == pytest.approx(2.0)
        assert np.allclose(plane.signed_distance([[0.3, -0.2, 2.0]]), 0.0)


# ----------------------------------------------------------- calibration scene
class TestCalibrationScene:
    def test_scene_has_board_mocap_and_geoms(self, scene):
        cols, rows = scene.pattern_size[0] + 1, scene.pattern_size[1] + 1
        dark = sum((a + b) % 2 == 0 for a in range(cols) for b in range(rows))
        assert scene.model.nmocap == 1
        assert len(scene.board_geom_ids) == 1 + dark

    def test_board_half_extent(self, scene):
        assert scene.half_extent == pytest.approx(HALF_EXTENT)

    def test_set_board_pose_places_board_in_world(self, scene):
        scene.set_board_pose(np.eye(3), [0.0, 0.0, 1.5])
        body_pos = scene.data.xpos[scene._board_body_id]
        assert np.allclose(body_pos, [0.0, -1.5, 1.5], atol=1e-9)  # camera at (0,-3,1.5) looking +Y
        rot, trans = scene.board_pose
        assert np.allclose(rot, np.eye(3)) and np.allclose(trans, [0.0, 0.0, 1.5])

    def test_reset_restores_board_pose(self, scene):
        pose = (Rotation.from_euler("x", 15.0, degrees=True).as_matrix(), np.array([0.1, 0.0, 1.6]))
        scene.set_board_pose(*pose)
        before = scene.data.xpos[scene._board_body_id].copy()
        scene.reset()
        assert np.allclose(scene.data.xpos[scene._board_body_id], before)

    def test_object_points(self):
        pts = checkerboard_object_points((8, 5), 0.1)
        assert pts.shape == (40, 3)
        assert np.allclose(pts.mean(axis=0), 0.0, atol=1e-12)
        assert np.allclose(pts[1] - pts[0], [0.1, 0.0, 0.0])  # x varies fastest
        assert np.allclose(pts[8] - pts[0], [0.0, 0.1, 0.0])
        assert np.all(pts[:, 2] == 0.0)

    def test_object_points_invalid(self):
        with pytest.raises(ValueError):
            checkerboard_object_points((1, 5), 0.1)
        with pytest.raises(ValueError):
            checkerboard_object_points((8, 5), 0.0)

    def test_sample_board_poses_are_valid(self):
        rng = np.random.default_rng(0)
        poses = sample_board_poses(TRUTH, HALF_EXTENT, 30, rng)
        assert len(poses) == 30
        outline = np.array([[-0.55, -0.4, 0], [0.55, -0.4, 0], [0.55, 0.4, 0], [-0.55, 0.4, 0]])
        for rot, trans in poses:
            assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
            tilt = np.degrees(np.arccos(rot[2, 2]))
            assert 10.0 - 1e-6 <= tilt <= 30.0 + 1e-6
            corners = outline @ rot.T + trans
            assert np.all(corners[:, 2] > 0.2)
            uv = TRUTH.project(corners)
            assert np.all(uv >= 0.0)
            assert np.all(uv[:, 0] <= 639) and np.all(uv[:, 1] <= 479)

    def test_sample_board_poses_impossible_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(RuntimeError):
            sample_board_poses(TRUTH, HALF_EXTENT, 5, rng, depth_range=(0.3, 0.4), max_attempts=200)


# --------------------------------------------------- synthetic camera calibration
def synthetic_views(n_views, noise, distortion=None, seed=0):
    rng = np.random.default_rng(seed)
    poses = sample_board_poses(TRUTH, HALF_EXTENT, n_views, rng)
    return project_board_corners(TRUTH, distortion, PATTERN, SQUARE, poses, noise_std=noise, rng=rng)


class TestCameraCalibrationSynthetic:
    def test_noise_free_recovers_intrinsics(self):
        views = synthetic_views(25, 0.0)
        result = calibrate_camera(views, PATTERN, SQUARE, (640, 480))
        err = evaluate_intrinsics(result, TRUTH)
        assert err.fx_rel < 1e-3 and err.fy_rel < 1e-3
        assert err.principal_point_px < 0.5
        assert err.distortion_max_abs < 5e-3
        assert result.rms < 1e-2

    def test_noise_free_recovers_distortion(self):
        true_dist = (-0.12, 0.03, 0.001, -0.0008, 0.0)
        views = synthetic_views(30, 0.0, distortion=true_dist)
        assert len(views) >= 15
        result = calibrate_camera(views, PATTERN, SQUARE, (640, 480), estimate_distortion=True)
        err = evaluate_intrinsics(result, TRUTH, true_dist)
        assert err.fx_rel < 5e-3 and err.fy_rel < 5e-3
        assert err.distortion_max_abs < 5e-3

    def test_noisy_corners_still_calibrate(self):
        views = synthetic_views(25, 0.3, seed=3)
        result = calibrate_camera(views, PATTERN, SQUARE, (640, 480), estimate_distortion=False)
        err = evaluate_intrinsics(result, TRUTH)
        assert err.fx_rel < 0.02 and err.fy_rel < 0.02
        assert err.principal_point_px < 4.0
        assert 0.05 < result.rms < 0.6

    def test_fixed_distortion_is_zero(self):
        views = synthetic_views(15, 0.2, seed=4)
        result = calibrate_camera(views, PATTERN, SQUARE, (640, 480), estimate_distortion=False)
        assert np.allclose(result.distortion, 0.0)

    def test_result_fields(self):
        views = synthetic_views(12, 0.2, seed=5)
        result = calibrate_camera(views, PATTERN, SQUARE, (640, 480), estimate_distortion=False)
        assert result.camera_matrix.shape == (3, 3)
        assert len(result.poses) == len(views)
        assert result.per_view_rms.shape == (len(views),)

    def test_needs_three_views(self):
        views = synthetic_views(2, 0.0)
        with pytest.raises(ValueError):
            calibrate_camera(views, PATTERN, SQUARE, (640, 480))

    def test_wrong_corner_count_raises(self):
        views = [np.zeros((10, 2))] * 3
        with pytest.raises(ValueError):
            calibrate_camera(views, PATTERN, SQUARE, (640, 480))

    def test_evaluate_identical_intrinsics_is_zero(self):
        views = synthetic_views(20, 0.0, seed=6)
        result = calibrate_camera(views, PATTERN, SQUARE, (640, 480))
        err = evaluate_intrinsics(result, result.intrinsics, result.distortion)
        assert err.fx_rel == 0.0 and err.principal_point_px == 0.0
        assert err.distortion_max_abs == 0.0


# -------------------------------------------------------- synthetic laser plane
LASER_TRUTH = Plane((-1.0, 0.03, -0.08), 0.25)


def synthetic_laser_views(truth_plane, n_poses, noise_px, seed=0):
    rng = np.random.default_rng(seed)
    poses = sample_board_poses(TRUTH, HALF_EXTENT, n_poses, rng)
    views = []
    for rot, trans in poses:
        board = board_plane(rot, trans)
        direction = np.cross(truth_plane.normal, board.normal)
        if np.linalg.norm(direction) < 1e-6:
            continue
        direction /= np.linalg.norm(direction)
        system = np.vstack([truth_plane.normal, board.normal, direction])
        p0 = np.linalg.solve(system, np.array([-truth_plane.d, -board.d, 0.0]))
        pts = p0 + np.linspace(-2.0, 2.0, 600)[:, None] * direction
        local = (pts - trans) @ rot
        on_board = (np.abs(local[:, 0]) <= HALF_EXTENT[0]) & (np.abs(local[:, 1]) <= HALF_EXTENT[1])
        pts = pts[on_board]
        if len(pts) < 20:
            continue
        pixels = TRUTH.project(pts)
        if noise_px > 0.0:
            pixels = pixels + rng.normal(0.0, noise_px, pixels.shape)
        views.append(LaserView(board, pixels))
    return views


class TestLaserPlaneSynthetic:
    def test_noise_free_recovers_plane(self):
        views = synthetic_laser_views(LASER_TRUTH, 15, 0.0)
        assert len(views) >= 10
        result = calibrate_laser_plane(views, TRUTH)
        err = evaluate_laser_plane(result.plane, LASER_TRUTH)
        assert err.normal_angle_deg < 1e-4
        assert err.distance_error < 1e-6
        assert result.rms < 1e-9
        assert result.num_views == len(views)

    def test_noisy_pixels_still_recover_plane(self):
        views = synthetic_laser_views(LASER_TRUTH, 20, 0.3, seed=2)
        result = calibrate_laser_plane(views, TRUTH)
        err = evaluate_laser_plane(result.plane, LASER_TRUTH)
        assert err.normal_angle_deg < 0.5
        assert err.distance_error < 0.005

    def test_single_view_is_degenerate(self):
        views = synthetic_laser_views(LASER_TRUTH, 15, 0.0)[:1]
        with pytest.raises(ValueError):
            calibrate_laser_plane(views, TRUTH)

    def test_no_points_raises(self):
        with pytest.raises(ValueError):
            calibrate_laser_plane([], TRUTH)

    def test_pixels_to_rays(self):
        rays = pixels_to_rays([[TRUTH.cx, TRUTH.cy], [TRUTH.cx + TRUTH.fx, TRUTH.cy]], TRUTH)
        assert np.allclose(rays[0], [0.0, 0.0, 1.0])
        assert np.allclose(rays[1], [1.0, 0.0, 1.0])

    def test_triangulation_lands_on_the_plane(self):
        rng = np.random.default_rng(0)
        pixels = np.column_stack([rng.uniform(TRUTH.cx + 50, TRUTH.cx + 300, 50), rng.uniform(100, 400, 50)])
        points = triangulate_laser_pixels(pixels, TRUTH, LASER_TRUTH)
        assert np.all(np.isfinite(points))
        assert np.allclose(LASER_TRUTH.signed_distance(points), 0.0, atol=1e-9)
        assert np.allclose(TRUTH.project(points), pixels, atol=1e-6)

    def test_plane_through_camera_centre_cannot_triangulate(self):
        degenerate = Plane((-1.0, 0.0, 0.0), 0.0)
        points = triangulate_laser_pixels([[400.0, 240.0], [500.0, 300.0]], TRUTH, degenerate)
        assert np.all(np.isnan(points))

    def test_evaluate_laser_plane(self):
        same = evaluate_laser_plane(LASER_TRUTH, LASER_TRUTH)
        assert same.normal_angle_deg == pytest.approx(0.0, abs=1e-6) and same.distance_error == 0.0
        tilted = Plane((-np.cos(np.radians(5.0)), np.sin(np.radians(5.0)), 0.0), 0.30)
        flat = Plane((-1.0, 0.0, 0.0), 0.25)
        err = evaluate_laser_plane(tilted, flat)
        assert err.normal_angle_deg == pytest.approx(5.0)
        assert err.distance_error == pytest.approx(0.05)


# --------------------------------------------------------- synthetic LiDAR extrinsic
T_TRUE = make_transform(Rotation.from_rotvec([0.1, -0.2, 0.3]).as_matrix(), [0.05, -0.12, 0.03])


def synthetic_lidar_views(t_c_r, n_poses, noise_std, seed=0):
    rng = np.random.default_rng(seed)
    poses = sample_board_poses(TRUTH, HALF_EXTENT, n_poses, rng)
    t_r_c = invert_transform(t_c_r)
    gx, gy = np.meshgrid(np.linspace(-0.5, 0.5, 25), np.linspace(-0.35, 0.35, 18))
    local = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    views = []
    for rot, trans in poses:
        pts_r = apply_transform(t_r_c, local @ rot.T + trans)
        if noise_std > 0.0:
            pts_r = pts_r + rng.normal(0.0, noise_std, pts_r.shape)
        views.append(PlaneCorrespondence(board_plane(rot, trans), pts_r))
    return views


class TestLidarExtrinsicSynthetic:
    def test_closed_form_recovers_transform(self):
        views = synthetic_lidar_views(T_TRUE, 12, 0.0)
        cam_planes = [v.camera_plane for v in views]
        lidar_planes = [fit_plane(v.lidar_points) for v in views]
        estimate = estimate_extrinsic_closed_form(cam_planes, lidar_planes)
        err = evaluate_extrinsic(estimate, T_TRUE)
        assert err.rotation_deg < 1e-6
        assert err.translation_m < 1e-9

    def test_full_calibration_noise_free(self):
        views = synthetic_lidar_views(T_TRUE, 12, 0.0, seed=1)
        result = calibrate_lidar_extrinsic(views)
        err = evaluate_extrinsic(result.transform, T_TRUE)
        assert err.rotation_deg < 1e-5
        assert err.translation_m < 1e-6
        assert result.rms < 1e-6
        assert result.num_views == 12
        assert result.num_points == 12 * 25 * 18

    def test_noisy_points_with_refinement(self):
        views = synthetic_lidar_views(T_TRUE, 15, 0.005, seed=2)
        result = calibrate_lidar_extrinsic(views, refine=True)
        err = evaluate_extrinsic(result.transform, T_TRUE)
        assert err.rotation_deg < 0.5
        assert err.translation_m < 0.01
        assert 0.001 < result.rms < 0.01

    def test_without_refinement_is_close(self):
        views = synthetic_lidar_views(T_TRUE, 15, 0.005, seed=3)
        result = calibrate_lidar_extrinsic(views, refine=False)
        assert np.allclose(result.transform, result.closed_form_transform)
        err = evaluate_extrinsic(result.transform, T_TRUE)
        assert err.rotation_deg < 0.5 and err.translation_m < 0.02

    def test_residuals_vanish_at_the_true_transform(self):
        views = synthetic_lidar_views(T_TRUE, 6, 0.0, seed=4)
        residuals = point_to_plane_residuals(
            T_TRUE, [v.camera_plane for v in views], [v.lidar_points for v in views]
        )
        assert np.allclose(residuals, 0.0, atol=1e-9)

    def test_identical_poses_are_degenerate(self):
        one = synthetic_lidar_views(T_TRUE, 1, 0.0)[0]
        with pytest.raises(ValueError):
            calibrate_lidar_extrinsic([one, one, one, one])

    def test_needs_three_correspondences(self):
        views = synthetic_lidar_views(T_TRUE, 2, 0.0)
        with pytest.raises(ValueError):
            calibrate_lidar_extrinsic(views)

    def test_mismatched_plane_lists_raise(self):
        views = synthetic_lidar_views(T_TRUE, 4, 0.0)
        with pytest.raises(ValueError):
            estimate_extrinsic_closed_form([v.camera_plane for v in views], [Plane((0, 0, 1), 1.0)])

    def test_bad_point_shape_raises(self):
        with pytest.raises(ValueError):
            PlaneCorrespondence(Plane((0, 0, 1), 1.0), np.zeros((5, 2)))


# ------------------------------------------------------ rendered MuJoCo calibration
class TestRenderedCalibration:
    def test_detected_corners_match_projected_ground_truth(self, rig):
        scene, camera = rig
        rng = np.random.default_rng(1)
        poses = sample_board_poses(camera.intrinsics, scene.half_extent, 5, rng)
        objp = checkerboard_object_points(scene.pattern_size, scene.square_size)
        for rot, trans in poses:
            corners = capture_board_corners(scene, camera, (rot, trans))
            assert corners is not None, "checkerboard was not detected in the rendered image"
            assert corners.shape == (40, 2)
            projected = camera.intrinsics.project(objp @ rot.T + trans)
            # The detector may return the corners in reversed order (180 degree ambiguity).
            err = min(
                np.linalg.norm(corners - projected, axis=1).mean(),
                np.linalg.norm(corners[::-1] - projected, axis=1).mean(),
            )
            assert err < 0.7

    def test_pnp_recovers_the_board_pose(self, rig):
        scene, camera = rig
        rng = np.random.default_rng(2)
        poses = sample_board_poses(camera.intrinsics, scene.half_extent, 4, rng)
        for rot_true, trans_true in poses:
            obs = observe_board(scene, camera, (rot_true, trans_true))
            assert obs is not None
            normal_angle = np.degrees(np.arccos(np.clip(rot_true[:, 2] @ obs.rotation[:, 2], -1, 1)))
            assert normal_angle < 1.0
            assert np.linalg.norm(obs.translation - trans_true) < 0.02
            assert obs.reprojection_rms < 1.0
            truth_plane = board_plane(rot_true, trans_true)
            assert obs.plane.angle_to(truth_plane) < 1.0

    def test_camera_intrinsics_from_rendered_views(self, rig):
        scene, camera = rig
        rng = np.random.default_rng(10)
        poses = sample_board_poses(camera.intrinsics, scene.half_extent, 16, rng)
        views = collect_board_corners(scene, camera, poses)
        assert len(views) >= 10, f"only {len(views)} of 16 board views were detected"
        result = calibrate_camera(
            views, scene.pattern_size, scene.square_size, scene.config.camera_resolution,
            estimate_distortion=False,
        )
        err = evaluate_intrinsics(result, camera.intrinsics)
        assert err.fx_rel < 0.03 and err.fy_rel < 0.03
        assert err.principal_point_px < 6.0
        assert result.rms < 0.8

    def test_laser_plane_from_rendered_views(self, rig):
        scene, camera = rig
        scanner = LaserScanner(
            scene, LaserConfig(fan_range_deg=(-35.0, 35.0), fan_step_deg=0.25)
        )
        rng = np.random.default_rng(11)
        poses = sample_board_poses(camera.intrinsics, scene.half_extent, 14, rng)
        views = collect_laser_views(scene, camera, scanner, poses, yaw_deg=0.0, rng=rng)
        assert len(views) >= 8, f"only {len(views)} of 14 poses produced a laser stripe"
        result = calibrate_laser_plane(views, camera.intrinsics)
        truth = ground_truth_laser_plane(scene, scanner, 0.0)
        assert truth.d == pytest.approx(0.25, abs=1e-6)  # plane x = 0.25 m in the camera frame
        err = evaluate_laser_plane(result.plane, truth)
        assert err.normal_angle_deg < 1.0
        assert err.distance_error < 0.02
        assert result.rms < 0.01

    def test_lidar_extrinsic_from_rendered_views(self, rig):
        scene, camera = rig
        lidar = Lidar(
            scene,
            LidarConfig(azimuth_range_deg=(-45.0, 45.0), elevation_range_deg=(-30.0, 30.0), resolution_deg=1.0),
        )
        rng = np.random.default_rng(12)
        poses = sample_board_poses(camera.intrinsics, scene.half_extent, 14, rng)
        views = collect_lidar_views(scene, camera, lidar, poses)
        assert len(views) >= 8, f"only {len(views)} of 14 poses were usable"
        result = calibrate_lidar_extrinsic(views)
        truth = ground_truth_frame_tree(scene).get_transform("C", "R")
        err = evaluate_extrinsic(result.transform, truth)
        assert err.rotation_deg < 2.0
        assert err.translation_m < 0.05
        assert result.rms < 0.01