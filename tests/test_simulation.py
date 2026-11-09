"""Unit tests for the simulation package."""

import numpy as np
import pytest

from simulation.environment import Environment, EnvironmentConfig
from simulation.ground_truth import GroundTruth
from simulation.motion_generator import (
    CircularTrajectory,
    LinearBounceTrajectory,
    SinusoidalTrajectory,
)
from simulation.obstacle_manager import ObstacleManager, ObstacleSpec, default_obstacles


# ----------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def env():
    return Environment()


# --------------------------------------------------------- motion generators
class TestMotionGenerator:
    def test_linear_bounce_stays_in_bounds(self):
        traj = LinearBounceTrajectory(
            start=(0.0, 0.0, 1.0),
            velocity=(0.7, -0.4, 0.3),
            bounds_min=(-1.0, -1.0, 0.5),
            bounds_max=(1.0, 1.0, 1.5),
        )
        for t in np.arange(0.0, 100.0, 0.05):
            p = traj.position(t)
            assert np.all(p >= traj.bounds_min - 1e-9)
            assert np.all(p <= traj.bounds_max + 1e-9)

    def test_linear_bounce_speed_is_constant_per_axis(self):
        v = np.array([0.7, -0.4, 0.3])
        traj = LinearBounceTrajectory(
            (0.0, 0.0, 1.0), v, (-1.0, -1.0, 0.5), (1.0, 1.0, 1.5)
        )
        for t in np.arange(0.0, 50.0, 0.1):
            assert np.allclose(np.abs(traj.velocity(t)), np.abs(v))

    def test_linear_bounce_pinned_axis_is_2d(self):
        traj = LinearBounceTrajectory(
            (0.0, 0.0, 0.2), (0.5, 0.3, 0.0), (-1, -1, 0.2), (1, 1, 0.2)
        )
        for t in np.arange(0.0, 30.0, 0.1):
            assert traj.position(t)[2] == pytest.approx(0.2)
            assert traj.velocity(t)[2] == 0.0

    def test_linear_bounce_rejects_start_outside_bounds(self):
        with pytest.raises(ValueError):
            LinearBounceTrajectory((5, 0, 0), (1, 0, 0), (-1, -1, -1), (1, 1, 1))

    def test_sinusoidal_velocity_matches_finite_difference(self):
        traj = SinusoidalTrajectory(
            (0, 0, 1), (1.0, 0.5, 0.3), (0.5, 0.9, 1.1), (0.0, 0.5, 1.0)
        )
        h = 1e-5
        for t in (0.0, 1.3, 7.7, 20.1):
            numeric = (traj.position(t + h) - traj.position(t - h)) / (2 * h)
            assert np.allclose(traj.velocity(t), numeric, atol=1e-6)

    def test_circular_radius_is_constant(self):
        traj = CircularTrajectory((1.0, 2.0, 0.5), radius=0.8, omega=0.6)
        for t in np.arange(0.0, 30.0, 0.2):
            p = traj.position(t)
            assert np.hypot(p[0] - 1.0, p[1] - 2.0) == pytest.approx(0.8)
            assert p[2] == pytest.approx(0.5)

    def test_circular_velocity_matches_finite_difference(self):
        traj = CircularTrajectory(
            (0, 0, 1), radius=0.9, omega=-0.8, phase=1.0, z_amplitude=0.4, z_omega=1.3
        )
        h = 1e-5
        for t in (0.0, 2.2, 9.9):
            numeric = (traj.position(t + h) - traj.position(t - h)) / (2 * h)
            assert np.allclose(traj.velocity(t), numeric, atol=1e-6)

    def test_circular_rejects_nonpositive_radius(self):
        with pytest.raises(ValueError):
            CircularTrajectory((0, 0, 0), radius=0.0, omega=1.0)

    def test_wrong_vector_length_raises(self):
        with pytest.raises(ValueError):
            SinusoidalTrajectory((0, 0), (1, 1, 1), (1, 1, 1))


# -------------------------------------------------------- obstacle manager
class TestObstacleManager:
    def test_default_obstacles_ids_and_counts(self):
        specs = default_obstacles()
        assert [s.obstacle_id for s in specs] == [f"O00{i}" for i in range(1, 7)]
        assert sum(s.motion_dim == "2D" for s in specs) == 3
        assert sum(s.motion_dim == "3D" for s in specs) == 3

    def test_default_obstacles_geometries(self):
        geoms = {s.obstacle_id: s.geometry for s in default_obstacles()}
        assert geoms == {
            "O001": "box", "O002": "sphere", "O003": "cylinder",
            "O004": "box", "O005": "sphere", "O006": "cylinder",
        }

    def test_2d_obstacles_keep_z_fixed_and_3d_obstacles_vary(self):
        times = np.arange(0.0, 120.0, 0.05)
        for spec in default_obstacles():
            z = np.array([spec.trajectory.position(t)[2] for t in times])
            vz = np.array([spec.trajectory.velocity(t)[2] for t in times])
            if spec.motion_dim == "2D":
                assert np.ptp(z) < 1e-9
                assert np.all(vz == 0.0)
            else:
                assert np.ptp(z) > 0.2

    def test_default_obstacles_stay_inside_arena(self):
        times = np.arange(0.0, 120.0, 0.05)
        for spec in default_obstacles():
            pts = np.array([spec.trajectory.position(t) for t in times])
            assert np.all(np.abs(pts[:, 0]) <= 2.5)
            assert np.all(np.abs(pts[:, 1]) <= 2.5)
            assert np.all(pts[:, 2] >= 0.0)
            assert np.all(pts[:, 2] <= 3.0)

    def test_invalid_geometry_raises(self):
        traj = CircularTrajectory((0, 0, 0), 1.0, 1.0)
        with pytest.raises(ValueError):
            ObstacleSpec("X", "cone", (0.1,), "2D", traj)

    def test_wrong_size_length_raises(self):
        traj = CircularTrajectory((0, 0, 0), 1.0, 1.0)
        with pytest.raises(ValueError):
            ObstacleSpec("X", "box", (0.1,), "2D", traj)

    def test_invalid_motion_dim_raises(self):
        traj = CircularTrajectory((0, 0, 0), 1.0, 1.0)
        with pytest.raises(ValueError):
            ObstacleSpec("X", "sphere", (0.1,), "4D", traj)

    def test_duplicate_ids_raise(self):
        traj = CircularTrajectory((0, 0, 0), 1.0, 1.0)
        a = ObstacleSpec("X", "sphere", (0.1,), "2D", traj)
        b = ObstacleSpec("X", "sphere", (0.1,), "2D", traj)
        with pytest.raises(ValueError):
            ObstacleManager([a, b])


# -------------------------------------------------------------- environment
class TestEnvironment:
    def test_model_has_camera_sites_and_mocap_bodies(self, env):
        import mujoco

        cfg = env.config
        assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cfg.camera_name) >= 0
        for site in (cfg.lidar_site_name, cfg.laser_site_name):
            assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, site) >= 0
        assert env.model.nmocap == 6

    def test_timestep_matches_config(self, env):
        assert env.model.opt.timestep == pytest.approx(env.config.timestep)

    def test_camera_looks_along_world_plus_y(self, env):
        env.reset()
        _, rot = env.get_camera_pose()
        forward = -rot[:, 2]  # MuJoCo camera looks along local -Z
        up = rot[:, 1]  # MuJoCo camera +Y is image-up
        assert np.allclose(forward, [0.0, 1.0, 0.0], atol=1e-9)
        assert np.allclose(up, [0.0, 0.0, 1.0], atol=1e-9)

    def test_sensor_poses_are_fixed_over_time(self, env):
        env.reset()
        cam0 = env.get_camera_pose()
        lidar0 = env.get_site_pose(env.config.lidar_site_name)
        for _ in range(200):
            env.step()
        cam1 = env.get_camera_pose()
        lidar1 = env.get_site_pose(env.config.lidar_site_name)
        assert np.allclose(cam0[0], cam1[0]) and np.allclose(cam0[1], cam1[1])
        assert np.allclose(lidar0[0], lidar1[0]) and np.allclose(lidar0[1], lidar1[1])

    def test_unknown_site_raises(self, env):
        with pytest.raises(ValueError):
            env.get_site_pose("does_not_exist")

    def test_reset_sets_time_zero_and_initial_positions(self, env):
        for _ in range(50):
            env.step()
        env.reset()
        assert env.time == 0.0
        for oid in env.obstacles.ids:
            expected = env.obstacles.get_spec(oid).trajectory.position(0.0)
            actual = env.data.xpos[env.obstacles.body_id(oid)]
            assert np.allclose(actual, expected, atol=1e-9)

    def test_step_advances_time_and_obstacles_follow_trajectories(self, env):
        env.reset()
        n = 500
        for _ in range(n):
            env.step()
        assert env.time == pytest.approx(n * env.config.timestep)
        for oid in env.obstacles.ids:
            expected = env.obstacles.get_spec(oid).trajectory.position(env.time)
            actual = env.data.xpos[env.obstacles.body_id(oid)]
            assert np.allclose(actual, expected, atol=1e-9)

    def test_steps_per_period(self, env):
        assert env.steps_per_period(50.0) == 10  # 1 / (50 * 0.002)
        assert env.steps_per_period(500.0) == 1

    def test_steps_per_period_rejects_invalid_rate(self, env):
        with pytest.raises(ValueError):
            env.steps_per_period(0.0)
        with pytest.raises(ValueError):
            env.steps_per_period(-10.0)

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError):
            EnvironmentConfig(timestep=0.0)
        with pytest.raises(ValueError):
            EnvironmentConfig(duration=-1.0)

    def test_custom_obstacle_list(self):
        traj = CircularTrajectory((0, 0, 0.5), 0.5, 1.0)
        spec = ObstacleSpec("A001", "sphere", (0.1,), "2D", traj)
        custom_env = Environment(obstacles=[spec])
        assert custom_env.model.nmocap == 1
        assert custom_env.obstacles.ids == ["A001"]


# ------------------------------------------------------------- ground truth
class TestGroundTruth:
    def test_snapshot_contains_all_obstacles(self, env):
        env.reset()
        records = GroundTruth(env).snapshot()
        assert [r.obstacle_id for r in records] == env.obstacles.ids

    def test_snapshot_matches_analytic_state(self, env):
        env.reset()
        for _ in range(250):
            env.step()
        for r in GroundTruth(env).snapshot():
            pos, vel = env.obstacles.state_at(r.obstacle_id, env.time)
            assert np.allclose(r.position, pos, atol=1e-9)
            assert np.allclose(r.velocity, vel, atol=1e-12)

    def test_2d_obstacles_have_zero_vertical_velocity(self, env):
        env.reset()
        for _ in range(100):
            env.step()
        for r in GroundTruth(env).snapshot():
            if r.motion_dim == "2D":
                assert r.velocity[2] == 0.0

    def test_record_and_to_array_shapes(self, env):
        env.reset()
        gt = GroundTruth(env)
        for _ in range(5):
            gt.record()
            for _ in range(10):
                env.step()
        assert len(gt) == 5
        assert gt.times().shape == (5,)
        assert gt.to_array().shape == (5, 6, 6)
        assert np.all(np.diff(gt.times()) > 0)

    def test_to_array_empty(self, env):
        assert GroundTruth(env).to_array().shape == (0, 6, 6)

    def test_clear(self, env):
        env.reset()
        gt = GroundTruth(env)
        gt.record()
        gt.clear()
        assert len(gt) == 0

    def test_save_csv(self, env, tmp_path):
        env.reset()
        gt = GroundTruth(env)
        for _ in range(3):
            gt.record()
            env.step()
        out = gt.save_csv(tmp_path / "logs" / "gt.csv")
        lines = out.read_text().strip().splitlines()
        assert lines[0].startswith("time,obstacle_id")
        assert len(lines) == 1 + 3 * 6