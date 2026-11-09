"""Integration tests for main.py (the full simulation pipeline).

Fast tests need no OpenGL. Tests that render images (RGB-D, dashboard, calibration)
use the `gl_available` fixture and are skipped when offscreen rendering is not
available (for example MUJOCO_GL is not configured). Slow tests carry the `slow` marker:

    pytest tests/test_integration.py                 # everything
    pytest tests/test_integration.py -m "not slow"   # skip calibration and video tests
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

mujoco = pytest.importorskip("mujoco")

import main as pipeline  # noqa: E402
from calibration.coordinate_frames import ground_truth_frame_tree, transform_error  # noqa: E402
from simulation.environment import Environment  # noqa: E402


# ---------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def gl_available() -> bool:
    """Skip the test when MuJoCo cannot create an offscreen renderer."""
    try:
        env = Environment()
        renderer = mujoco.Renderer(env.model, height=120, width=160)
        renderer.close()
    except Exception as exc:  # noqa: BLE001 - any GL failure means "cannot render here"
        pytest.skip(f"offscreen rendering not available: {exc}")
    return True


def make_config(tmp_path: Path, **overrides) -> pipeline.RunConfig:
    """Short, deterministic, quiet configuration writing into tmp_path."""
    params = dict(
        duration=4.0,
        seed=0,
        sensors=("lidar",),
        output_dir=str(tmp_path / "run"),
        noise=False,
        verbose=False,
    )
    params.update(overrides)
    return pipeline.RunConfig(**params)


def fake_track(track_id: int, position, status: str = "confirmed", p_3d: float = 0.7, updated: bool = True):
    """Minimal stand-in for tracking.Track (only what RunLog reads)."""
    return SimpleNamespace(
        track_id=track_id,
        position=np.asarray(position, dtype=float),
        velocity=np.array([0.1, 0.2, 0.0]),
        status=SimpleNamespace(value=status),
        model_probabilities={"CV2D": 1.0 - p_3d, "CV3D": p_3d},
        updated=updated,
    )


# ------------------------------------------------------------------- config and parsing
def test_parse_sensors_orders_and_validates():
    assert pipeline.parse_sensors("lidar, rgbd") == ("rgbd", "lidar")
    with pytest.raises(ValueError):
        pipeline.parse_sensors("radar")
    with pytest.raises(ValueError):
        pipeline.parse_sensors(" , ")


def test_run_config_validation():
    pipeline.RunConfig().validate()
    for bad in (
        dict(duration=0.0),
        dict(sensors=()),
        dict(sensors=("sonar",)),
        dict(estimator="pf"),
        dict(fusion="average"),
        dict(calibration="magic"),
        dict(lidar_rate_hz=0.0),
        dict(calibration_views=2),
    ):
        with pytest.raises(ValueError):
            pipeline.RunConfig(**bad).validate()


def test_parse_args_maps_flags_to_config():
    cfg = pipeline.parse_args(
        ["--duration", "5", "--sensors", "lidar,rgbd", "--estimator", "ukf", "--no-imm",
         "--fusion", "merged", "--no-noise", "--no-save", "--quiet", "--laser-line"]
    )
    assert cfg.duration == 5.0
    assert cfg.sensors == ("rgbd", "lidar")
    assert cfg.estimator == "ukf" and not cfg.use_imm
    assert cfg.fusion == "merged"
    assert not cfg.noise and not cfg.save_log and not cfg.verbose and not cfg.laser_sweep


def test_next_due_time_skips_past_instants():
    assert pipeline.next_due_time(0.0, 0.1, 0.0) == pytest.approx(0.1)
    assert pipeline.next_due_time(0.1, 0.1, 0.1) == pytest.approx(0.2)
    # A late loop iteration must not produce a burst of catch-up samples.
    assert pipeline.next_due_time(0.1, 0.1, 0.55) == pytest.approx(0.6)


# ---------------------------------------------------------------------- extrinsics
def test_nominal_transforms_match_the_model():
    """The mounting geometry hard-coded in main.py must agree with the MuJoCo model."""
    env = Environment()
    t_w_c, t_w_r, t_w_l = pipeline.nominal_transforms(env.config)
    tree = ground_truth_frame_tree(env)
    for nominal, frame in ((t_w_c, "C"), (t_w_r, "R"), (t_w_l, "L")):
        rot_err_deg, trans_err_m = transform_error(nominal, tree.get_transform("W", frame))
        assert rot_err_deg < 1e-4, f"rotation of frame {frame} differs from the model"
        assert trans_err_m < 1e-9, f"translation of frame {frame} differs from the model"


def test_camera_axes_look_along_world_y():
    env = Environment()
    t_w_c, _, _ = pipeline.nominal_transforms(env.config)
    forward = t_w_c[:3, 2]  # optical axis (OpenCV z) in world coordinates
    down = t_w_c[:3, 1]
    assert np.allclose(forward, [0.0, 1.0, 0.0])
    assert np.allclose(down, [0.0, 0.0, -1.0])


def test_ground_truth_mode_equals_nominal_mode():
    env = Environment()
    nominal = pipeline.resolve_extrinsics(env, pipeline.RunConfig(calibration="nominal"))
    truth = pipeline.resolve_extrinsics(env, pipeline.RunConfig(calibration="ground_truth"))
    assert np.allclose(nominal.t_w_c, truth.t_w_c, atol=1e-9)
    assert np.allclose(nominal.t_w_r, truth.t_w_r, atol=1e-9)
    assert np.allclose(nominal.t_w_l, truth.t_w_l, atol=1e-9)


def test_sensor_positions_only_for_active_sensors():
    env = Environment()
    ext = pipeline.resolve_extrinsics(env, pipeline.RunConfig())
    positions = ext.sensor_positions(("rgbd", "lidar"))
    assert set(positions) == {"rgbd", "lidar"}
    assert positions["lidar"][2] > positions["rgbd"][2]  # the LiDAR sits above the camera


# --------------------------------------------------------------------------- RunLog
def test_run_log_builds_dense_arrays_with_gaps():
    log = pipeline.RunLog()
    log.add(0.0, [fake_track(1, [0, 0, 1])], [])
    log.add(0.1, [fake_track(1, [0.1, 0, 1]), fake_track(3, [1, 1, 1], status="tentative")],
            [SimpleNamespace(sensor="lidar", position=np.array([1.0, 1.0, 1.0]))])
    log.add(0.2, [fake_track(3, [1.1, 1, 1], status="lost")], [])
    arrays = log.to_arrays()

    assert arrays["times"].shape == (3,)
    assert list(arrays["track_ids"]) == [1, 3]
    assert arrays["est_states"].shape == (3, 2, 6)
    assert np.isnan(arrays["est_states"][0, 1]).all()  # track 3 not alive yet
    assert np.isnan(arrays["est_states"][2, 0]).all()  # track 1 gone
    assert arrays["est_status"][1, 1] == pipeline.STATUS_CODES["tentative"]
    assert arrays["est_status"][2, 1] == pipeline.STATUS_CODES["lost"]
    assert arrays["est_status"][2, 0] == 0
    assert arrays["meas_sensor"].tolist() == ["lidar"]
    assert arrays["meas_position"].shape == (1, 3)


def test_run_log_empty():
    arrays = pipeline.RunLog().to_arrays()
    assert arrays["times"].shape == (0,)
    assert arrays["est_states"].shape == (0, 0, 6)
    assert arrays["meas_position"].shape == (0, 3)


def test_summarize_counts_confirmed_matches():
    gt = np.zeros((2, 1, 6))
    gt[:, 0, :3] = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    est = np.full((2, 1, 6), np.nan)
    est[0, 0, :3] = [1.1, 1.0, 1.0]
    est[1, 0, :3] = [1.0, 1.0, 1.0]
    status = np.array([[2], [2]], dtype=np.int8)
    summary = pipeline.summarize({"gt_states": gt, "est_states": est, "est_status": status})
    assert summary["coverage"] == pytest.approx(1.0)
    assert summary["tracks_ever_confirmed"] == 1
    assert summary["nearest_track_rmse_m"] == pytest.approx(np.sqrt(0.5 * 0.1**2), rel=1e-6)


# ------------------------------------------------------------------- full runs (no GL)
def test_lidar_only_run_tracks_obstacles(tmp_path):
    cfg = make_config(tmp_path, duration=6.0, sensors=("lidar",))
    result = pipeline.run(cfg)
    arrays = result.arrays

    n_frames = len(arrays["times"])
    assert n_frames > 0
    assert np.all(np.diff(arrays["times"]) > 0.0), "frame times must increase"
    assert arrays["gt_states"].shape == (n_frames, 6, 6)
    assert arrays["gt_ids"].tolist() == ["O001", "O002", "O003", "O004", "O005", "O006"]
    assert arrays["est_states"].shape[0] == n_frames
    assert set(np.unique(arrays["est_status"])) <= {0, 1, 2, 3}

    # Only the LiDAR is active, so every measurement must come from it.
    assert len(arrays["meas_sensor"]) > 0
    assert set(arrays["meas_sensor"].tolist()) == {"lidar"}

    # Sanity of the tracking result (the formal metrics belong to run_evaluation.py).
    assert result.summary["tracks_ever_confirmed"] >= 2
    assert result.summary["coverage"] > 0.1
    assert result.summary["nearest_track_rmse_m"] < 0.3


def test_run_writes_log_files(tmp_path):
    cfg = make_config(tmp_path, duration=2.0)
    result = pipeline.run(cfg)

    out = Path(cfg.output_dir)
    assert result.log_path == out / "tracking_log.npz"
    assert (out / "ground_truth.csv").is_file()
    info = json.loads((out / "run_info.json").read_text())
    assert info["config"]["sensors"] == ["lidar"]
    assert info["calibration"]["mode"] == "nominal"

    with np.load(result.log_path) as data:  # no pickle needed: all arrays are numeric / unicode
        for key in ("times", "gt_ids", "gt_states", "track_ids", "est_states", "est_status", "meas_sensor"):
            assert key in data.files
        assert data["gt_states"].shape[0] == data["times"].shape[0]


def test_no_save_writes_nothing(tmp_path):
    cfg = make_config(tmp_path, duration=1.0, save_log=False)
    result = pipeline.run(cfg)
    assert result.log_path is None
    assert not Path(cfg.output_dir).exists()


def test_same_seed_gives_identical_logs(tmp_path):
    first = pipeline.run(make_config(tmp_path / "a", duration=2.0, noise=True, seed=3, save_log=False))
    second = pipeline.run(make_config(tmp_path / "b", duration=2.0, noise=True, seed=3, save_log=False))
    assert np.array_equal(first.arrays["times"], second.arrays["times"])
    assert np.allclose(first.arrays["est_states"], second.arrays["est_states"], equal_nan=True)
    assert np.allclose(first.arrays["meas_position"], second.arrays["meas_position"])


@pytest.mark.parametrize("estimator,use_imm", [("ekf", False), ("ukf", True)])
def test_estimator_variants_run(tmp_path, estimator, use_imm):
    cfg = make_config(tmp_path, duration=2.0, estimator=estimator, use_imm=use_imm, save_log=False)
    result = pipeline.run(cfg)
    assert len(result.arrays["times"]) > 0
    if use_imm:
        assert np.isfinite(result.arrays["est_model_prob_3d"]).any()


def test_ground_truth_is_not_the_measurement(tmp_path):
    """Measurements come from sensor data: they must not coincide with the exact obstacle positions."""
    cfg = make_config(tmp_path, duration=3.0, noise=True, save_log=False)
    arrays = pipeline.run(cfg).arrays
    gt_all = arrays["gt_states"][:, :, :3]
    times = arrays["times"]
    distances = []
    for t_meas, position in zip(arrays["meas_time"], arrays["meas_position"]):
        frame = int(np.argmin(np.abs(times - t_meas)))
        distances.append(np.linalg.norm(gt_all[frame] - position, axis=1).min())
    assert len(distances) > 0
    assert min(distances) > 1e-6  # centroids of noisy clusters never equal the true centre


# -------------------------------------------------------------------- full runs (GL)
@pytest.mark.parametrize("fusion", ["sequential", "merged"])
def test_all_sensors_run(tmp_path, gl_available, fusion):
    cfg = make_config(
        tmp_path, duration=1.5, sensors=pipeline.ALL_SENSORS, fusion=fusion, noise=True
    )
    result = pipeline.run(cfg)
    assert len(result.arrays["times"]) > 0
    sensors_seen = set()
    for name in result.arrays["meas_sensor"].tolist():
        sensors_seen.update(name.split("+"))  # fused measurements are named "a+b"
    assert sensors_seen <= set(pipeline.ALL_SENSORS)
    assert "lidar" in sensors_seen


def test_rgbd_only_run_produces_rgbd_measurements(tmp_path, gl_available):
    cfg = make_config(tmp_path, duration=1.5, sensors=("rgbd",), noise=True, save_log=False)
    arrays = pipeline.run(cfg).arrays
    assert len(arrays["meas_sensor"]) > 0
    assert set(arrays["meas_sensor"].tolist()) == {"rgbd"}


@pytest.mark.slow
def test_calibrated_mode_recovers_lidar_extrinsic(gl_available):
    env = Environment()
    cfg = pipeline.RunConfig(calibration="calibrated", calibration_views=10, noise=False, seed=1)
    ext = pipeline.resolve_extrinsics(env, cfg)
    assert ext.mode == "calibrated"
    assert ext.info["camera_reprojection_rms_px"] < 1.0
    assert ext.info["lidar_rotation_error_deg"] < 2.0
    assert ext.info["lidar_translation_error_m"] < 0.05


@pytest.mark.slow
def test_dashboard_video_is_written(tmp_path, gl_available):
    video_path = tmp_path / "dashboard.mp4"
    cfg = make_config(tmp_path, duration=1.0, video_path=str(video_path), video_period=0.5, save_log=False)
    try:
        pipeline.run(cfg)
    except RuntimeError as exc:
        if "video writer" in str(exc):
            pytest.skip(str(exc))
        raise
    assert video_path.is_file() and video_path.stat().st_size > 0