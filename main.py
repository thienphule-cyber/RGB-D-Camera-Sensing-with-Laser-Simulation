"""Run the full multi-object tracking pipeline in MuJoCo.

Pipeline (one pass of the main loop):

    Environment.step()                      scripted obstacles, exact ground truth
    -> sensors (RGB-D / laser / LiDAR)    each at its own rate
    -> detectors                          point cloud -> clusters -> Detection
    -> MeasurementFusion                  Detection -> Measurement (covariance, fusion)
    -> ObjectTracker                      association + EKF/UKF/IMM + track life cycle
    -> RunLog / GroundTruth               saved for run_evaluation.py

Ground truth is only logged (and drawn as a reference in the dashboard); it is never
given to the detectors, the fusion or the tracker.

Examples:
    python main.py                                   # 60 s, all sensors, EKF + IMM, headless
    python main.py --duration 20 --estimator ukf
    python main.py --sensors lidar --no-imm          # sensor ablation (E04 / E05)
    python main.py --calibration calibrated          # run the calibration first
    python main.py --video outputs/run/dashboard.mp4 # dashboard video (slow)
    python main.py --viewer                          # MuJoCo viewer (macOS: use `mjpython`)

Output (in --output-dir):
    tracking_log.npz   times, gt_ids, gt_states, track_ids, est_states, est_status, ...
    ground_truth.csv   ground truth log
    run_info.json      configuration, calibration info and a quick summary
"""

from __future__ import annotations

import argparse
import json
import time as wall_clock
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import mujoco

from calibration.coordinate_frames import (
    ground_truth_frame_tree,
    make_transform,
    transform_error,
)
from fusion.measurement_fusion import MeasurementFusion
from perception.object_detection import (
    LaserDetector,
    LidarDetector,
    RGBDDetector,
    default_detector_config,
)
from sensors.common import FORWARD_Y_MOUNT, MUJOCO_TO_OPENCV
from sensors.depth_camera import DepthCamera
from sensors.laser_scanner import LaserConfig, LaserScanner
from sensors.lidar import Lidar, LidarConfig
from sensors.rgb_camera import CameraIntrinsics, RGBCamera
from simulation.environment import Environment, EnvironmentConfig
from simulation.ground_truth import GroundTruth
from tracking.object_tracker import ObjectTracker, TrackerConfig
from tracking.track_manager import make_filter_factory
from fusion.measurement_covariance import CovarianceConfig, SensorNoise
from perception.segmentation import ClusterConfig
from tracking.track_manager import TrackManagerConfig, make_filter_factory

ALL_SENSORS = ("rgbd", "laser", "lidar")
ESTIMATORS = ("ekf", "ukf")
FUSION_MODES = ("sequential", "merged")
CALIBRATION_MODES = ("nominal", "calibrated", "ground_truth")

# Track status codes stored in the log (0 = track not alive in that frame).
STATUS_CODES = {"tentative": 1, "confirmed": 2, "lost": 3}

# The LiDAR site sits 0.12 m above the sensor assembly origin (see Environment._build_xml).
LIDAR_SITE_OFFSET = np.array([0.0, 0.0, 0.12])

# Sensor noise used by the simulated sensors. These match default_sensor_noise() of the
# covariance model, so the filter covariances are consistent with the simulated errors.
RGBD_DEPTH_NOISE = (0.005, 0.002)  # (constant [m], per m^2 [m / m^2])
LASER_NOISE = (0.010, 0.002)
LIDAR_RANGE_NOISE = 0.010  # [m]


# ----------------------------------------------------------------------------- config
@dataclass
class RunConfig:
    """All parameters of one simulation run."""

    duration: float = 60.0  # simulated time [s]
    seed: int = 0
    sensors: Tuple[str, ...] = ALL_SENSORS
    estimator: str = "ekf"  # "ekf" or "ukf"
    use_imm: bool = True  # IMM over the 2D / 3D motion models, else one CV3D filter
    fusion: str = "sequential"  # "sequential" (inflated covariances) or "merged" (cross-sensor fusion)
    calibration: str = "nominal"  # "nominal", "calibrated" or "ground_truth"
    calibration_views: int = 12
    rgbd_rate_hz: float = 30.0
    laser_rate_hz: float = 20.0
    lidar_rate_hz: float = 10.0
    estimator_rate_hz: float = 100.0  # time resolution of the main loop
    laser_sweep: bool = True  # sweep the laser plane (else a single line)
    noise: bool = True  # simulate sensor noise
    viewer: bool = False
    speed: float = 1.0  # viewer pacing: simulated seconds per wall-clock second
    video_path: Optional[str] = None
    video_period: float = 0.2  # simulated time between dashboard frames [s]
    output_dir: str = "outputs/run"
    save_log: bool = True
    verbose: bool = True

    def validate(self) -> None:
        if self.duration <= 0.0:
            raise ValueError("duration must be positive")
        if not self.sensors:
            raise ValueError("at least one sensor is required")
        unknown = [s for s in self.sensors if s not in ALL_SENSORS]
        if unknown:
            raise ValueError(f"unknown sensors {unknown}; choose from {ALL_SENSORS}")
        if self.estimator not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {ESTIMATORS}")
        if self.fusion not in FUSION_MODES:
            raise ValueError(f"fusion must be one of {FUSION_MODES}")
        if self.calibration not in CALIBRATION_MODES:
            raise ValueError(f"calibration must be one of {CALIBRATION_MODES}")
        for name in ("rgbd_rate_hz", "laser_rate_hz", "lidar_rate_hz", "estimator_rate_hz", "speed", "video_period"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.calibration_views < 3:
            raise ValueError("calibration_views must be >= 3")

    def sensor_rates(self) -> Dict[str, float]:
        rates = {"rgbd": self.rgbd_rate_hz, "laser": self.laser_rate_hz, "lidar": self.lidar_rate_hz}
        return {name: rates[name] for name in self.sensors}


def parse_sensors(text: str) -> Tuple[str, ...]:
    """Parse "rgbd,lidar" into a tuple in canonical order; raises ValueError on unknown names."""
    names = {part.strip().lower() for part in text.split(",") if part.strip()}
    unknown = sorted(names - set(ALL_SENSORS))
    if unknown:
        raise ValueError(f"unknown sensors {unknown}; choose from {ALL_SENSORS}")
    if not names:
        raise ValueError("at least one sensor is required")
    return tuple(s for s in ALL_SENSORS if s in names)


def next_due_time(previous_due: float, period: float, now: float) -> float:
    """Next sampling time of a sensor: previous + period, skipping instants already in the past."""
    due = previous_due + period
    while due <= now - 1e-9:
        due += period
    return due


# ------------------------------------------------------------------------ extrinsics
@dataclass
class Extrinsics:
    """Sensor-to-world transforms and camera intrinsics used by the estimation pipeline."""

    mode: str
    intrinsics: CameraIntrinsics
    t_w_c: np.ndarray  # camera (OpenCV optical frame) -> world
    t_w_r: np.ndarray  # LiDAR -> world
    t_w_l: np.ndarray  # laser -> world
    info: Dict[str, float] = field(default_factory=dict)

    def sensor_positions(self, sensors: Sequence[str]) -> Dict[str, np.ndarray]:
        """World position of each active sensor (used for the measurement covariances)."""
        all_positions = {
            "rgbd": self.t_w_c[:3, 3],
            "laser": self.t_w_l[:3, 3],
            "lidar": self.t_w_r[:3, 3],
        }
        return {name: all_positions[name].copy() for name in sensors}


def nominal_transforms(env_cfg: EnvironmentConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(T_W_C, T_W_R, T_W_L) from the known mounting geometry (no simulation state is read).

    The assembly body has no rotation. The camera is defined by xyaxes="1 0 0 0 0 1", so its
    MuJoCo axes are x = +X, y = +Z, z = x cross y = -Y (it looks along world +Y). The LiDAR
    and laser sites share the world orientation and use the forward-Y mount.
    """
    origin = np.asarray(env_cfg.sensor_position, dtype=float)
    axis_x = np.array([1.0, 0.0, 0.0])
    axis_y = np.array([0.0, 0.0, 1.0])
    rot_mujoco = np.column_stack([axis_x, axis_y, np.cross(axis_x, axis_y)])
    t_w_c = make_transform(rot_mujoco @ MUJOCO_TO_OPENCV, origin)
    t_w_r = make_transform(FORWARD_Y_MOUNT, origin + LIDAR_SITE_OFFSET)
    t_w_l = make_transform(FORWARD_Y_MOUNT, origin + np.asarray(env_cfg.laser_offset, dtype=float))
    return t_w_c, t_w_r, t_w_l


def lidar_config(cfg: RunConfig) -> LidarConfig:
    return LidarConfig(range_noise_std=LIDAR_RANGE_NOISE if cfg.noise else 0.0)


def laser_config(cfg: RunConfig) -> LaserConfig:
    noise_std, noise_per_m2 = LASER_NOISE if cfg.noise else (0.0, 0.0)
    return LaserConfig(noise_std=noise_std, noise_std_per_m2=noise_per_m2)


def covariance_config(cfg: RunConfig) -> CovarianceConfig:
    rgbd_c, rgbd_q = RGBD_DEPTH_NOISE if cfg.noise else (0.0, 0.0)
    laser_c, laser_q = LASER_NOISE if cfg.noise else (0.0, 0.0)
    lidar_r = LIDAR_RANGE_NOISE if cfg.noise else 0.0
    return CovarianceConfig(
        sensors={
            "rgbd": SensorNoise(rgbd_c, rgbd_q, 0.003),
            "laser": SensorNoise(laser_c, laser_q, 0.015),  # coarse sweep: larger cross-range error
            "lidar": SensorNoise(lidar_r, 0.0, 0.008),
        },
        centroid_shift=0.06,
    )


def calibrate_sensors(
    env_cfg: EnvironmentConfig,
    cfg: RunConfig,
    nominal_intrinsics: CameraIntrinsics,
    t_w_c: np.ndarray,
    t_w_r: np.ndarray,
    t_w_l: np.ndarray,
) -> Extrinsics:
    """Run Zhang calibration and LiDAR-to-camera calibration on a separate calibration scene.

    T_W_C comes from the mounting geometry (the camera is fixed). The laser keeps its
    mounting transform: the laser plane calibration (see calibration.laser_plane) is
    needed for image-based triangulation, which this ray-cast laser model does not use.
    """
    from calibration.camera_intrinsic import calibrate_camera, collect_board_corners
    from calibration.lidar_extrinsic import calibrate_lidar_extrinsic, collect_lidar_views
    from calibration.target_scene import CalibrationScene, sample_board_poses

    scene = CalibrationScene(config=env_cfg)
    camera = RGBCamera(scene, seed=cfg.seed)
    lidar = Lidar(scene, lidar_config(cfg), seed=cfg.seed + 1)
    try:
        rng = np.random.default_rng(cfg.seed)
        poses = sample_board_poses(camera.intrinsics, scene.half_extent, cfg.calibration_views, rng)
        corners = collect_board_corners(scene, camera, poses)
        if len(corners) < 3:
            raise RuntimeError(f"checkerboard detected in only {len(corners)} views; need >= 3")
        camera_result = calibrate_camera(
            corners,
            scene.pattern_size,
            scene.square_size,
            (nominal_intrinsics.width, nominal_intrinsics.height),
            estimate_distortion=False,
        )
        views = collect_lidar_views(scene, camera, lidar, poses, camera_result.intrinsics)
        lidar_result = calibrate_lidar_extrinsic(views)
    finally:
        camera.close()

    t_w_r_cal = t_w_c @ lidar_result.transform
    rot_err, trans_err = transform_error(t_w_r_cal, t_w_r)  # against the known mounting
    info = {
        "camera_reprojection_rms_px": float(camera_result.rms),
        "camera_views_used": float(len(corners)),
        "lidar_views_used": float(lidar_result.num_views),
        "lidar_point_to_plane_rms_m": float(lidar_result.rms),
        "lidar_rotation_error_deg": float(rot_err),
        "lidar_translation_error_m": float(trans_err),
    }
    return Extrinsics("calibrated", camera_result.intrinsics, t_w_c, t_w_r_cal, t_w_l, info)


def resolve_extrinsics(env: Environment, cfg: RunConfig) -> Extrinsics:
    """Sensor transforms and intrinsics for the chosen calibration mode.

    "nominal"       mounting geometry and nominal intrinsics (default, no ground truth read).
    "calibrated"    intrinsics and LiDAR extrinsic estimated by the calibration modules.
    "ground_truth"  transforms read from the MuJoCo model (debugging only).
    """
    env_cfg = env.config
    width, height = env_cfg.camera_resolution
    nominal_intrinsics = CameraIntrinsics.from_fovy(width, height, env_cfg.camera_fovy_deg)
    t_w_c, t_w_r, t_w_l = nominal_transforms(env_cfg)

    if cfg.calibration == "nominal":
        return Extrinsics("nominal", nominal_intrinsics, t_w_c, t_w_r, t_w_l)
    if cfg.calibration == "ground_truth":
        tree = ground_truth_frame_tree(env)
        return Extrinsics(
            "ground_truth",
            nominal_intrinsics,
            tree.get_transform("W", "C"),
            tree.get_transform("W", "R"),
            tree.get_transform("W", "L"),
        )
    return calibrate_sensors(env_cfg, cfg, nominal_intrinsics, t_w_c, t_w_r, t_w_l)


# --------------------------------------------------------------------- sensors/detectors
class SensorRig:
    """The sensors that are active in this run, created on the environment."""

    def __init__(self, env: Environment, cfg: RunConfig) -> None:
        self.rgb: Optional[RGBCamera] = None
        self.depth: Optional[DepthCamera] = None
        self.laser: Optional[LaserScanner] = None
        self.lidar: Optional[Lidar] = None
        self.last_rgb: Optional[np.ndarray] = None

        if "rgbd" in cfg.sensors or cfg.video_path:
            self.rgb = RGBCamera(env, seed=cfg.seed)
        if "rgbd" in cfg.sensors:
            noise_std, noise_per_m2 = RGBD_DEPTH_NOISE if cfg.noise else (0.0, 0.0)
            self.depth = DepthCamera(
                env, noise_std=noise_std, noise_std_per_m2=noise_per_m2, seed=cfg.seed + 1
            )
        if "laser" in cfg.sensors:
            self.laser = LaserScanner(env, laser_config(cfg), seed=cfg.seed + 2)
        if "lidar" in cfg.sensors:
            self.lidar = Lidar(env, lidar_config(cfg), seed=cfg.seed + 3)

    def close(self) -> None:
        for camera in (self.rgb, self.depth):
            if camera is not None:
                camera.close()


def build_detectors(cfg: RunConfig, env_cfg: EnvironmentConfig, ext: Extrinsics) -> Dict[str, object]:
    """One detector per active sensor, cropped to the arena."""
    size_x, size_y, size_z = env_cfg.arena_size
    bounds = dict(bounds_min=(-size_x / 2.0, -size_y / 2.0, 0.0), bounds_max=(size_x / 2.0, size_y / 2.0, size_z))
    detectors: Dict[str, object] = {}
    if "rgbd" in cfg.sensors:
        detectors["rgbd"] = RGBDDetector(
            ext.intrinsics,
            ext.t_w_c,
            default_detector_config(
                "rgbd", cluster=ClusterConfig(radius=0.08, min_points=10), ground_margin=0.08, **bounds
            ),
            stride=2,
        )
    if "laser" in cfg.sensors:
        detectors["laser"] = LaserDetector(ext.t_w_l, default_detector_config("laser", **bounds))
    if "lidar" in cfg.sensors:
        detectors["lidar"] = LidarDetector(ext.t_w_r, default_detector_config("lidar", **bounds))
    return detectors


def sense(name: str, env: Environment, rig: SensorRig, detectors: Dict[str, object], cfg: RunConfig) -> list:
    """Capture one measurement of a sensor at the current simulation time and detect objects."""
    if name == "rgbd":
        rgb = rig.rgb.capture()
        depth = rig.depth.capture()
        rig.last_rgb = rgb
        return detectors["rgbd"].detect(rgb, depth, env.time)
    if name == "laser":
        scan = rig.laser.scan_sweep() if cfg.laser_sweep else rig.laser.scan_line()
        return detectors["laser"].detect(scan)
    if name == "lidar":
        return detectors["lidar"].detect(rig.lidar.scan())
    raise ValueError(f"unknown sensor {name!r}")


# ------------------------------------------------------------------------------ logging
class RunLog:
    """Per-frame record of the tracker output and the measurements (for run_evaluation.py)."""

    def __init__(self) -> None:
        self.times: List[float] = []
        self.frames: List[Dict[int, tuple]] = []
        self.meas_time: List[float] = []
        self.meas_sensor: List[str] = []
        self.meas_position: List[np.ndarray] = []

    def add(self, time: float, tracks: Sequence, measurements: Sequence) -> None:
        """Record the alive tracks and the measurements used in one frame."""
        frame: Dict[int, tuple] = {}
        for track in tracks:
            state = np.concatenate(
                [np.asarray(track.position, dtype=float), np.asarray(track.velocity, dtype=float)]
            )
            code = STATUS_CODES.get(str(track.status.value), 0)
            p_3d = float(track.model_probabilities.get("CV3D", np.nan))
            frame[int(track.track_id)] = (state, code, p_3d, bool(track.updated))
        self.times.append(float(time))
        self.frames.append(frame)
        for m in measurements:
            self.meas_time.append(float(time))
            self.meas_sensor.append(str(m.sensor))
            self.meas_position.append(np.asarray(m.position, dtype=float))

    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Dense arrays: est_* have shape (T, K) / (T, K, 6) with one column per track id."""
        ids = sorted({track_id for frame in self.frames for track_id in frame})
        column = {track_id: k for k, track_id in enumerate(ids)}
        n_frames, n_tracks = len(self.frames), len(ids)
        est_states = np.full((n_frames, n_tracks, 6), np.nan)
        est_status = np.zeros((n_frames, n_tracks), dtype=np.int8)
        est_p3d = np.full((n_frames, n_tracks), np.nan)
        est_updated = np.zeros((n_frames, n_tracks), dtype=bool)
        for t, frame in enumerate(self.frames):
            for track_id, (state, code, p_3d, updated) in frame.items():
                k = column[track_id]
                est_states[t, k] = state
                est_status[t, k] = code
                est_p3d[t, k] = p_3d
                est_updated[t, k] = updated
        return {
            "times": np.asarray(self.times, dtype=float),
            "track_ids": np.asarray(ids, dtype=int),
            "est_states": est_states,
            "est_status": est_status,
            "est_model_prob_3d": est_p3d,
            "est_updated": est_updated,
            "meas_time": np.asarray(self.meas_time, dtype=float),
            "meas_sensor": np.asarray(self.meas_sensor, dtype="U16"),
            "meas_position": np.asarray(self.meas_position, dtype=float).reshape(-1, 3),
        }


def summarize(arrays: Dict[str, np.ndarray], gate: float = 0.5) -> Dict[str, float]:
    """Quick sanity numbers (nearest confirmed track per obstacle), not the evaluation metrics."""
    gt = arrays["gt_states"]
    est = arrays["est_states"]
    status = arrays["est_status"]
    errors: List[float] = []
    total = 0
    for t in range(gt.shape[0]):
        confirmed = status[t] == STATUS_CODES["confirmed"]
        for n in range(gt.shape[1]):
            total += 1
            if not confirmed.any():
                continue
            distance = float(np.linalg.norm(est[t, confirmed, :3] - gt[t, n, :3], axis=1).min())
            if distance <= gate:
                errors.append(distance)
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else float("nan")
    return {
        "frames": int(gt.shape[0]),
        "tracks_created": int(status.shape[1]),
        "tracks_ever_confirmed": int((status == STATUS_CODES["confirmed"]).any(axis=0).sum()),
        "coverage": float(len(errors) / total) if total else float("nan"),
        "nearest_track_rmse_m": rmse,
    }


class VideoSink:
    """Incremental video writer for dashboard frames (RGB uint8 arrays)."""

    def __init__(self, path: str, fps: float, fourcc: str = "mp4v") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.fourcc = fourcc
        self.frames = 0
        self._writer = None

    def write(self, rgb: np.ndarray) -> None:
        if self._writer is None:
            height, width = rgb.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*self.fourcc), self.fps, (width, height)
            )
            if not self._writer.isOpened():
                self._writer = None
                raise RuntimeError(f"could not open a video writer for {self.path} (fourcc {self.fourcc!r})")
        self._writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.frames += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


# --------------------------------------------------------------------------------- run
@dataclass
class RunResult:
    """What run() returns."""

    output_dir: Path
    log_path: Optional[Path]
    arrays: Dict[str, np.ndarray]
    summary: Dict[str, float]
    calibration: Dict[str, float]
    wall_time_s: float


def _advance(env: Environment, steps: int, viewer) -> None:
    """Advance the physics; take the viewer lock so the viewer never reads half-updated data."""
    if viewer is None:
        for _ in range(steps):
            env.step()
        return
    with viewer.lock():
        for _ in range(steps):
            env.step()


def draw_laser_rays(viewer, origin, points, max_rays=300, width=0.004, rgba=(1.0, 0.1, 0.1, 0.6)) -> None:
    """Draw a line from the laser to each detected point in the MuJoCo viewer."""
    scene = viewer.user_scn
    scene.ngeom = 0
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) > max_rays:  # keep the scene light
        pts = pts[np.linspace(0, len(pts) - 1, max_rays).astype(int)]
    for point in pts:
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
            np.asarray(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(origin, dtype=float), point)
        scene.ngeom += 1

def run(cfg: RunConfig) -> RunResult:
    """Run one simulation and return (and optionally save) the log."""
    cfg.validate()
    log_print = print if cfg.verbose else (lambda *args, **kwargs: None)
    wall_start = wall_clock.perf_counter()

    env_cfg = EnvironmentConfig(duration=cfg.duration)
    env = Environment(env_cfg)
    ext = resolve_extrinsics(env, cfg)
    log_print(f"[main] calibration mode: {ext.mode} {ext.info if ext.info else ''}")

    rig = SensorRig(env, cfg)
    detectors = build_detectors(cfg, env_cfg, ext)
    sensor_positions = ext.sensor_positions(cfg.sensors)
    fusion = MeasurementFusion(sensor_positions, covariance_config(cfg))
    rates_list = list(cfg.sensor_rates().values())
    if len(rates_list) == 1 or max(rates_list) - min(rates_list) < 1e-9:
        tentative_misses = 2
    else:
        tentative_misses = max(2, int(np.ceil(sum(rates_list) / min(rates_list))))
    tracker = ObjectTracker(
        make_filter_factory(kind=cfg.estimator, use_imm=cfg.use_imm),
        TrackerConfig(manager=TrackManagerConfig(max_tentative_misses=tentative_misses)),
    )
    ground_truth = GroundTruth(env)
    log = RunLog()

    dashboard = None
    video = None
    if cfg.video_path:
        from visualization.tracking_dashboard import DashboardConfig, TrackingDashboard

        size_x, size_y, size_z = env_cfg.arena_size
        dashboard = TrackingDashboard(
            ext.intrinsics,
            ext.t_w_c,
            bounds_min=(-size_x / 2.0, -size_y / 2.0, 0.0),
            bounds_max=(size_x / 2.0, size_y / 2.0, size_z),
            sensor_positions=sensor_positions,
            config=DashboardConfig(velocity_horizon=1.0),
        )
        video = VideoSink(cfg.video_path, fps=1.0 / cfg.video_period)

    viewer = None
    if cfg.viewer:
        import mujoco.viewer as mj_viewer

        viewer = mj_viewer.launch_passive(env.model, env.data)

    rates = cfg.sensor_rates()
    periods = {name: 1.0 / rate for name, rate in rates.items()}
    next_due = {name: 0.0 for name in cfg.sensors}
    next_video = 0.0
    steps_per_tick = env.steps_per_period(cfg.estimator_rate_hz)
    next_report = 5.0

    try:
        while env.time < cfg.duration - 1e-9:
            due = [name for name in cfg.sensors if env.time >= next_due[name] - 1e-9]
            if due:
                per_sensor: Dict[str, list] = {}
                detections: list = []
                for name in due:
                    per_sensor[name] = sense(name, env, rig, detectors, cfg)
                    if viewer is not None and "laser" in per_sensor:
                        dets = per_sensor["laser"]
                        laser_points = np.vstack([d.points for d in dets]) if dets else np.zeros((0, 3))
                        with viewer.lock():
                            draw_laser_rays(viewer, sensor_positions["laser"], laser_points)
                    detections.extend(per_sensor[name])
                    next_due[name] = next_due_time(next_due[name], periods[name], env.time)

                if cfg.fusion == "merged":
                    measurements = fusion.fuse_detections(detections)
                else:
                    measurements = fusion.measurements_from_detections(detections)

                tracker.update(measurements, time=env.time)
                ground_truth.record()
                log.add(env.time, tracker.tracks, measurements)

                if dashboard is not None and env.time >= next_video - 1e-9:
                    from visualization.tracking_dashboard import DashboardFrame

                    if rig.rgb is not None:
                        rig.last_rgb = rig.rgb.capture()
                    clouds = {n: np.vstack([d.points for d in dets]) for n, dets in per_sensor.items() if dets}
                    truth = {r.obstacle_id: r.position for r in ground_truth.snapshot()}
                    frame = DashboardFrame(
                        time=env.time,
                        image=rig.last_rgb,
                        tracks=tracker.tracks,
                        measurements=measurements,
                        clouds=clouds,
                        truth=truth,
                    )
                    video.write(dashboard.render(frame))
                    next_video = next_due_time(next_video, cfg.video_period, env.time)

            _advance(env, steps_per_tick, viewer)

            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                ahead = env.time / cfg.speed - (wall_clock.perf_counter() - wall_start)
                if ahead > 0.0:
                    wall_clock.sleep(ahead)

            if env.time >= next_report:
                log_print(
                    f"[main] t = {env.time:6.1f} s | confirmed tracks: {len(tracker.confirmed_tracks)}"
                )
                next_report += 5.0
    finally:
        rig.close()
        if video is not None:
            video.close()

    arrays = log.to_arrays()
    arrays["gt_ids"] = np.asarray(env.obstacles.ids, dtype="U8")
    arrays["gt_states"] = ground_truth.to_array()
    if len(ground_truth) != len(arrays["times"]):
        raise RuntimeError("ground-truth log and tracker log are out of sync")
    summary = summarize(arrays)

    out_dir = Path(cfg.output_dir)
    log_path: Optional[Path] = None
    wall_time = wall_clock.perf_counter() - wall_start
    if cfg.save_log:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "tracking_log.npz"
        np.savez_compressed(log_path, **arrays)
        ground_truth.save_csv(out_dir / "ground_truth.csv")
        info = {
            "config": asdict(cfg),
            "calibration": {"mode": ext.mode, **ext.info},
            "summary": summary,
            "wall_time_s": wall_time,
        }
        (out_dir / "run_info.json").write_text(json.dumps(info, indent=2, default=str))

    log_print(f"[main] done in {wall_time:.1f} s | summary: {summary}")
    if log_path is not None:
        log_print(f"[main] log saved to {log_path}")
    return RunResult(out_dir, log_path, arrays, summary, dict(ext.info), wall_time)


# ----------------------------------------------------------------------------------- CLI
def parse_args(argv: Optional[Sequence[str]] = None) -> RunConfig:
    defaults = RunConfig()
    parser = argparse.ArgumentParser(description="Multi-object 3D tracking simulation in MuJoCo.")
    parser.add_argument("--duration", type=float, default=defaults.duration, help="simulated time [s]")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--sensors", default=",".join(defaults.sensors), help="comma list of rgbd,laser,lidar")
    parser.add_argument("--estimator", choices=ESTIMATORS, default=defaults.estimator)
    parser.add_argument("--no-imm", action="store_true", help="use one CV3D filter instead of the IMM")
    parser.add_argument("--fusion", choices=FUSION_MODES, default=defaults.fusion)
    parser.add_argument("--calibration", choices=CALIBRATION_MODES, default=defaults.calibration)
    parser.add_argument("--calibration-views", type=int, default=defaults.calibration_views)
    parser.add_argument("--rgbd-rate", type=float, default=defaults.rgbd_rate_hz, help="RGB-D rate [Hz]")
    parser.add_argument("--laser-rate", type=float, default=defaults.laser_rate_hz, help="laser rate [Hz]")
    parser.add_argument("--lidar-rate", type=float, default=defaults.lidar_rate_hz, help="LiDAR rate [Hz]")
    parser.add_argument("--estimator-rate", type=float, default=defaults.estimator_rate_hz, help="main loop rate [Hz]")
    parser.add_argument("--laser-line", action="store_true", help="single laser line instead of a sweep")
    parser.add_argument("--no-noise", action="store_true", help="disable simulated sensor noise")
    parser.add_argument("--viewer", action="store_true", help="open the MuJoCo viewer")
    parser.add_argument("--speed", type=float, default=defaults.speed, help="viewer pacing factor")
    parser.add_argument("--video", default=None, help="write a dashboard video to this path (slow)")
    parser.add_argument("--video-period", type=float, default=defaults.video_period)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--no-save", action="store_true", help="do not write the log files")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        sensors = parse_sensors(args.sensors)
    except ValueError as exc:
        parser.error(str(exc))
    return RunConfig(
        duration=args.duration,
        seed=args.seed,
        sensors=sensors,
        estimator=args.estimator,
        use_imm=not args.no_imm,
        fusion=args.fusion,
        calibration=args.calibration,
        calibration_views=args.calibration_views,
        rgbd_rate_hz=args.rgbd_rate,
        laser_rate_hz=args.laser_rate,
        lidar_rate_hz=args.lidar_rate,
        estimator_rate_hz=args.estimator_rate,
        laser_sweep=not args.laser_line,
        noise=not args.no_noise,
        viewer=args.viewer,
        speed=args.speed,
        video_path=args.video,
        video_period=args.video_period,
        output_dir=args.output_dir,
        save_log=not args.no_save,
        verbose=not args.quiet,
    )

def relaunch_under_mjpython() -> None:
    """On macOS the passive viewer needs mjpython: re-run this script with it.

    The venv's mjpython is a script whose #! line breaks when the project path
    contains spaces, so it is started through the interpreter instead of exec'd directly.
    """
    if sys.platform != "darwin" or os.environ.get("MAIN_UNDER_MJPYTHON") == "1":
        return
    launcher = Path(sys.executable).parent / "mjpython"
    if not launcher.exists():
        found = shutil.which("mjpython")
        if found is None:
            raise RuntimeError("mjpython not found; reinstall the mujoco package")
        launcher = Path(found)

    with open(launcher, "rb") as f:
        first_line = f.readline()
    is_python_script = first_line.startswith(b"#")  # mjpython is a text (Python) script

    script = str(Path(__file__).resolve())
    if is_python_script:
        command = [sys.executable, str(launcher), script, *sys.argv[1:]]  # bypass the #! line
    else:
        command = [str(launcher), script, *sys.argv[1:]]  # real executable

    os.environ["MAIN_UNDER_MJPYTHON"] = "1"
    try:
        os.execv(command[0], command)
    except OSError as exc:
        raise RuntimeError(f"cannot start mjpython ({exc}); command was {command}") from exc

def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = parse_args(argv)
    if cfg.viewer and argv is None:  # only for real command-line runs
        relaunch_under_mjpython()
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())