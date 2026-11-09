"""Experiment definitions (E01-E08) and the machinery that runs them.

main.py is passed in as `pipeline` (dependency injection), so this package does not
import the simulation itself. Robustness perturbations (noise scaling, random detection
dropout, an occluded region) are applied by temporarily patching main.py's noise
constants and its sense() function; ground truth is never touched.
"""

from __future__ import annotations

import time as wall_clock
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .localization_metrics import (
    DEFAULT_GATE,
    Matching,
    localization_summary,
    match_tracks_to_truth,
    model_identification,
    obstacle_info,
    per_object_localization,
)
from .tracking_metrics import per_object_tracking, tracking_metrics

ALL_SENSORS = ("rgbd", "laser", "lidar")
# Detections whose XY position falls inside this box are removed (an invisible occluder).
OCCLUSION_ZONE = ((-0.6, 0.0), (0.6, 1.0))  # ((x_min, y_min), (x_max, y_max)) [m]


# --------------------------------------------------------------------------- specs
@dataclass(frozen=True)
class RunSpec:
    """One simulation configuration."""

    name: str
    sensors: Tuple[str, ...] = ALL_SENSORS
    estimator: str = "ekf"
    use_imm: bool = True
    noise_scale: float = 1.0  # multiplies the simulated sensor noise (0 = noise free)
    drop_prob: float = 0.0  # probability that a detection is lost
    occlusion: bool = False  # remove detections inside OCCLUSION_ZONE
    description: str = ""


def build_run_specs() -> Dict[str, RunSpec]:
    specs = [
        RunSpec("all_ekf_imm", description="RGB-D + laser + LiDAR, EKF + IMM (baseline, T6)"),
        RunSpec("all_ekf_imm_noiseless", noise_scale=0.0, description="baseline without sensor noise"),
        RunSpec("rgbd_only", sensors=("rgbd",), description="RGB-D only (T1)"),
        RunSpec("laser_only", sensors=("laser",), description="laser only (T2)"),
        RunSpec("lidar_only", sensors=("lidar",), description="LiDAR only (T3)"),
        RunSpec("rgbd_laser", sensors=("rgbd", "laser"), description="RGB-D + laser (T4)"),
        RunSpec("rgbd_lidar", sensors=("rgbd", "lidar"), description="RGB-D + LiDAR (T5)"),
        RunSpec("all_ukf_imm", estimator="ukf", description="all sensors, UKF + IMM (T7)"),
        RunSpec("all_ekf_cv3d", use_imm=False, description="all sensors, EKF with one CV3D model"),
        RunSpec("noise_x2", noise_scale=2.0, description="sensor noise x2"),
        RunSpec("noise_x4", noise_scale=4.0, description="sensor noise x4"),
        RunSpec("dropout_20", drop_prob=0.2, description="20% of the detections lost"),
        RunSpec("dropout_40", drop_prob=0.4, description="40% of the detections lost"),
        RunSpec("occlusion_zone", occlusion=True, description="detections inside a zone removed"),
    ]
    return {s.name: s for s in specs}


@dataclass(frozen=True)
class Experiment:
    exp_id: str
    slug: str
    title: str
    objective: str
    runs: Tuple[str, ...]


EXPERIMENTS: Dict[str, Experiment] = {
    e.exp_id: e
    for e in (
        Experiment("E01", "calibration", "Calibration validation",
                "Accuracy of camera / LiDAR calibration versus the number of board views", ()),
        Experiment("E02", "noise_free_baseline", "Noise-free baseline accuracy",
                "Per-object accuracy floor of the pipeline (no sensor noise) versus the noisy baseline",
                ("all_ekf_imm_noiseless", "all_ekf_imm")),
        Experiment("E03", "multi_object", "Multi-object localization and tracking",
                "Localization and tracking quality with all six obstacles", ("all_ekf_imm",)),
        Experiment("E04", "sensor_comparison", "RGB-D vs laser vs LiDAR",
                "Compare the sensors one by one", ("rgbd_only", "laser_only", "lidar_only")),
        Experiment("E05", "sensor_fusion", "Sensor fusion",
                "Benefit of combining sensors",
                ("rgbd_only", "laser_only", "lidar_only", "rgbd_laser", "rgbd_lidar", "all_ekf_imm")),
        Experiment("E06", "ekf_vs_ukf", "EKF vs UKF",
                "Accuracy and computation cost of the two estimators", ("all_ekf_imm", "all_ukf_imm")),
        Experiment("E07", "motion_models", "2D vs 3D motion models",
                "IMM (2D + 3D models) versus a single 3D model, by true motion class",
                ("all_ekf_imm", "all_ekf_cv3d")),
        Experiment("E08", "robustness", "Occlusion and noise robustness",
                "Behaviour under more noise, lost detections and an occluded region",
                ("all_ekf_imm_noiseless", "all_ekf_imm", "noise_x2", "noise_x4",
                    "dropout_20", "dropout_40", "occlusion_zone")),
    )
}


def runs_needed(exp_ids: Sequence[str]) -> List[str]:
    """Unique run names needed by the experiments, in order."""
    names: List[str] = []
    for exp_id in exp_ids:
        for name in EXPERIMENTS[exp_id].runs:
            if name not in names:
                names.append(name)
    return names


# --------------------------------------------------------------------- perturbations
def _in_zone(position, zone=OCCLUSION_ZONE) -> bool:
    (x0, y0), (x1, y1) = zone
    return bool(x0 <= position[0] <= x1 and y0 <= position[1] <= y1)


@contextmanager
def perturbed_pipeline(pipeline, spec: RunSpec, seed: int):
    """Temporarily apply the noise scale, detection dropout and occlusion of `spec`."""
    names = ("RGBD_DEPTH_NOISE", "LASER_NOISE", "LIDAR_RANGE_NOISE")
    originals = {name: getattr(pipeline, name) for name in names}
    original_sense = pipeline.sense
    rng = np.random.default_rng(seed + 1234)
    scale = float(spec.noise_scale)

    def scaled(value):
        return tuple(v * scale for v in value) if isinstance(value, tuple) else value * scale

    def perturbed_sense(name, env, rig, detectors, cfg):
        kept = []
        for det in original_sense(name, env, rig, detectors, cfg):
            if spec.occlusion and _in_zone(det.position):
                continue
            if spec.drop_prob > 0.0 and rng.random() < spec.drop_prob:
                continue
            kept.append(det)
        return kept

    try:
        for name in names:
            setattr(pipeline, name, scaled(originals[name]))
        pipeline.sense = perturbed_sense
        yield
    finally:
        for name, value in originals.items():
            setattr(pipeline, name, value)
        pipeline.sense = original_sense


# --------------------------------------------------------------------------- runs
@dataclass(eq=False)
class RunRecord:
    spec: RunSpec
    arrays: Dict[str, np.ndarray]
    metrics: Dict[str, float]
    per_object: List[Dict[str, object]]
    matching: Matching


def evaluate_arrays(arrays: Dict[str, np.ndarray], gate: float = DEFAULT_GATE):
    """Metrics, per-object table and matching of one run log (the arrays saved by main.py)."""
    gt_ids = [str(i) for i in arrays["gt_ids"]]
    infos = obstacle_info(gt_ids)
    gt, est, status = arrays["gt_states"], arrays["est_states"], arrays["est_status"]
    matching = match_tracks_to_truth(gt, est, status, gate)

    metrics: Dict[str, float] = {}
    metrics.update(localization_summary(gt_ids, gt, est, matching, infos))
    metrics.update(tracking_metrics(gt, est, status, matching, gate))
    metrics.update(model_identification(gt_ids, arrays["est_model_prob_3d"], matching, infos))

    loc_rows = per_object_localization(gt_ids, gt, est, matching, infos)
    trk_rows = per_object_tracking(gt_ids, matching)
    per_object = [{**a, **b} for a, b in zip(loc_rows, trk_rows)]
    return metrics, per_object, matching


def execute_run(pipeline, spec: RunSpec, duration: float, seed: int,
                output_root: Optional[Path] = None, gate: float = DEFAULT_GATE) -> RunRecord:
    """Run one configuration with main.py and evaluate it."""
    out_dir = str(Path(output_root) / spec.name) if output_root else "outputs/eval_tmp"
    cfg = pipeline.RunConfig(
        duration=duration,
        seed=seed,
        sensors=spec.sensors,
        estimator=spec.estimator,
        use_imm=spec.use_imm,
        output_dir=out_dir,
        save_log=output_root is not None,
        verbose=False,
    )
    with perturbed_pipeline(pipeline, spec, seed):
        result = pipeline.run(cfg)
    metrics, per_object, matching = evaluate_arrays(result.arrays, gate)
    metrics["wall_time_s"] = float(result.wall_time_s)
    metrics["frames"] = int(len(result.arrays["times"]))
    return RunRecord(spec, result.arrays, metrics, per_object, matching)


def run_suite(pipeline, run_names: Sequence[str], duration: float, seed: int,
            output_root: Optional[Path], gate: float, log: Callable[[str], None] = print) -> Dict[str, RunRecord]:
    """Run all named configurations; a failing run is reported and skipped."""
    specs = build_run_specs()
    records: Dict[str, RunRecord] = {}
    for i, name in enumerate(run_names, start=1):
        spec = specs[name]
        log(f"[eval] run {i}/{len(run_names)}: {name} - {spec.description}")
        start = wall_clock.perf_counter()
        try:
            records[name] = execute_run(pipeline, spec, duration, seed, output_root, gate)
        except Exception as exc:  # noqa: BLE001 - keep going with the other runs
            log(f"[eval]   FAILED: {type(exc).__name__}: {exc}")
            continue
        m = records[name].metrics
        log(f"[eval]   pos RMSE {m['pos_rmse_m']:.3f} m | MOTA {m['mota']:.3f} | IDF1 {m['idf1']:.3f} "
            f"| {wall_clock.perf_counter() - start:.0f} s")
    return records


# ------------------------------------------------------------------------ E01
def run_calibration_study(pipeline, views: Sequence[int] = (4, 6, 8, 12, 16), seeds: Sequence[int] = (0, 1, 2),
                        log: Callable[[str], None] = print) -> List[Dict[str, object]]:
    """Calibrate with a growing number of board views; returns one row per (views, seed)."""
    from simulation.environment import Environment

    rows: List[Dict[str, object]] = []
    for n_views in views:
        for seed in seeds:
            log(f"[eval] E01 calibration: {n_views} views, seed {seed}")
            cfg = pipeline.RunConfig(calibration="calibrated", calibration_views=int(n_views), seed=int(seed),
                                    duration=1.0, sensors=("lidar",), verbose=False)
            row: Dict[str, object] = {"views": int(n_views), "seed": int(seed)}
            try:
                ext = pipeline.resolve_extrinsics(Environment(), cfg)
                row.update(ext.info)
                row["ok"] = 1
            except Exception as exc:  # noqa: BLE001 - e.g. too few detected boards
                row["ok"] = 0
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    return rows