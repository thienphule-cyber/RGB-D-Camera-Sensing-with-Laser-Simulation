# Multi-Sensor 3D Dynamic Obstacle Localization and Tracking (MuJoCo)

A simulation and evaluation framework for localizing and tracking several independently moving
obstacles with a **fixed sensor assembly** (RGB-D camera, laser line scanner and LiDAR) inside a
MuJoCo scene. Some obstacles move in the XY plane (2D), others move freely in XYZ (3D). The
pipeline estimates the position, velocity and identity of every obstacle with EKF/UKF filters and
an IMM (2D/3D motion models), and an evaluation suite compares sensors, estimators and motion
models under noise, dropouts and occlusion.

The project is built to answer research questions, not only to track objects: every stage can be
switched on or off (sensor ablation, EKF vs UKF, IMM vs a single model) and every run is logged
for offline evaluation.

## Highlights

- **Simulation**: scripted obstacles on analytic trajectories, so ground-truth position *and*
  velocity are exact.
- **Sensors**: RGB camera, depth camera, ray-cast LiDAR and a simulated laser line scanner
  (laser plane, camera visibility check, distance-dependent triangulation noise), each with its
  own rate.
- **Calibration**: Zhang camera calibration, laser-plane calibration and checkerboard-plane
  LiDAR-to-camera extrinsic calibration (closed form + robust point-to-plane refinement).
- **Perception**: world-frame point clouds, ground removal, Euclidean clustering, per-object
  features (centroid, bounding box, sphere fit, colour descriptor).
- **Fusion**: line-of-sight measurement covariances, covariance inflation, covariance intersection
  for correlated sensors, cross-sensor association.
- **Estimation**: constant-velocity 2D/3D models, EKF, UKF, IMM, and a spherical (range / azimuth /
  elevation) measurement model.
- **Tracking**: Mahalanobis gating + Hungarian assignment (with a colour cue), track life cycle
  (tentative, confirmed, lost, deleted) and re-acquisition after occlusion.
- **Evaluation**: position / velocity RMSE, MOTA, MOTP, IDF1, ID switches, fragmentation,
  coverage, ghost tracks, IMM model-identification accuracy, computation time.
- **Visualization**: camera overlays, 3D scene, trajectories and a four-panel tracking dashboard
  (also as a video).
- **Tests**: unit tests for every package.

## Architecture

```text
 MuJoCo scene (scripted obstacles, exact ground truth)
        │
        ▼
 Sensors ── RGB-D (30 Hz) · laser scanner (20 Hz) · LiDAR (10 Hz)
        │
        ▼
 Perception ── world point cloud → ground removal → Euclidean clustering → Detection
        │
        ▼
 Fusion ── Detection → Measurement (covariance, inflation, covariance intersection)
        │
        ▼
 Tracking ── association (Mahalanobis gate + Hungarian) → EKF/UKF (+ IMM) → track life cycle
        │
        ▼
 Log (tracking_log.npz) ──► Evaluation (E01–E08) ──► CSV tables + figures
```

Ground truth is **never** used by the detectors, the fusion or the tracker. It is only logged
(and drawn as a reference marker in the dashboard). The one exception is the
`--calibration ground_truth` mode, which exists for debugging.

### Coordinate frames

| Frame | Symbol | Description |
|-------|--------|-------------|
| World | `W` | MuJoCo global frame (z up) |
| Sensor assembly | `B` | fixed mounting body |
| Camera | `C` | RGB-D optical frame (OpenCV: x right, y down, z forward) |
| Laser | `L` | x forward, y left, z up |
| LiDAR | `R` | x forward, y left, z up |

Transforms are 4x4 matrices `T_a_b` with `p_a = T_a_b @ p_b`. MuJoCo cameras look along their
local -Z axis; `sensors.common.MUJOCO_TO_OPENCV` converts between the two conventions.

### State and motion models

Every obstacle uses one state vector `[px, py, pz, vx, vy, vz]`.

- **CV3D**: free 3D constant-velocity motion with white-noise-acceleration process noise.
- **CV2D**: planar motion; `z` is held and `vz` is forced to zero.
- **IMM**: a mixture of CV2D and CV3D, so the filter does not need to know in advance which
  obstacles are planar.

## Simulation scene

A 5 x 5 x 3 m arena. The sensor assembly stands at the edge of the arena and looks along world +Y.
Six obstacles follow scripted trajectories (no random collisions), so the exact velocity is known.

| ID | Geometry | Motion | Trajectory |
|----|----------|--------|------------|
| O001 | box | 2D | linear, bouncing in XY |
| O002 | sphere | 3D | Lissajous in XYZ |
| O003 | cylinder | 2D | circle in XY |
| O004 | box | 3D | linear, bouncing in XYZ |
| O005 | sphere | 3D | circle in XY with a vertical oscillation |
| O006 | cylinder | 2D | variable-velocity Lissajous in XY |

Physics runs at 500 Hz (0.002 s step). Default sensor rates: RGB-D 30 Hz, laser 20 Hz, LiDAR
10 Hz, main loop 100 Hz.

## Installation

Python 3.10 – 3.12 is recommended (developed and tested on 3.12).

```bash
git clone <repo-url>
cd <repo-folder>
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Verify the installation:

```bash
pytest
```

## Quick start

```bash
# 20 s simulation with all sensors, EKF + IMM (log files go to outputs/run/)
python main.py --duration 20

# Sensor ablation: LiDAR only, single CV3D filter
python main.py --sensors lidar --no-imm --duration 20

# UKF, with the calibration routines run first
python main.py --estimator ukf --calibration calibrated

# Dashboard video (slow) and MuJoCo viewer
python main.py --duration 20 --video outputs/run/dashboard.mp4
python main.py --viewer          # macOS: main.py restarts itself under `mjpython`

# Run the evaluation suite (smoke test first, then the full one)
python run_evaluation.py --quick
python run_evaluation.py
```

### `main.py` options

| Option | Default | Description |
|--------|---------|-------------|
| `--duration` | 60 | simulated time [s] |
| `--seed` | 0 | seed of the sensor noise |
| `--sensors` | `rgbd,laser,lidar` | comma list of active sensors |
| `--estimator` | `ekf` | `ekf` or `ukf` |
| `--no-imm` | off | one CV3D filter instead of the 2D/3D IMM |
| `--fusion` | `sequential` | `sequential` (inflated covariances) or `merged` (cross-sensor fusion) |
| `--calibration` | `nominal` | `nominal` (mounting geometry), `calibrated` (run the calibration), `ground_truth` (debug) |
| `--calibration-views` | 12 | checkerboard views used by `calibrated` |
| `--rgbd-rate`, `--laser-rate`, `--lidar-rate` | 30 / 20 / 10 | sensor rates [Hz] |
| `--estimator-rate` | 100 | main-loop rate [Hz] |
| `--laser-line` | off | single laser line instead of a sweep |
| `--no-noise` | off | disable simulated sensor noise |
| `--viewer`, `--speed` | off, 1.0 | MuJoCo viewer and its pacing factor |
| `--video`, `--video-period` | none, 0.2 | dashboard video path and frame period [s] |
| `--output-dir` | `outputs/run` | where the logs are written |
| `--no-save`, `--quiet` | off | skip the log files / silence the console |

Output of a run (in `--output-dir`):

- `tracking_log.npz`: `times`, `gt_ids`, `gt_states` `(T, N, 6)`, `track_ids`,
  `est_states` `(T, K, 6)`, `est_status` `(T, K)` (0 none, 1 tentative, 2 confirmed, 3 lost),
  `est_model_prob_3d`, `est_updated`, and the raw measurements.
- `ground_truth.csv`: ground-truth log.
- `run_info.json`: configuration, calibration info and a quick summary.

### `run_evaluation.py` options

| Option | Default | Description |
|--------|---------|-------------|
| `--experiments` | `all` | `all` or a comma list such as `E04,E06` |
| `--duration` | 30 (10 with `--quick`) | simulated seconds per run |
| `--quick` | off | short runs and fewer calibration seeds |
| `--seed` | 0 | random seed |
| `--gate` | 0.5 | matching gate [m] |
| `--results-dir` | `results` | output folder |
| `--save-run-logs` | off | keep every `tracking_log.npz` |

## Experiments

| ID | Experiment | Question |
|----|-----------|----------|
| E01 | Calibration validation | how do camera / LiDAR calibration errors depend on the number of board views? |
| E02 | Noise-free baseline | what is the accuracy floor of the pipeline, per obstacle? |
| E03 | Multi-object tracking | how well are all six obstacles localized and tracked? |
| E04 | RGB-D vs laser vs LiDAR | what does each sensor contribute on its own? |
| E05 | Sensor fusion | does combining sensors help? |
| E06 | EKF vs UKF | accuracy and computation cost of the two estimators |
| E07 | 2D vs 3D motion models | IMM (2D + 3D) versus a single CV3D model, by true motion class |
| E08 | Robustness | sensor noise x2 / x4, 20 % / 40 % lost detections, an occluded zone |

Metrics are computed from the logs: confirmed tracks are matched to the ground-truth obstacles
frame by frame (CLEAR-MOT style, gate 0.5 m). Localization errors are computed on matched pairs
only; coverage, identity switches and fragmentation measure tracking quality.

Outputs in `results/`: `runs_summary.csv`, `per_object_metrics.csv`,
`baseline_error_timeseries.csv`, one `E0x_<name>.csv` per experiment, `E01_calibration_*.csv`,
`run_config.json` and figures in `results/figures/` (error bars, CDFs, trajectories, track
timeline, IMM model probability, noise sweep...).

> Each call overwrites `runs_summary.csv` and `per_object_metrics.csv`. If you run the
> experiments in separate groups, use the per-experiment `E0x_*.csv` files.

## Project layout

```text
.
├── main.py                  # run one simulation (CLI)
├── run_evaluation.py        # run experiments E01–E08, write tables and figures
├── simulation/              # MuJoCo scene, obstacles, analytic trajectories, ground truth
├── sensors/                 # RGB, depth, LiDAR, laser scanner, ray casting
├── calibration/             # frames, checkerboard scene, intrinsics, laser plane, LiDAR extrinsics
├── perception/              # point clouds, clustering, object features, detectors
├── fusion/                  # measurement covariances and multi-sensor fusion
├── estimation/              # motion/measurement models, EKF, UKF, IMM
├── tracking/                # Hungarian assignment, data association, track manager, tracker
├── visualization/           # camera overlays, 3D views, trajectories, dashboard
├── evaluation/              # localization metrics, tracking metrics, experiment definitions
├── tests/                   # unit tests (one file per package)
├── references.md
├── LICENSE.md
└── requirements.txt
```

## Testing

```bash
pytest                                              # all tests
pytest tests/test_estimation.py -v                  # one package
pytest --cov=estimation --cov-report=term-missing   # coverage
```

Tests that need an OpenGL context (offscreen rendering) skip themselves when none is available.

## Platform notes

- **macOS**: the `ARB_clip_control unavailable ... depth accuracy will be limited` warning is
  harmless. The MuJoCo viewer needs `mjpython`; `main.py --viewer` relaunches itself with it.
- **Linux / WSL2 (headless)**: set `export MUJOCO_GL=egl` before running (use `osmesa` if EGL
  fails).
- A full `run_evaluation.py` can take several minutes, and the noise x4 run is much slower than the
  others because the number of false tracks explodes. On macOS, one long suite crashed with a
  segmentation fault in a native library; running the experiment groups separately avoids it, for
  example `python run_evaluation.py --experiments E04,E05`.

## Known limitations

- **Fusion does not yet beat the best single sensor.** With the current settings, combining sensors
  creates duplicate tracks because the sensors see different parts of an object, so their
  centroids differ by a systematic offset. LiDAR-only or RGB-D-only tracking can score higher than
  the three-sensor configuration. Treat E05 as an honest negative result, not a proof that fusion
  is useless.
- **EKF and UKF give identical results** when the measurement is a Cartesian position and the motion
  model is linear. The spherical measurement model (`estimation.measurement_models`) is
  implemented and tested, but the pipeline does not feed it yet.
- Object shape is only classified as "sphere" or "other"; boxes and cylinders are not
  distinguished.
- A fixed viewpoint sees only the visible surface, so centroids are biased toward the sensors, and
  objects that touch or overlap can merge into one cluster.
- The laser scanner is modelled by ray casting. The laser-plane calibration and triangulation code
  is implemented and tested, but `main.py` does not use image-based triangulation.
- Evaluation runs use a single seed and report no confidence intervals.

## References

See [references.md](references.md).

## License

Released under the MIT License. See [LICENSE.md](LICENSE.md).