"""Run the evaluation experiments and write CSV tables and PNG figures to results/.

Usage:
    python run_evaluation.py                          # E01-E08, 30 s per run
    python run_evaluation.py --quick                  # 10 s per run (smoke test)
    python run_evaluation.py --experiments E04,E06 --duration 60
    python run_evaluation.py --save-run-logs          # also keep every tracking_log.npz

Outputs:
    results/runs_summary.csv, per_object_metrics.csv, baseline_error_timeseries.csv,
    results/E0x_<name>.csv, E01_calibration_*.csv, run_config.json
    results/figures/*.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time as wall_clock
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import main as pipeline
from evaluation.experiments import (
    EXPERIMENTS,
    RunRecord,
    run_calibration_study,
    run_suite,
    runs_needed,
)
from evaluation.localization_metrics import (
    DEFAULT_GATE,
    error_timeseries,
    obstacle_info,
    position_error_norms,
    position_error_vectors,
    velocity_error_norms,
)
from visualization.common import id_color, new_figure, save_figure, to_unit_rgb

BASELINE = "all_ekf_imm"


# ------------------------------------------------------------------------------ CSV
def _fmt(value):
    if isinstance(value, (float, np.floating)):
        return "" if not math.isfinite(float(value)) else f"{float(value):.6g}"
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def write_csv(path: Path, rows: Sequence[Dict[str, object]], columns: Optional[Sequence[str]] = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    columns = list(columns) if columns else list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k in columns})


def run_row(rec: RunRecord) -> Dict[str, object]:
    spec = rec.spec
    row: Dict[str, object] = {
        "run": spec.name,
        "sensors": "+".join(spec.sensors),
        "estimator": spec.estimator.upper(),
        "imm": spec.use_imm,
        "noise_scale": spec.noise_scale,
        "drop_prob": spec.drop_prob,
        "occlusion": spec.occlusion,
    }
    row.update(rec.metrics)
    return row


# ---------------------------------------------------------------------------- plots
def _finish(fig, path: Path) -> None:
    fig.tight_layout()
    save_figure(fig, path)


def grouped_bars(labels, series: Dict[str, Sequence[float]], title: str, ylabel: str, path: Path) -> None:
    fig = new_figure((max(6.0, 1.1 * len(labels) + 3.0), 4.6), dpi=120)
    ax = fig.add_subplot(111)
    x = np.arange(len(labels))
    width = 0.8 / max(len(series), 1)
    for i, (name, values) in enumerate(series.items()):
        vals = np.array([np.nan if v is None else v for v in values], dtype=float)
        ax.bar(x + (i - (len(series) - 1) / 2.0) * width, np.nan_to_num(vals), width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", linewidth=0.3)
    ax.legend(fontsize=8)
    _finish(fig, path)


def plot_error_cdf(records: Sequence[RunRecord], title: str, path: Path) -> None:
    fig = new_figure((7.0, 4.6), dpi=120)
    ax = fig.add_subplot(111)
    for rec in records:
        errors = position_error_norms(rec.arrays["gt_states"], rec.arrays["est_states"], rec.matching).ravel()
        errors = np.sort(errors[np.isfinite(errors)])
        if errors.size:
            ax.plot(errors, np.arange(1, errors.size + 1) / errors.size, label=rec.spec.name)
    ax.set_xlabel("position error [m]")
    ax.set_ylabel("cumulative fraction of matched frames")
    ax.set_title(title, fontsize=10)
    ax.grid(True, linewidth=0.3)
    ax.legend(fontsize=8)
    _finish(fig, path)


def plot_comparison(exp, records: Sequence[RunRecord], fig_dir: Path) -> None:
    """Standard set of comparison figures for a group of runs."""
    labels = [r.spec.name for r in records]
    prefix = fig_dir / exp.exp_id

    def col(key):
        return [r.metrics.get(key, float("nan")) for r in records]

    head = f"{exp.exp_id} {exp.title}"
    grouped_bars(labels, {"position RMSE [m]": col("pos_rmse_m"), "position p95 [m]": col("pos_p95_m"),
                        "velocity RMSE [m/s]": col("vel_rmse_mps")},
                f"{head}: localization error", "error", Path(f"{prefix}_localization.png"))
    grouped_bars(labels, {"MOTA": col("mota"), "IDF1": col("idf1"), "mean coverage": col("mean_coverage")},
                f"{head}: tracking quality", "score", Path(f"{prefix}_tracking.png"))
    grouped_bars(labels, {"ID switches": col("id_switches"), "fragmentations": col("fragmentations"),
                        "ghost tracks": col("ghost_tracks")},
                f"{head}: identity errors", "count", Path(f"{prefix}_identity.png"))
    grouped_bars(labels, {"wall time [s]": col("wall_time_s")},
                f"{head}: computation time", "seconds", Path(f"{prefix}_runtime.png"))
    plot_error_cdf(records, f"{head}: position error distribution", Path(f"{prefix}_error_cdf.png"))


def plot_motion_models(records: Sequence[RunRecord], fig_dir: Path) -> None:
    labels = [r.spec.name for r in records]
    grouped_bars(labels,
                {"2D objects": [r.metrics["pos_rmse_2d_m"] for r in records],
                "3D objects": [r.metrics["pos_rmse_3d_m"] for r in records]},
                "E07: position RMSE by true motion class", "position RMSE [m]", fig_dir / "E07_motion_class.png")
    grouped_bars(labels,
                {"2D objects": [r.metrics["model_acc_2d"] for r in records],
                "3D objects": [r.metrics["model_acc_3d"] for r in records]},
                "E07: IMM model identification accuracy (NaN = no IMM)", "fraction of frames",
                fig_dir / "E07_model_accuracy.png")


def plot_model_probability(rec: RunRecord, path: Path) -> None:
    """P(3D model) of the track matched to each obstacle (dashed = truly 2D motion)."""
    a = rec.arrays
    p3 = a["est_model_prob_3d"]
    if p3.shape[1] == 0 or len(a["times"]) == 0:
        return
    gt_ids = [str(i) for i in a["gt_ids"]]
    infos = obstacle_info(gt_ids)
    frames = np.arange(len(a["times"]))
    fig = new_figure((10.0, 4.6), dpi=120)
    ax = fig.add_subplot(111)
    for n, oid in enumerate(gt_ids):
        col = rec.matching.assign[:, n]
        series = np.where(col >= 0, p3[frames, np.clip(col, 0, None)], np.nan)
        ax.plot(a["times"], series, "--" if infos[oid].motion_dim == "2D" else "-",
                color=to_unit_rgb(id_color(n + 1)), lw=1.2, label=f"{oid} ({infos[oid].motion_dim})")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("P(3D model)")
    ax.set_title("IMM model probability of the matched track", fontsize=10)
    ax.grid(True, linewidth=0.3)
    ax.legend(fontsize=7, ncol=3)
    _finish(fig, path)


def plot_noise_sweep(records: Sequence[RunRecord], path: Path) -> None:
    sweep = sorted((r for r in records if r.spec.drop_prob == 0.0 and not r.spec.occlusion),
                   key=lambda r: r.spec.noise_scale)
    if len(sweep) < 2:
        return
    scales = [r.spec.noise_scale for r in sweep]
    fig = new_figure((10.0, 4.2), dpi=120)
    ax1, ax2 = fig.subplots(1, 2)
    ax1.plot(scales, [r.metrics["pos_rmse_m"] for r in sweep], "o-")
    ax1.set_xlabel("noise scale")
    ax1.set_ylabel("position RMSE [m]")
    ax2.plot(scales, [r.metrics["mota"] for r in sweep], "o-", label="MOTA")
    ax2.plot(scales, [r.metrics["idf1"] for r in sweep], "s-", label="IDF1")
    ax2.set_xlabel("noise scale")
    ax2.set_ylabel("score")
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        ax.grid(True, linewidth=0.3)
    fig.suptitle("E08: sensitivity to sensor noise", fontsize=10)
    _finish(fig, path)


def plot_per_object(records: Sequence[RunRecord], key: str, title: str, ylabel: str, path: Path) -> None:
    ids = [row["obstacle_id"] for row in records[0].per_object]
    series = {r.spec.name: [row[key] for row in r.per_object] for r in records}
    grouped_bars(ids, series, title, ylabel, path)


# ---- baseline run: trajectories, errors, identities
def _baseline_context(rec: RunRecord):
    a = rec.arrays
    gt_ids = [str(i) for i in a["gt_ids"]]
    gt, est = a["gt_states"], a["est_states"]
    err_vec = position_error_vectors(gt, est, rec.matching)
    est_pos = gt[:, :, :3] + err_vec  # NaN where the obstacle is not tracked
    return a["times"], gt_ids, obstacle_info(gt_ids), gt, est, est_pos


def plot_baseline_errors(rec: RunRecord, path: Path) -> None:
    times, gt_ids, infos, gt, est, _ = _baseline_context(rec)
    pe = position_error_norms(gt, est, rec.matching)
    ve = velocity_error_norms(gt, est, rec.matching)
    fig = new_figure((10.0, 7.0), dpi=120)
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    for n, oid in enumerate(gt_ids):
        color = to_unit_rgb(id_color(n + 1))
        label = f"{oid} ({infos[oid].motion_dim})"
        ax1.plot(times, pe[:, n], color=color, lw=1.0, label=label)
        ax2.plot(times, ve[:, n], color=color, lw=1.0)
    ax1.set_ylabel("position error [m]")
    ax2.set_ylabel("velocity error [m/s]")
    ax2.set_xlabel("time [s]")
    ax1.legend(fontsize=7, ncol=3)
    for ax in (ax1, ax2):
        ax.grid(True, linewidth=0.3)
    fig.suptitle("Baseline: estimation error over time (matched confirmed tracks)", fontsize=10)
    _finish(fig, path)


def plot_trajectory_xy(rec: RunRecord, path: Path) -> None:
    times, gt_ids, infos, gt, est, est_pos = _baseline_context(rec)
    fig = new_figure((7.0, 6.5), dpi=120)
    ax = fig.add_subplot(111)
    for n, oid in enumerate(gt_ids):
        color = to_unit_rgb(id_color(n + 1))
        ax.plot(gt[:, n, 0], gt[:, n, 1], "--", color=color, lw=0.8, alpha=0.6)
        ax.plot(est_pos[:, n, 0], est_pos[:, n, 1], "-", color=color, lw=1.4, label=oid)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Top view: truth (dashed) vs estimate (solid)", fontsize=10)
    ax.grid(True, linewidth=0.3)
    ax.legend(fontsize=7)
    _finish(fig, path)


def plot_trajectory_3d(rec: RunRecord, path: Path) -> None:
    times, gt_ids, infos, gt, est, est_pos = _baseline_context(rec)
    fig = new_figure((8.0, 7.0), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    for n, oid in enumerate(gt_ids):
        color = to_unit_rgb(id_color(n + 1))
        ax.plot(gt[:, n, 0], gt[:, n, 1], gt[:, n, 2], "--", color=color, lw=0.8, alpha=0.6)
        ax.plot(est_pos[:, n, 0], est_pos[:, n, 1], est_pos[:, n, 2], "-", color=color, lw=1.4, label=oid)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("3D trajectories: truth (dashed) vs estimate (solid)", fontsize=10)
    ax.legend(fontsize=7)
    _finish(fig, path)


def plot_xyz_vs_time(rec: RunRecord, path: Path) -> None:
    times, gt_ids, infos, gt, est, est_pos = _baseline_context(rec)
    ncols = 2
    nrows = int(math.ceil(len(gt_ids) / ncols))
    fig = new_figure((12.0, 3.0 * nrows), dpi=110)
    axes = fig.subplots(nrows, ncols, sharex=True, squeeze=False)
    for n, oid in enumerate(gt_ids):
        ax = axes[n // ncols][n % ncols]
        for c, (name, color) in enumerate(zip("xyz", ("tab:red", "tab:green", "tab:blue"))):
            ax.plot(times, gt[:, n, c], "--", color=color, lw=0.8)
            ax.plot(times, est_pos[:, n, c], "-", color=color, lw=1.2, label=name)
        ax.set_title(f"{oid} ({infos[oid].geometry}, {infos[oid].motion_dim})", fontsize=9)
        ax.grid(True, linewidth=0.3)
    for k in range(len(gt_ids), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    axes[0][0].legend(fontsize=7, ncol=3)
    fig.suptitle("Position components: truth (dashed) vs estimate (solid) [m]", fontsize=10)
    _finish(fig, path)


def plot_track_timeline(rec: RunRecord, path: Path) -> None:
    """Status of every track over time: shows ghost tracks and re-created tracks."""
    times = rec.arrays["times"]
    status = rec.arrays["est_status"].T  # (K, T)
    if status.size == 0:
        return
    cmap = ListedColormap(["white", "#bdbdbd", "#2ca02c", "#ff7f0e"])
    fig = new_figure((11.0, min(14.0, max(3.0, 0.12 * status.shape[0] + 1.5))), dpi=110)
    ax = fig.add_subplot(111)
    ax.imshow(status, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=3,
            extent=[times[0], times[-1], status.shape[0], 0])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("track (ordered by creation)")
    ax.set_title("Track life cycle", fontsize=10)
    ax.legend(handles=[Patch(color="#bdbdbd", label="tentative"), Patch(color="#2ca02c", label="confirmed"),
                    Patch(color="#ff7f0e", label="lost")], fontsize=7, loc="upper right")
    _finish(fig, path)


def plot_id_assignment(rec: RunRecord, path: Path) -> None:
    """Track id matched to each obstacle over time (a colour change is an identity switch)."""
    times = rec.arrays["times"]
    track_ids = rec.arrays["track_ids"]
    gt_ids = [str(i) for i in rec.arrays["gt_ids"]]
    assign = rec.matching.assign
    fig = new_figure((11.0, 3.8), dpi=120)
    ax = fig.add_subplot(111)
    for n in range(len(gt_ids)):
        idx = np.flatnonzero(assign[:, n] >= 0)
        if idx.size == 0:
            continue
        colors = [to_unit_rgb(id_color(int(track_ids[assign[t, n]]))) for t in idx]
        ax.scatter(times[idx], np.full(idx.size, n), c=colors, s=14, marker="|")
    ax.set_yticks(range(len(gt_ids)))
    ax.set_yticklabels(gt_ids)
    ax.set_xlabel("time [s]")
    ax.set_title("Matched track id per obstacle (colour = track id)", fontsize=10)
    _finish(fig, path)


# ---- E01
def aggregate_calibration(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    keys = ("camera_reprojection_rms_px", "lidar_point_to_plane_rms_m",
            "lidar_rotation_error_deg", "lidar_translation_error_m")
    summary: List[Dict[str, object]] = []
    for views in sorted({int(r["views"]) for r in rows}):
        ok = [r for r in rows if int(r["views"]) == views and r.get("ok") == 1]
        row: Dict[str, object] = {"views": views, "successful_runs": len(ok),
                                "total_runs": sum(1 for r in rows if int(r["views"]) == views)}
        for key in keys:
            values = np.array([float(r[key]) for r in ok], dtype=float) if ok else np.array([])
            row[f"{key}_mean"] = float(values.mean()) if values.size else float("nan")
            row[f"{key}_std"] = float(values.std()) if values.size else float("nan")
        summary.append(row)
    return summary


def plot_calibration(summary: Sequence[Dict[str, object]], path: Path) -> None:
    views = [r["views"] for r in summary]
    panels = (("camera_reprojection_rms_px", "camera reprojection RMS [px]"),
            ("lidar_rotation_error_deg", "LiDAR rotation error [deg]"),
            ("lidar_translation_error_m", "LiDAR translation error [m]"),
            ("lidar_point_to_plane_rms_m", "LiDAR point-to-plane RMS [m]"))
    fig = new_figure((10.0, 7.0), dpi=120)
    axes = fig.subplots(2, 2)
    for ax, (key, label) in zip(axes.ravel(), panels):
        mean = np.array([r[f"{key}_mean"] for r in summary], dtype=float)
        std = np.array([r[f"{key}_std"] for r in summary], dtype=float)
        ax.errorbar(views, mean, yerr=np.nan_to_num(std), marker="o", capsize=3)
        ax.set_xlabel("number of board views")
        ax.set_ylabel(label)
        ax.grid(True, linewidth=0.3)
    fig.suptitle("E01: calibration accuracy vs number of views (mean +/- std over seeds)", fontsize=10)
    _finish(fig, path)


# ------------------------------------------------------------------------------ main
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the tracking evaluation experiments.")
    parser.add_argument("--experiments", default="all", help="'all' or a comma list such as E03,E04")
    parser.add_argument("--duration", type=float, default=None,
                        help="simulated seconds per run (default 30, or 10 with --quick)")
    parser.add_argument("--quick", action="store_true", help="short runs and fewer calibration seeds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gate", type=float, default=DEFAULT_GATE, help="matching gate [m]")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--save-run-logs", action="store_true", help="keep tracking_log.npz of every run")
    return parser.parse_args(argv)


def parse_experiments(text: str) -> List[str]:
    if text.strip().lower() == "all":
        return list(EXPERIMENTS)
    chosen = [part.strip().upper() for part in text.split(",") if part.strip()]
    unknown = [c for c in chosen if c not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"unknown experiments {unknown}; choose from {list(EXPERIMENTS)}")
    return chosen


def print_summary(records: Dict[str, RunRecord]) -> None:
    print("\n" + f"{'run':<24}{'pos RMSE':>10}{'vel RMSE':>10}{'MOTA':>8}{'IDF1':>8}{'IDsw':>6}{'ghost':>7}{'wall[s]':>9}")
    for name, rec in records.items():
        m = rec.metrics
        print(f"{name:<24}{m['pos_rmse_m']:>10.3f}{m['vel_rmse_mps']:>10.3f}{m['mota']:>8.3f}"
            f"{m['idf1']:>8.3f}{m['id_switches']:>6d}{m['ghost_tracks']:>7d}{m['wall_time_s']:>9.1f}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    selected = parse_experiments(args.experiments)
    duration = args.duration if args.duration is not None else (10.0 if args.quick else 30.0)
    results = Path(args.results_dir)
    fig_dir = results / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    start = wall_clock.perf_counter()

    # E01: calibration study (no tracking run needed)
    if "E01" in selected:
        views = (4, 8, 12) if args.quick else (4, 6, 8, 12, 16)
        seeds = (0, 1) if args.quick else (0, 1, 2)
        rows = run_calibration_study(pipeline, views, seeds)
        write_csv(results / "E01_calibration_raw.csv", rows)
        summary = aggregate_calibration(rows)
        write_csv(results / "E01_calibration_summary.csv", summary)
        if any(r["successful_runs"] for r in summary):
            plot_calibration(summary, fig_dir / "E01_calibration.png")

    # Tracking experiments
    run_names = runs_needed([e for e in selected if e != "E01"])
    records = run_suite(pipeline, run_names, duration, args.seed,
                        results / "runs" if args.save_run_logs else None, args.gate)

    if records:
        write_csv(results / "runs_summary.csv", [run_row(r) for r in records.values()])
        write_csv(results / "per_object_metrics.csv",
                  [{"run": name, **row} for name, rec in records.items() for row in rec.per_object])

    for exp_id in selected:
        if exp_id == "E01":
            continue
        exp = EXPERIMENTS[exp_id]
        recs = [records[n] for n in exp.runs if n in records]
        if not recs:
            continue
        write_csv(results / f"{exp_id}_{exp.slug}.csv", [{"experiment": exp_id, **run_row(r)} for r in recs])
        if exp_id == "E02":
            plot_per_object(recs, "pos_rmse_m", "E02: position RMSE per obstacle", "position RMSE [m]",
                            fig_dir / "E02_per_object_position.png")
            plot_per_object(recs, "vel_rmse_mps", "E02: velocity RMSE per obstacle", "velocity RMSE [m/s]",
                            fig_dir / "E02_per_object_velocity.png")
        elif exp_id == "E03":
            plot_per_object(recs, "coverage", "E03: tracking coverage per obstacle", "fraction of frames tracked",
                            fig_dir / "E03_per_object_coverage.png")
            plot_per_object(recs, "pos_rmse_m", "E03: position RMSE per obstacle", "position RMSE [m]",
                            fig_dir / "E03_per_object_position.png")
        else:
            plot_comparison(exp, recs, fig_dir)
            if exp_id == "E07":
                plot_motion_models(recs, fig_dir)
            if exp_id == "E08":
                plot_noise_sweep(recs, fig_dir / "E08_noise_sweep.png")

    baseline = records.get(BASELINE)
    if baseline is not None:
        a = baseline.arrays
        gt_ids = [str(i) for i in a["gt_ids"]]
        write_csv(results / "baseline_error_timeseries.csv",
                error_timeseries(a["times"], gt_ids, a["track_ids"], a["gt_states"], a["est_states"],
                                baseline.matching))
        plot_baseline_errors(baseline, fig_dir / "baseline_error_vs_time.png")
        plot_trajectory_xy(baseline, fig_dir / "baseline_trajectory_xy.png")
        plot_trajectory_3d(baseline, fig_dir / "baseline_trajectory_3d.png")
        plot_xyz_vs_time(baseline, fig_dir / "baseline_xyz_vs_time.png")
        plot_track_timeline(baseline, fig_dir / "baseline_track_timeline.png")
        plot_id_assignment(baseline, fig_dir / "baseline_id_assignment.png")
        plot_model_probability(baseline, fig_dir / "baseline_imm_model_probability.png")

    elapsed = wall_clock.perf_counter() - start
    (results / "run_config.json").write_text(json.dumps(
        {"experiments": selected, "duration_s": duration, "seed": args.seed, "gate_m": args.gate,
        "runs": list(records), "elapsed_s": elapsed}, indent=2))
    print_summary(records)
    print(f"\n[eval] done in {elapsed:.0f} s. Results in {results.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())