"""Evaluation package: localization / tracking metrics and the experiment suite."""

from .experiments import (
    EXPERIMENTS,
    RunRecord,
    RunSpec,
    build_run_specs,
    evaluate_arrays,
    execute_run,
    run_calibration_study,
    run_suite,
    runs_needed,
)
from .localization_metrics import (
    DEFAULT_GATE,
    Matching,
    error_timeseries,
    localization_summary,
    match_tracks_to_truth,
    model_identification,
    obstacle_info,
    per_object_localization,
    position_error_norms,
    position_error_vectors,
    rmse,
    velocity_error_norms,
    velocity_error_vectors,
)
from .tracking_metrics import per_object_tracking, tracking_metrics

__all__ = [
    "DEFAULT_GATE", "EXPERIMENTS", "Matching", "RunRecord", "RunSpec", "build_run_specs",
    "error_timeseries", "evaluate_arrays", "execute_run", "localization_summary",
    "match_tracks_to_truth", "model_identification", "obstacle_info", "per_object_localization",
    "per_object_tracking", "position_error_norms", "position_error_vectors", "rmse",
    "run_calibration_study", "run_suite", "runs_needed", "tracking_metrics",
    "velocity_error_norms", "velocity_error_vectors",
]