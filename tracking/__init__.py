"""Tracking package: assignment, data association, track management and the tracker."""

from .data_association import (
    AssociationConfig,
    AssociationResult,
    associate,
    association_cost,
    build_cost_matrix,
    hue_distance_deg,
)
from .hungarian import Assignment, solve_assignment
from .object_tracker import FrameResult, ObjectTracker, TrackerConfig
from .track_manager import (
    Track,
    TrackManager,
    TrackManagerConfig,
    TrackSample,
    TrackStatus,
    blend_colors,
    make_filter_factory,
)

__all__ = [
    "AssociationConfig",
    "AssociationResult",
    "associate",
    "association_cost",
    "build_cost_matrix",
    "hue_distance_deg",
    "Assignment",
    "solve_assignment",
    "FrameResult",
    "ObjectTracker",
    "TrackerConfig",
    "Track",
    "TrackManager",
    "TrackManagerConfig",
    "TrackSample",
    "TrackStatus",
    "blend_colors",
    "make_filter_factory",
]