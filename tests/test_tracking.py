"""Unit tests for the tracking package."""

from types import SimpleNamespace

import numpy as np
import pytest

from estimation import EKF, IMM, UKF
from fusion.measurement_fusion import Measurement
from perception.object_features import ColorDescriptor, hue_difference_deg
from tracking.data_association import (
    AssociationConfig,
    associate,
    association_cost,
    build_cost_matrix,
    hue_distance_deg,
)
from tracking.hungarian import solve_assignment
from tracking.object_tracker import ObjectTracker, TrackerConfig
from tracking.track_manager import (
    TrackManager,
    TrackManagerConfig,
    TrackStatus,
    blend_colors,
    make_filter_factory,
)

SIGMA_R = 0.03  # measurement standard deviation declared to the tracker
SIGMA_TRUE = 0.01  # actual noise of the simulated measurements (smaller, so the gate never misses)


# ------------------------------------------------------------------ helpers
def make_color(hue, saturation=0.8, value=0.8):
    return ColorDescriptor(
        chromaticity=np.full(3, 1.0 / 3.0),
        hue_deg=float(hue),
        saturation=saturation,
        value=value,
    )


def make_measurement(
    position, time=0.0, sigma=SIGMA_R, color=None, ambiguous=False, num_points=100,
    sensor="lidar", features=None,
):
    return Measurement(
        (sensor,), time, np.asarray(position, dtype=float), np.eye(3) * sigma**2,
        color=color, ambiguous=ambiguous, num_points=num_points, features=features,
    )


def noisy_measurement(position, time, rng, **kwargs):
    noisy = np.asarray(position, dtype=float) + SIGMA_TRUE * rng.standard_normal(3)
    return make_measurement(noisy, time, **kwargs)


def ekf_factory(accel_std=0.2):
    return make_filter_factory("ekf", use_imm=False, accel_std=accel_std)


class FakeTrack:
    """Minimal track for testing the association alone."""

    def __init__(self, position, sigma=0.05, color=None):
        self.position = np.asarray(position, dtype=float)
        self.cov = np.eye(3) * sigma**2
        self.color = color

    def gating_distance(self, z, r):
        diff = np.asarray(z, dtype=float) - self.position
        return float(diff @ np.linalg.solve(self.cov + np.asarray(r), diff))


# ---------------------------------------------------------------- Hungarian
class TestHungarian:
    def test_square_problem(self):
        result = solve_assignment([[1.0, 2.0], [2.0, 1.0]])
        assert result.matches == [(0, 0), (1, 1)]
        assert result.unmatched_rows == [] and result.unmatched_cols == []
        assert result.total_cost == pytest.approx(2.0)

    def test_global_optimum_beats_greedy(self):
        # Greedy would take (0, 0) = 1 first and be forced into (1, 1) = 100.
        result = solve_assignment([[1.0, 2.0], [1.0, 100.0]])
        assert result.matches == [(0, 1), (1, 0)]
        assert result.total_cost == pytest.approx(3.0)

    def test_forbidden_pairs_stay_unmatched(self):
        inf = np.inf
        result = solve_assignment([[1.0, inf], [inf, inf]])
        assert result.matches == [(0, 0)]
        assert result.unmatched_rows == [1]
        assert result.unmatched_cols == [1]

    def test_max_cost_gate(self):
        cost = [[1.0, 5.0], [5.0, 1.0]]
        assert solve_assignment(cost, max_cost=2.0).matches == [(0, 0), (1, 1)]
        none = solve_assignment(cost, max_cost=0.5)
        assert none.matches == [] and none.unmatched_rows == [0, 1] and none.unmatched_cols == [0, 1]

    def test_prefers_more_matches_over_lower_total_cost(self):
        inf = np.inf
        # Matching only (0, 0) or (1, 0) is cheaper, but two matches are possible.
        result = solve_assignment([[1.0, inf], [1.0, 5.0]])
        assert result.matches == [(0, 0), (1, 1)]
        assert result.total_cost == pytest.approx(6.0)

    def test_rectangular_problems(self):
        rows_more = solve_assignment([[1.0, 9.0], [9.0, 1.0], [5.0, 5.0]])
        assert rows_more.matches == [(0, 0), (1, 1)]
        assert rows_more.unmatched_rows == [2] and rows_more.unmatched_cols == []
        cols_more = solve_assignment([[1.0, 9.0, 9.0], [9.0, 9.0, 1.0]])
        assert cols_more.matches == [(0, 0), (1, 2)]
        assert cols_more.unmatched_cols == [1]

    def test_negative_costs(self):
        inf = np.inf
        result = solve_assignment([[-5.0, inf], [inf, -1.0]])
        assert result.matches == [(0, 0), (1, 1)]
        assert result.total_cost == pytest.approx(-6.0)

    def test_empty_inputs(self):
        no_rows = solve_assignment(np.zeros((0, 3)))
        assert no_rows.matches == [] and no_rows.unmatched_cols == [0, 1, 2]
        no_cols = solve_assignment(np.zeros((2, 0)))
        assert no_cols.matches == [] and no_cols.unmatched_rows == [0, 1]

    def test_all_forbidden(self):
        result = solve_assignment(np.full((2, 2), np.inf))
        assert result.matches == [] and result.total_cost == 0.0

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            solve_assignment([[np.nan, 1.0], [1.0, 1.0]])
        with pytest.raises(ValueError):
            solve_assignment([1.0, 2.0, 3.0])


# ---------------------------------------------------------- data association
class TestDataAssociation:
    def test_hue_distance(self):
        assert hue_distance_deg(make_color(0.0), make_color(240.0)) == pytest.approx(120.0, abs=1e-6)
        assert hue_distance_deg(make_color(350.0), make_color(10.0)) == pytest.approx(20.0, abs=1e-6)
        assert hue_distance_deg(None, make_color(10.0)) is None
        assert hue_distance_deg(make_color(10.0), None) is None
        assert hue_distance_deg(make_color(np.nan), make_color(10.0)) is None  # grey: no hue

    def test_cost_inside_and_outside_the_gate(self):
        track = FakeTrack([0.0, 0.0, 0.0], sigma=0.05)
        inside = make_measurement([0.05, 0.0, 0.0], sigma=0.02)
        assert association_cost(track, inside) == pytest.approx(0.05**2 / (0.05**2 + 0.02**2))
        assert association_cost(track, make_measurement([0.3, 0.0, 0.0], sigma=0.02)) == np.inf

    def test_color_adds_a_penalty_and_a_hard_gate(self):
        track = FakeTrack([0.0, 0.0, 0.0], sigma=0.05, color=make_color(0.0))
        base = association_cost(track, make_measurement([0.05, 0.0, 0.0], sigma=0.02, color=make_color(0.0)))
        near = association_cost(track, make_measurement([0.05, 0.0, 0.0], sigma=0.02, color=make_color(30.0)))
        far = association_cost(track, make_measurement([0.05, 0.0, 0.0], sigma=0.02, color=make_color(90.0)))
        assert near == pytest.approx(base + 2.0)  # color_weight * (30 / 30)^2
        assert far == np.inf  # above max_hue_diff_deg

    def test_color_can_be_disabled_and_missing_colors_are_ignored(self):
        track = FakeTrack([0.0, 0.0, 0.0], sigma=0.05, color=make_color(0.0))
        measurement = make_measurement([0.05, 0.0, 0.0], sigma=0.02, color=make_color(90.0))
        no_color = AssociationConfig(use_color=False)
        assert np.isfinite(association_cost(track, measurement, no_color))
        colorless = make_measurement([0.05, 0.0, 0.0], sigma=0.02, color=None)
        grey = make_measurement([0.05, 0.0, 0.0], sigma=0.02, color=make_color(np.nan))
        base = 0.05**2 / (0.05**2 + 0.02**2)
        assert association_cost(track, colorless) == pytest.approx(base)
        assert association_cost(track, grey) == pytest.approx(base)

    def test_cost_matrix(self):
        tracks = [FakeTrack([0.0, 0.0, 0.0]), FakeTrack([1.0, 0.0, 0.0])]
        ms = [
            make_measurement([1.01, 0.0, 0.0], sigma=0.02),
            make_measurement([0.01, 0.0, 0.0], sigma=0.02),
            make_measurement([5.0, 5.0, 5.0], sigma=0.02),
        ]
        cost = build_cost_matrix(tracks, ms)
        assert cost.shape == (2, 3)
        assert np.isfinite(cost[0, 1]) and np.isfinite(cost[1, 0])
        assert np.isinf(cost[0, 0]) and np.isinf(cost[1, 1])
        assert np.all(np.isinf(cost[:, 2]))

    def test_associate_matches_swapped_order_and_reports_unmatched(self):
        tracks = [FakeTrack([0.0, 0.0, 0.0]), FakeTrack([1.0, 0.0, 0.0]), FakeTrack([9.0, 9.0, 9.0])]
        ms = [
            make_measurement([1.01, 0.0, 0.0], sigma=0.02),
            make_measurement([0.01, 0.0, 0.0], sigma=0.02),
            make_measurement([5.0, 5.0, 5.0], sigma=0.02),
        ]
        result = associate(tracks, ms)
        assert sorted(result.matches) == [(0, 1), (1, 0)]
        assert result.unmatched_tracks == [2]
        assert result.unmatched_measurements == [2]
        assert result.cost.shape == (3, 3)

    def test_associate_with_no_tracks_or_no_measurements(self):
        ms = [make_measurement([0.0, 0.0, 0.0])]
        no_tracks = associate([], ms)
        assert no_tracks.matches == [] and no_tracks.unmatched_measurements == [0]
        no_meas = associate([FakeTrack([0.0, 0.0, 0.0])], [])
        assert no_meas.matches == [] and no_meas.unmatched_tracks == [0]

    def test_color_prevents_an_identity_swap_between_close_objects(self):
        red, blue = make_color(0.0), make_color(240.0)
        tracks = [FakeTrack([0.0, 0.0, 0.0], color=red), FakeTrack([0.03, 0.0, 0.0], color=blue)]
        ms = [
            make_measurement([0.02, 0.0, 0.0], sigma=0.02, color=red),
            make_measurement([0.01, 0.0, 0.0], sigma=0.02, color=blue),
        ]
        assert sorted(associate(tracks, ms).matches) == [(0, 0), (1, 1)]
        # Position alone prefers the swapped (wrong) assignment.
        position_only = associate(tracks, ms, AssociationConfig(use_color=False))
        assert sorted(position_only.matches) == [(0, 1), (1, 0)]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"gate_chi2": 0.0},
            {"color_weight": -1.0},
            {"hue_scale_deg": 0.0},
            {"max_hue_diff_deg": 0.0},
            {"max_hue_diff_deg": 200.0},
        ],
    )
    def test_invalid_config_raises(self, kwargs):
        with pytest.raises(ValueError):
            AssociationConfig(**kwargs)


# ------------------------------------------------------------ filter factory
class TestFilterFactory:
    POS = np.array([0.1, 0.2, 0.3])
    COV = np.eye(3) * 0.01

    def test_single_filter_kinds(self):
        ekf = make_filter_factory("ekf", use_imm=False)(self.POS, self.COV, 1.5)
        assert isinstance(ekf, EKF) and ekf.time == pytest.approx(1.5)
        assert np.allclose(ekf.x[:3], self.POS) and np.allclose(ekf.x[3:], 0.0)
        ukf = make_filter_factory("UKF", use_imm=False)(self.POS, self.COV, 0.0)
        assert isinstance(ukf, UKF)

    def test_imm_kinds(self):
        imm = make_filter_factory("ekf")(self.POS, self.COV, 2.0)
        assert isinstance(imm, IMM) and imm.time == pytest.approx(2.0)
        assert imm.model_names == ["CV2D", "CV3D"]
        assert isinstance(make_filter_factory("ukf")(self.POS, self.COV, 0.0).filters[0], UKF)

    def test_invalid_arguments_raise(self):
        with pytest.raises(ValueError):
            make_filter_factory("particle")
        with pytest.raises(ValueError):
            make_filter_factory("ekf", velocity_std=0.0)


# ------------------------------------------------------------------- colours
class TestBlendColors:
    def test_none_handling(self):
        color = make_color(10.0)
        assert blend_colors(None, color, 0.3) is color
        assert blend_colors(color, None, 0.3) is color
        assert blend_colors(None, None, 0.3) is None

    def test_hue_is_blended_on_the_circle(self):
        out = blend_colors(make_color(0.0), make_color(60.0), 0.5)
        assert out.hue_deg == pytest.approx(30.0, abs=1e-6)
        wrapped = blend_colors(make_color(350.0), make_color(10.0), 0.5)
        assert hue_difference_deg(wrapped.hue_deg, 0.0) < 1e-6

    def test_other_fields_are_linear(self):
        old = ColorDescriptor(np.array([0.6, 0.2, 0.2]), 0.0, 1.0, 0.5)
        new = ColorDescriptor(np.array([0.2, 0.6, 0.2]), 0.0, 0.5, 1.0)
        out = blend_colors(old, new, 0.5)
        assert np.allclose(out.chromaticity, [0.4, 0.4, 0.2])
        assert out.saturation == pytest.approx(0.75) and out.value == pytest.approx(0.75)

    def test_missing_hue_falls_back_to_the_available_one(self):
        out = blend_colors(make_color(np.nan), make_color(120.0), 0.5)
        assert out.hue_deg == pytest.approx(120.0)
        assert np.isnan(blend_colors(make_color(np.nan), make_color(np.nan), 0.5).hue_deg)


# ------------------------------------------------------------- track manager
def make_manager(**kwargs):
    config = TrackManagerConfig(**{**dict(max_position_std=1e3), **kwargs})
    return TrackManager(ekf_factory(), config)


class TestTrackManager:
    def test_create_track(self):
        manager = make_manager()
        m = make_measurement([0.5, 1.0, 1.5], 0.0, color=make_color(0.0))
        track = manager.create_track(m, 0.0)
        assert track.track_id == 1 and track.status is TrackStatus.TENTATIVE and track.hits == 1
        assert np.allclose(track.position, [0.5, 1.0, 1.5]) and np.allclose(track.velocity, 0.0)
        assert track.speed == pytest.approx(0.0)
        assert track.color is m.color
        assert track.filter.time == pytest.approx(0.0)
        assert track.position_std == pytest.approx(SIGMA_R)
        assert track.motion_model == "CV3D" and track.model_probabilities == {"CV3D": 1.0}
        assert manager.tracks == [track] and manager.get(1) is track and manager.get(99) is None

    def test_confirmation_after_confirm_hits(self):
        manager = make_manager(confirm_hits=3)
        track = manager.create_track(make_measurement([0.0, 0.0, 1.0], 0.0), 0.0)
        for k in (1, 2):
            t = 0.1 * k
            manager.predict(t)
            manager.register_hit(track, make_measurement([0.0, 0.0, 1.0], t))
            manager.finalize(t)
            expected = TrackStatus.TENTATIVE if k == 1 else TrackStatus.CONFIRMED
            assert track.status is expected
        assert track.confirmed_time == pytest.approx(0.2)
        assert manager.confirmed_tracks == [track] and manager.tentative_tracks == []

    def test_confirm_hits_one_confirms_immediately(self):
        manager = make_manager(confirm_hits=1)
        track = manager.create_track(make_measurement([0.0, 0.0, 1.0], 0.0), 0.0)
        assert track.status is TrackStatus.CONFIRMED

    def test_tentative_track_is_deleted_after_too_many_misses(self):
        manager = make_manager(max_tentative_misses=2)
        track = manager.create_track(make_measurement([0.0, 0.0, 1.0], 0.0), 0.0)
        for k in (1, 2):
            manager.predict(0.1 * k)
            manager.register_miss(track)
            assert manager.finalize(0.1 * k) == []
        assert track in manager.tracks
        manager.predict(0.3)
        manager.register_miss(track)
        assert manager.finalize(0.3) == [track]
        assert track.status is TrackStatus.DELETED and track.delete_reason == "tentative_misses"
        assert manager.tracks == [] and manager.deleted_tracks == [track]

    def test_ids_are_never_reused(self):
        manager = make_manager(max_tentative_misses=0)
        first = manager.create_track(make_measurement([0.0, 0.0, 1.0], 0.0), 0.0)
        manager.register_miss(first)
        manager.finalize(0.0)
        second = manager.create_track(make_measurement([1.0, 0.0, 1.0], 0.0), 0.0)
        assert (first.track_id, second.track_id) == (1, 2)
        assert manager.num_created == 2

    def test_confirmed_track_becomes_lost_and_is_deleted_by_timeout(self):
        manager = make_manager(confirm_hits=1, lost_after_time=0.5, delete_after_time=3.0)
        track = manager.create_track(make_measurement([0.0, 0.0, 1.0], 0.0), 0.0)
        manager.predict(0.4)
        manager.finalize(0.4)
        assert track.status is TrackStatus.CONFIRMED
        manager.predict(0.6)
        manager.finalize(0.6)
        assert track.status is TrackStatus.LOST and manager.lost_tracks == [track]
        manager.predict(2.9)
        assert manager.finalize(2.9) == []
        manager.predict(3.1)
        assert manager.finalize(3.1) == [track]
        assert track.status is TrackStatus.DELETED and track.delete_reason == "lost_timeout"

    def test_lost_track_is_reacquired(self):
        manager = make_manager(confirm_hits=1, lost_after_time=0.5)
        track = manager.create_track(make_measurement([0.0, 0.0, 1.0], 0.0), 0.0)
        manager.register_miss(track)
        manager.predict(0.6)
        manager.finalize(0.6)
        assert track.status is TrackStatus.LOST
        manager.predict(0.7)
        manager.register_hit(track, make_measurement([0.0, 0.0, 1.0], 0.7))
        assert track.status is TrackStatus.CONFIRMED
        assert track.reacquisitions == 1 and track.consecutive_misses == 0

    def test_uncertain_track_is_deleted(self):
        manager = make_manager(max_position_std=0.5)
        track = manager.create_track(make_measurement([0.0, 0.0, 1.0], 0.0), 0.0)
        manager.predict(1.0)  # velocity_std = 1 m/s -> about 1 m of position uncertainty
        assert manager.finalize(1.0) == [track]
        assert track.delete_reason == "uncertain"

    def test_secondary_hit_does_not_count_as_a_hit(self):
        manager = make_manager()
        track = manager.create_track(make_measurement([0.0, 0.0, 1.0], 0.0), 0.0)
        manager.register_hit(track, make_measurement([0.01, 0.0, 1.0], 0.0), secondary=True)
        assert track.hits == 1 and track.updated

    def test_predict_advances_filters_and_clears_the_update_flag(self):
        manager = make_manager()
        track = manager.create_track(make_measurement([0.0, 0.0, 1.0], 0.0), 0.0)
        assert track.updated
        manager.predict(1.0)
        assert track.filter.time == pytest.approx(1.0) and not track.updated

    def test_predict_backwards_raises(self):
        manager = make_manager()
        manager.create_track(make_measurement([0.0, 0.0, 1.0], 1.0), 1.0)
        with pytest.raises(ValueError):
            manager.predict(0.5)

    def test_color_is_smoothed_by_hits(self):
        manager = make_manager(color_alpha=0.5, confirm_hits=1)
        track = manager.create_track(make_measurement([0.0, 0.0, 1.0], 0.0, color=make_color(0.0)), 0.0)
        manager.predict(0.1)
        manager.register_hit(track, make_measurement([0.0, 0.0, 1.0], 0.1, color=make_color(60.0)))
        assert track.color.hue_deg == pytest.approx(30.0, abs=1e-6)

    def test_shape_hint_follows_the_majority(self):
        manager = make_manager(confirm_hits=1)
        sphere = SimpleNamespace(shape_hint="sphere")
        other = SimpleNamespace(shape_hint="other")
        track = manager.create_track(make_measurement([0.0, 0.0, 1.0], 0.0, features=sphere), 0.0)
        assert track.shape_hint == "sphere"
        for features in (other, other):
            manager.register_hit(track, make_measurement([0.0, 0.0, 1.0], 0.0, features=features), secondary=True)
        assert track.shape_hint == "other"

    def test_history_and_trajectory(self):
        manager = make_manager()
        track = manager.create_track(make_measurement([0.0, 0.0, 1.0], 0.0), 0.0)
        assert track.age == 0
        track.record(0.0)
        manager.predict(0.1)
        track.record(0.1)
        times, positions = track.trajectory()
        assert track.age == 2 and np.allclose(times, [0.0, 0.1]) and positions.shape == (2, 3)
        assert track.history[0].updated and not track.history[1].updated

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"confirm_hits": 0},
            {"max_tentative_misses": -1},
            {"lost_after_time": 0.0},
            {"lost_after_time": 2.0, "delete_after_time": 1.0},
            {"max_position_std": 0.0},
            {"color_alpha": 1.5},
        ],
    )
    def test_invalid_config_raises(self, kwargs):
        with pytest.raises(ValueError):
            TrackManagerConfig(**kwargs)


# -------------------------------------------------------------- object tracker
class TestObjectTracker:
    def test_tracks_a_constant_velocity_object(self):
        rng = np.random.default_rng(0)
        tracker = ObjectTracker(ekf_factory())
        p0, v = np.array([-1.0, 0.0, 1.0]), np.array([0.5, 0.2, 0.0])
        ids = set()
        for k in range(31):
            t = 0.1 * k
            confirmed = tracker.update([noisy_measurement(p0 + v * t, t, rng)], t)
            if k >= 2:  # creation + 2 hits = 3 hits
                assert len(confirmed) == 1
                ids.add(confirmed[0].track_id)
        assert ids == {1}
        track = tracker.confirmed_tracks[0]
        assert np.linalg.norm(track.position - (p0 + v * 3.0)) < 0.05
        assert np.linalg.norm(track.velocity - v) < 0.15
        times, positions = track.trajectory()
        assert track.age == 31 and len(times) == 31 and positions.shape == (31, 3)

    def test_two_crossing_objects_keep_their_ids(self):
        rng = np.random.default_rng(1)
        tracker = ObjectTracker(ekf_factory())
        a0, va = np.array([-1.0, 0.0, 1.0]), np.array([0.5, 0.0, 0.0])
        b0, vb = np.array([1.0, 0.3, 1.0]), np.array([-0.5, 0.0, 0.0])

        def nearest_object(track, truths):
            return int(np.argmin([np.linalg.norm(track.position - p) for p in truths]))

        early = None
        for k in range(41):
            t = 0.1 * k
            ms = [noisy_measurement(a0 + va * t, t, rng), noisy_measurement(b0 + vb * t, t, rng)]
            tracker.update([ms[i] for i in rng.permutation(2)], t)  # arbitrary order every frame
            if k == 5:
                truths = [a0 + va * t, b0 + vb * t]
                early = {tr.track_id: nearest_object(tr, truths) for tr in tracker.confirmed_tracks}
        truths = [a0 + va * 4.0, b0 + vb * 4.0]
        late = {tr.track_id: nearest_object(tr, truths) for tr in tracker.confirmed_tracks}
        assert set(early) == {1, 2} and set(late) == {1, 2}
        assert set(early.values()) == {0, 1}
        assert early == late  # every id still follows the same object after the crossing
        assert len(tracker.all_tracks) == 2

    def test_occluded_object_is_lost_then_reacquired_with_the_same_id(self):
        rng = np.random.default_rng(2)
        config = TrackerConfig(manager=TrackManagerConfig(lost_after_time=0.5, delete_after_time=3.0))
        tracker = ObjectTracker(ekf_factory(0.3), config)
        p0, v = np.array([-1.0, 0.0, 1.0]), np.array([0.4, 0.1, 0.0])
        for k in range(31):
            t = 0.1 * k
            tracker.update([noisy_measurement(p0 + v * t, t, rng)], t)
        track_id = tracker.confirmed_tracks[0].track_id

        statuses = []
        for k in range(31, 41):  # one second without any measurement
            tracker.update([], 0.1 * k)
            statuses.append(tracker.get(track_id).status)
        assert statuses[0] is TrackStatus.CONFIRMED
        assert statuses[-1] is TrackStatus.LOST
        assert tracker.confirmed_tracks == []

        t = 4.1
        tracker.update([noisy_measurement(p0 + v * t, t, rng)], t)
        track = tracker.get(track_id)
        assert track.status is TrackStatus.CONFIRMED and track.reacquisitions == 1
        assert len(tracker.all_tracks) == 1  # no duplicate track was started

    def test_new_object_starts_tentative_and_is_confirmed_later(self):
        rng = np.random.default_rng(3)
        tracker = ObjectTracker(ekf_factory())
        a, b = np.array([-1.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0])
        for k in range(10):
            tracker.update([noisy_measurement(a, 0.1 * k, rng)], 0.1 * k)
        for k in range(10, 13):
            t = 0.1 * k
            tracker.update([noisy_measurement(a, t, rng), noisy_measurement(b, t, rng)], t)
            if k == 10:
                assert tracker.last_result.created == [2]
            if k == 11:
                assert tracker.get(2).status is TrackStatus.TENTATIVE
        assert tracker.get(2).status is TrackStatus.CONFIRMED
        assert tracker.get(1).status is TrackStatus.CONFIRMED

    def test_clutter_never_becomes_a_confirmed_track(self):
        rng = np.random.default_rng(4)
        tracker = ObjectTracker(ekf_factory())
        a = np.array([-1.0, 0.0, 1.0])
        for k in range(12):
            t = 0.1 * k
            ms = [noisy_measurement(a, t, rng)]
            if k == 5:
                ms.append(make_measurement([1.5, -1.0, 1.5], t))  # a single spurious detection
            tracker.update(ms, t)
            if k == 7:
                assert tracker.get(2) is not None and tracker.get(2).status is TrackStatus.TENTATIVE
        assert tracker.get(2) is None
        deleted = [tr for tr in tracker.all_tracks if tr.status is TrackStatus.DELETED]
        assert len(deleted) == 1 and deleted[0].delete_reason == "tentative_misses"
        assert [tr.track_id for tr in tracker.confirmed_tracks] == [1]

    def test_ambiguous_measurements_do_not_start_tracks_by_default(self):
        tracker = ObjectTracker(ekf_factory())
        tracker.update([make_measurement([0.0, 0.0, 1.0], 0.0, ambiguous=True)], 0.0)
        assert tracker.tracks == [] and tracker.last_result.dropped == [0]
        allowing = ObjectTracker(ekf_factory(), TrackerConfig(spawn_ambiguous=True))
        allowing.update([make_measurement([0.0, 0.0, 1.0], 0.0, ambiguous=True)], 0.0)
        assert len(allowing.tracks) == 1

    def test_two_measurements_of_one_object_do_not_create_a_duplicate_track(self):
        rng = np.random.default_rng(5)
        tracker = ObjectTracker(ekf_factory())
        p = np.array([0.0, 0.0, 1.0])
        for k in range(6):
            t = 0.1 * k
            tracker.update(
                [
                    noisy_measurement(p, t, rng, sensor="rgbd"),
                    noisy_measurement(p + [0.01, 0.0, 0.0], t, rng, sensor="lidar"),
                ],
                t,
            )
            assert len(tracker.all_tracks) == 1
            if k >= 1:
                assert len(tracker.last_result.secondary) == 1
        assert tracker.get(1).hits == 6  # one hit per frame, however many sensors reported

    def test_secondary_updates_can_be_disabled(self):
        rng = np.random.default_rng(6)
        tracker = ObjectTracker(ekf_factory(), TrackerConfig(allow_secondary_updates=False))
        p = np.array([0.0, 0.0, 1.0])
        tracker.update(
            [noisy_measurement(p, 0.0, rng), noisy_measurement(p + [0.01, 0.0, 0.0], 0.0, rng)], 0.0
        )
        assert len(tracker.all_tracks) == 2

    def test_color_keeps_ids_of_two_close_stationary_objects(self):
        rng = np.random.default_rng(7)
        tracker = ObjectTracker(ekf_factory())
        red_pos, blue_pos = np.array([0.0, 0.0, 1.0]), np.array([0.03, 0.0, 1.0])
        red, blue = make_color(0.0), make_color(240.0)
        for k in range(30):
            t = 0.1 * k
            ms = [
                noisy_measurement(red_pos, t, rng, color=red),
                noisy_measurement(blue_pos, t, rng, color=blue),
            ]
            tracker.update([ms[i] for i in rng.permutation(2)], t)
        assert len(tracker.all_tracks) == 2
        for track in tracker.confirmed_tracks:
            is_red = hue_difference_deg(track.color.hue_deg, 0.0) < 1.0
            truth = red_pos if is_red else blue_pos
            assert np.linalg.norm(track.position - truth) < 0.015

    def test_default_imm_tracker_selects_the_3d_model_for_vertical_motion(self):
        rng = np.random.default_rng(8)
        tracker = ObjectTracker()  # default factory: IMM over EKFs
        p0, v = np.array([-0.5, 0.0, 0.5]), np.array([0.3, 0.2, 1.0])
        for k in range(61):
            t = 0.05 * k
            tracker.update([noisy_measurement(p0 + v * t, t, rng)], t)
        track = tracker.confirmed_tracks[0]
        assert track.motion_model == "CV3D"
        assert sum(track.model_probabilities.values()) == pytest.approx(1.0)
        assert np.linalg.norm(track.position - (p0 + v * 3.0)) < 0.1

    def test_frame_result_reports_what_happened(self):
        tracker = ObjectTracker(ekf_factory())
        tracker.update([make_measurement([0.0, 0.0, 1.0], 0.0)], 0.0)
        assert tracker.last_result.created == [1] and tracker.last_result.matches == []
        tracker.update([make_measurement([0.01, 0.0, 1.0], 0.1)], 0.1)
        assert tracker.last_result.matches == [(1, 0)] and tracker.last_result.created == []
        tracker.update([], 0.2)
        assert tracker.last_result.missed == [1] and tracker.last_result.time == pytest.approx(0.2)

    def test_time_defaults_to_the_latest_measurement_time(self):
        tracker = ObjectTracker(ekf_factory())
        assert tracker.time is None
        tracker.update([make_measurement([0.0, 0.0, 1.0], 0.5)])
        assert tracker.time == pytest.approx(0.5)

    def test_invalid_time_handling_raises(self):
        tracker = ObjectTracker(ekf_factory())
        with pytest.raises(ValueError):
            tracker.update([])  # no measurements and no time
        tracker.update([make_measurement([0.0, 0.0, 1.0], 1.0)], 1.0)
        with pytest.raises(ValueError):
            tracker.update([], 0.5)  # backwards in time

    def test_reset_clears_the_tracks_and_the_id_counter(self):
        tracker = ObjectTracker(ekf_factory())
        tracker.update([make_measurement([0.0, 0.0, 1.0], 0.0)], 0.0)
        tracker.reset()
        assert tracker.tracks == [] and tracker.time is None and tracker.last_result is None
        tracker.update([make_measurement([1.0, 0.0, 1.0], 0.0)], 0.0)
        assert tracker.tracks[0].track_id == 1