"""Unit tests for the fusion package."""

from dataclasses import replace

import numpy as np
import pytest

from fusion.measurement_covariance import (
    CovarianceConfig,
    SensorNoise,
    default_sensor_noise,
    inflation_factor,
    line_of_sight,
    los_covariance,
    measurement_covariance,
    sensor_position_from_transform,
    uses_sphere_center,
)
from fusion.measurement_fusion import (
    FusionConfig,
    Measurement,
    MeasurementFusion,
    covariance_intersection,
    fuse_independent,
    mahalanobis_sq,
)
from perception.object_detection import Detection
from perception.object_features import color_descriptor, compute_object_features

# Explicit noise model so the expected values below do not depend on the defaults.
CFG = CovarianceConfig(
    sensors={
        "rgbd": SensorNoise(0.01, 0.002, 0.003),
        "laser": SensorNoise(0.01, 0.002, 0.004),
        "lidar": SensorNoise(0.02, 0.0, 0.01),
    },
    centroid_bias_sigma=0.05,
    sphere_bias_sigma=0.005,
    reference_points=50,
    ambiguous_inflation=9.0,
    min_sigma=1e-3,
)
SENSOR_POS = {
    "rgbd": (0.0, -3.0, 1.5),
    "laser": (0.25, -3.0, 1.38),
    "lidar": (0.0, -3.0, 1.62),
}
SENSOR = np.array(SENSOR_POS["rgbd"])
POS = np.array([0.0, 0.0, 1.5])  # 3 m straight ahead of the RGB-D camera (line of sight = +Y)


# ------------------------------------------------------------------ helpers
def make_cap(center, radius, n=3000, half_angle_deg=80.0, toward=(0.0, -1.0, 0.0)):
    """Points on the part of a sphere facing `toward` (a stand-in for the visible surface)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + np.sqrt(5.0)) * i
    dirs = np.column_stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    )
    axis = np.asarray(toward, dtype=float)
    axis = axis / np.linalg.norm(axis)
    keep = dirs @ axis >= np.cos(np.radians(half_angle_deg))
    return np.asarray(center, dtype=float) + radius * dirs[keep]


def make_detection(sensor, position, num_points=200, ambiguous=False, color=None, time=0.0, use_sphere=False):
    """Synthetic detection. use_sphere=True gives a sphere-centre position."""
    position = np.asarray(position, dtype=float)
    if use_sphere:
        pts = make_cap(position, 0.15)
        features = compute_object_features(pts)
        assert features.sphere is not None
        pos = features.sphere.center.copy()
    else:
        rng = np.random.default_rng(0)
        pts = position + rng.uniform(-0.05, 0.05, (num_points, 3))
        features = compute_object_features(pts, fit_sphere_model=False)
        pos = position.copy()
    return Detection(
        sensor=sensor, time=time, position=pos, features=features,
        color=color, ambiguous=ambiguous, points=pts,
    )


def make_fusion(sensors=("rgbd", "laser", "lidar"), covariance_config=CFG, config=None):
    return MeasurementFusion({s: SENSOR_POS[s] for s in sensors}, covariance_config, config)


# ---------------------------------------------------------- noise & covariance config
class TestSensorNoiseAndConfig:
    def test_range_and_cross_range_std(self):
        noise = SensorNoise(0.01, 0.002, 0.003)
        assert noise.range_std(3.0) == pytest.approx(0.01 + 0.002 * 9.0)
        assert noise.cross_range_std(3.0) == pytest.approx(0.009)

    def test_negative_noise_raises(self):
        with pytest.raises(ValueError):
            SensorNoise(-0.01)
        with pytest.raises(ValueError):
            SensorNoise(0.01, angular_sigma=-1.0)

    def test_default_sensor_noise_covers_all_sensors(self):
        assert set(default_sensor_noise()) == {"rgbd", "laser", "lidar"}
        assert set(CovarianceConfig().sensors) == {"rgbd", "laser", "lidar"}

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"centroid_bias_sigma": -0.1},
            {"sphere_bias_sigma": -0.1},
            {"reference_points": 0},
            {"ambiguous_inflation": 0.5},
            {"min_sigma": 0.0},
            {"centroid_shift": -0.1},
            {"sensors": {}},
        ],
    )
    def test_invalid_config_raises(self, kwargs):
        with pytest.raises(ValueError):
            CovarianceConfig(**kwargs)


# ----------------------------------------------------------------------- geometry
class TestGeometry:
    def test_line_of_sight(self):
        u, rng = line_of_sight(POS, SENSOR)
        assert np.allclose(u, [0.0, 1.0, 0.0])
        assert rng == pytest.approx(3.0)

    def test_line_of_sight_zero_distance_raises(self):
        with pytest.raises(ValueError):
            line_of_sight(SENSOR, SENSOR)

    def test_los_covariance_eigenstructure(self):
        cov = los_covariance([1.0, 0.0, 0.0], 0.1, 0.01)
        assert np.allclose(cov, np.diag([0.01, 1e-4, 1e-4]))

    def test_uses_sphere_center(self):
        sphere_det = make_detection("rgbd", POS, use_sphere=True)
        assert uses_sphere_center(sphere_det)
        centroid_det = Detection(
            sensor="rgbd", time=0.0, position=sphere_det.features.centroid,
            features=sphere_det.features, color=None, ambiguous=False, points=sphere_det.points,
        )
        assert not uses_sphere_center(centroid_det)  # sphere fit exists but the centroid was reported
        assert not uses_sphere_center(make_detection("rgbd", POS))

    def test_inflation_factor(self):
        assert inflation_factor(make_detection("rgbd", POS), CFG) == 1.0
        assert inflation_factor(make_detection("rgbd", POS, num_points=10), CFG) == pytest.approx(5.0)
        assert inflation_factor(make_detection("rgbd", POS, ambiguous=True), CFG) == pytest.approx(9.0)
        both = make_detection("rgbd", POS, num_points=10, ambiguous=True)
        assert inflation_factor(both, CFG) == pytest.approx(45.0)

    def test_sensor_position_from_transform(self):
        transform = np.eye(4)
        transform[:3, 3] = [1.0, 2.0, 3.0]
        assert np.allclose(sensor_position_from_transform(transform), [1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            sensor_position_from_transform(np.eye(3))


# ------------------------------------------------------------ measurement covariance
class TestMeasurementCovariance:
    def test_axes_follow_the_line_of_sight(self):
        cov = measurement_covariance(make_detection("rgbd", POS), SENSOR, CFG)
        sigma_los = np.hypot(0.01 + 0.002 * 9.0, 0.05)
        sigma_cross = 0.003 * 3.0
        assert np.allclose(cov, np.diag([sigma_cross**2, sigma_los**2, sigma_cross**2]), atol=1e-12)

    def test_general_direction(self):
        det = make_detection("lidar", [1.0, 0.5, 2.0])
        sensor = np.array(SENSOR_POS["lidar"])
        cov = measurement_covariance(det, sensor, CFG)
        u, rng = line_of_sight(det.position, sensor)
        sigma_los = np.hypot(0.02, 0.05)
        sigma_cross = 0.01 * rng
        assert u @ cov @ u == pytest.approx(sigma_los**2)
        perpendicular = np.cross(u, [0.0, 0.0, 1.0])
        perpendicular /= np.linalg.norm(perpendicular)
        assert perpendicular @ cov @ perpendicular == pytest.approx(sigma_cross**2)

    def test_symmetric_positive_definite(self):
        cov = measurement_covariance(make_detection("laser", [0.7, 1.0, 1.1]), SENSOR_POS["laser"], CFG)
        assert np.allclose(cov, cov.T)
        assert np.linalg.eigvalsh(cov).min() > 0.0

    def test_sphere_center_is_more_certain_along_the_line_of_sight(self):
        sphere_det = make_detection("rgbd", POS, use_sphere=True)
        centroid_det = make_detection("rgbd", POS)
        u, _ = line_of_sight(sphere_det.position, SENSOR)
        var_sphere = u @ measurement_covariance(sphere_det, SENSOR, CFG) @ u
        var_centroid = u @ measurement_covariance(centroid_det, SENSOR, CFG) @ u
        assert var_sphere < var_centroid

    def test_covariance_grows_with_range(self):
        near = measurement_covariance(make_detection("rgbd", [0.0, -1.0, 1.5]), SENSOR, CFG)
        far = measurement_covariance(make_detection("rgbd", [0.0, 2.0, 1.5]), SENSOR, CFG)
        assert np.trace(far) > np.trace(near)

    def test_few_points_inflate_the_covariance(self):
        base = measurement_covariance(make_detection("rgbd", POS), SENSOR, CFG)
        few = measurement_covariance(make_detection("rgbd", POS, num_points=10), SENSOR, CFG)
        assert np.allclose(few, 5.0 * base)

    def test_ambiguous_clusters_inflate_the_covariance(self):
        base = measurement_covariance(make_detection("rgbd", POS), SENSOR, CFG)
        ambiguous = measurement_covariance(make_detection("rgbd", POS, ambiguous=True), SENSOR, CFG)
        assert np.allclose(ambiguous, 9.0 * base)

    def test_min_sigma_floor(self):
        cfg = CovarianceConfig(
            sensors={"rgbd": SensorNoise(0.0)}, centroid_bias_sigma=0.0, min_sigma=1e-3
        )
        cov = measurement_covariance(make_detection("rgbd", POS), SENSOR, cfg)
        assert np.allclose(cov, 1e-6 * np.eye(3))

    def test_unknown_sensor_raises(self):
        with pytest.raises(KeyError):
            measurement_covariance(make_detection("sonar", POS), SENSOR, CFG)

    def test_detection_at_the_sensor_position_raises(self):
        with pytest.raises(ValueError):
            measurement_covariance(make_detection("rgbd", SENSOR), SENSOR, CFG)


# -------------------------------------------------------------------- Measurement
class TestMeasurement:
    def test_valid_measurement(self):
        m = Measurement(("rgbd",), 1.0, [0.0, 0.0, 1.0], np.eye(3) * 1e-4)
        assert m.sensor == "rgbd" and not m.is_fused
        assert m.position.shape == (3,) and m.covariance.shape == (3, 3)
        fused = Measurement(("rgbd", "lidar"), 1.0, [0.0, 0.0, 1.0], np.eye(3) * 1e-4)
        assert fused.sensor == "rgbd+lidar" and fused.is_fused

    def test_invalid_measurements_raise(self):
        cov = np.eye(3) * 1e-4
        with pytest.raises(ValueError):
            Measurement((), 0.0, [0, 0, 0], cov)
        with pytest.raises(ValueError):
            Measurement(("rgbd",), 0.0, [0, 0], cov)
        with pytest.raises(ValueError):
            Measurement(("rgbd",), 0.0, [0, 0, 0], np.eye(2))
        asymmetric = np.array([[1e-4, 5e-5, 0.0], [0.0, 1e-4, 0.0], [0.0, 0.0, 1e-4]])
        with pytest.raises(ValueError):
            Measurement(("rgbd",), 0.0, [0, 0, 0], asymmetric)
        with pytest.raises(ValueError):
            Measurement(("rgbd",), 0.0, [0, 0, 0], -np.eye(3))


# --------------------------------------------------------- fusion mathematics
class TestIndependentFusion:
    def test_equal_covariances_halve_the_covariance(self):
        cov = np.eye(3) * 1e-4
        mean, fused = fuse_independent([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [cov, cov])
        assert np.allclose(mean, [0.5, 0.0, 0.0])
        assert np.allclose(fused, 0.5 * cov)

    def test_mean_is_weighted_by_the_information(self):
        mean, _ = fuse_independent(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [np.eye(3) * 1e-4, np.eye(3) * 4e-4]
        )
        assert mean[0] == pytest.approx(0.2)

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            fuse_independent([], [])
        with pytest.raises(ValueError):
            fuse_independent([[0, 0, 0]], [np.eye(3), np.eye(3)])

    def test_mahalanobis_sq(self):
        d2 = mahalanobis_sq([0, 0, 0], np.eye(3) * 1e-2, [0.1, 0, 0], np.eye(3) * 1e-2)
        assert d2 == pytest.approx(0.5)


class TestCovarianceIntersection:
    A = np.diag([1e-4, 4e-4, 2e-4])
    B = np.diag([3e-4, 1e-4, 2e-4])
    ZA = np.array([0.0, 0.0, 0.0])
    ZB = np.array([0.02, -0.01, 0.0])

    def test_identical_estimates_keep_their_covariance(self):
        cov = np.diag([1e-4, 2e-4, 3e-4])
        z = np.array([1.0, 2.0, 3.0])
        mean, fused, weights = covariance_intersection([z, z], [cov, cov])
        assert np.allclose(mean, z)
        assert np.allclose(fused, cov)
        assert weights.sum() == pytest.approx(1.0)

    def test_never_more_confident_than_independent_fusion(self):
        _, ci_cov, _ = covariance_intersection([self.ZA, self.ZB], [self.A, self.B])
        _, ind_cov = fuse_independent([self.ZA, self.ZB], [self.A, self.B])
        assert np.linalg.eigvalsh(ci_cov - ind_cov).min() > -1e-12

    def test_trace_is_not_worse_than_the_best_input(self):
        _, ci_cov, _ = covariance_intersection([self.ZA, self.ZB], [self.A, self.B])
        assert np.trace(ci_cov) <= min(np.trace(self.A), np.trace(self.B)) + 1e-12

    def test_prefers_the_precise_estimate(self):
        precise, vague = np.eye(3) * 1e-4, np.eye(3)
        z1, z2 = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
        mean, fused, _ = covariance_intersection([z1, z2], [precise, vague])
        assert np.linalg.norm(mean - z1) < 1e-3
        assert np.trace(fused) <= np.trace(precise) + 1e-9

    def test_weights_form_a_convex_combination(self):
        _, _, weights = covariance_intersection([self.ZA, self.ZB], [self.A, self.B])
        assert np.all(weights >= 0.0) and np.all(weights <= 1.0)
        assert weights.sum() == pytest.approx(1.0)

    def test_single_estimate_is_returned_unchanged(self):
        mean, cov, weights = covariance_intersection([self.ZB], [self.A])
        assert np.allclose(mean, self.ZB) and np.allclose(cov, self.A)
        assert np.allclose(weights, [1.0])

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            covariance_intersection([], [])
        with pytest.raises(ValueError):
            covariance_intersection([self.ZA], [self.A, self.B])


# ------------------------------------------------------------- fusion configuration
class TestFusionConfig:
    def test_defaults(self):
        cfg = FusionConfig()
        assert cfg.correlation_groups == (("rgbd", "laser"),)
        assert cfg.correlated_inflation == 2.0

    def test_group_key(self):
        cfg = FusionConfig()
        assert cfg.group_key("rgbd") == cfg.group_key("laser")
        assert cfg.group_key("lidar") == "lidar"
        assert cfg.group_key("lidar") != cfg.group_key("rgbd")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"correlated_inflation": 0.5},
            {"gate_chi2": 0.0},
            {"max_time_gap": -1.0},
            {"correlation_groups": (("rgbd", "laser"), ("laser", "lidar"))},
        ],
    )
    def test_invalid_config_raises(self, kwargs):
        with pytest.raises(ValueError):
            FusionConfig(**kwargs)


# --------------------------------------------------------- detections -> measurements
class TestMeasurementFusionConversion:
    def test_measurement_copies_the_detection_fields(self):
        color = color_descriptor(np.tile(np.array([[200, 30, 30]], dtype=np.uint8), (10, 1)))
        det = make_detection("rgbd", POS, color=color, time=2.5)
        fusion = make_fusion(("rgbd", "lidar"))  # not a correlated pair, so no inflation
        m = fusion.measurement_from_detection(det)
        assert m.sensors == ("rgbd",) and m.time == 2.5
        assert np.allclose(m.position, det.position)
        assert np.allclose(m.covariance, measurement_covariance(det, SENSOR_POS["rgbd"], CFG))
        assert m.color is color and m.features is det.features
        assert m.num_points == det.num_points and m.ambiguous is False

    def test_correlated_sensors(self):
        assert make_fusion().correlated_sensors == frozenset({"rgbd", "laser"})
        assert make_fusion(("rgbd", "lidar")).correlated_sensors == frozenset()
        assert make_fusion(("rgbd",)).correlated_sensors == frozenset()

    def test_correlated_sensors_are_inflated(self):
        fusion = make_fusion()
        rgbd = make_detection("rgbd", POS)
        lidar = make_detection("lidar", POS)
        base_rgbd = measurement_covariance(rgbd, SENSOR_POS["rgbd"], CFG)
        base_lidar = measurement_covariance(lidar, SENSOR_POS["lidar"], CFG)
        assert np.allclose(fusion.measurement_from_detection(rgbd).covariance, 2.0 * base_rgbd)
        assert np.allclose(fusion.measurement_from_detection(lidar).covariance, base_lidar)
        no_inflation = fusion.measurement_from_detection(rgbd, inflate_correlated=False)
        assert np.allclose(no_inflation.covariance, base_rgbd)

    def test_centroid_shift_moves_only_centroid_positions(self):
        cfg = replace(CFG, centroid_shift=0.06)
        fusion = make_fusion(("lidar",), cfg)
        centroid_det = make_detection("lidar", POS)
        u, _ = line_of_sight(centroid_det.position, SENSOR_POS["lidar"])
        shifted = fusion.measurement_from_detection(centroid_det)
        assert np.allclose(shifted.position, centroid_det.position + 0.06 * u)
        sphere_det = make_detection("lidar", POS, use_sphere=True)
        assert np.allclose(fusion.measurement_from_detection(sphere_det).position, sphere_det.position)

    def test_measurements_from_detections_keep_the_order(self):
        fusion = make_fusion()
        dets = [make_detection("lidar", [0.0, 0.0, 1.5]), make_detection("lidar", [1.0, 0.5, 1.0])]
        out = fusion.measurements_from_detections(dets)
        assert len(out) == 2
        assert np.allclose(out[0].position, dets[0].position)
        assert np.allclose(out[1].position, dets[1].position)

    def test_inactive_sensor_raises(self):
        fusion = make_fusion(("rgbd",))
        with pytest.raises(KeyError):
            fusion.measurement_from_detection(make_detection("lidar", POS))

    def test_invalid_construction_raises(self):
        with pytest.raises(ValueError):
            MeasurementFusion({"sonar": (0.0, 0.0, 0.0)}, CFG)
        with pytest.raises(ValueError):
            MeasurementFusion({}, CFG)
        with pytest.raises(ValueError):
            MeasurementFusion({"rgbd": (0.0, 0.0)}, CFG)


# ------------------------------------------------------------------------- merging
class TestMeasurementFusionMerging:
    def test_single_sensor_passes_through(self):
        fusion = make_fusion()
        dets = [make_detection("rgbd", [-0.8, 0.0, 1.5]), make_detection("rgbd", [0.8, 0.0, 1.5])]
        out = fusion.fuse_detections(dets)
        assert len(out) == 2
        assert all(not m.is_fused for m in out)

    def test_same_object_from_two_sensors_is_merged(self):
        fusion = make_fusion()
        rgbd = make_detection("rgbd", POS, num_points=200)
        lidar = make_detection("lidar", POS + [0.01, 0.0, 0.01], num_points=200)
        out = fusion.fuse_detections([rgbd, lidar])
        assert len(out) == 1
        fused = out[0]
        assert fused.sensors == ("rgbd", "lidar") and fused.is_fused
        assert np.linalg.norm(fused.position - POS) < 0.05
        m_rgbd = fusion.measurement_from_detection(rgbd, inflate_correlated=False)
        m_lidar = fusion.measurement_from_detection(lidar, inflate_correlated=False)
        assert np.trace(fused.covariance) < min(np.trace(m_rgbd.covariance), np.trace(m_lidar.covariance))
        assert fused.num_points == rgbd.num_points + lidar.num_points

    def test_different_objects_are_not_merged(self):
        fusion = make_fusion()
        out = fusion.fuse_detections(
            [make_detection("rgbd", POS), make_detection("lidar", [1.5, 0.5, 1.0])]
        )
        assert len(out) == 2
        assert all(not m.is_fused for m in out)

    def test_time_gap_prevents_merging(self):
        fusion = make_fusion()
        out = fusion.fuse_detections(
            [make_detection("rgbd", POS, time=0.0), make_detection("lidar", POS, time=0.5)]
        )
        assert len(out) == 2

    def test_small_time_gap_merges_and_uses_the_latest_time(self):
        fusion = make_fusion()
        out = fusion.fuse_detections(
            [make_detection("rgbd", POS, time=1.000), make_detection("lidar", POS, time=1.005)]
        )
        assert len(out) == 1
        assert out[0].time == pytest.approx(1.005)

    def test_hungarian_assignment_handles_swapped_order(self):
        fusion = make_fusion()
        a, b = np.array([-0.8, 0.0, 1.5]), np.array([0.8, 0.0, 1.5])
        rgbd = [make_detection("rgbd", a), make_detection("rgbd", b)]
        lidar = [make_detection("lidar", b + 0.01), make_detection("lidar", a + 0.01)]  # swapped
        out = sorted(fusion.fuse_detections(rgbd + lidar), key=lambda m: m.position[0])
        assert len(out) == 2
        assert all(m.is_fused for m in out)
        assert np.linalg.norm(out[0].position - a) < 0.03
        assert np.linalg.norm(out[1].position - b) < 0.03

    def test_three_sensors_on_one_object(self):
        fusion = make_fusion()
        dets = [
            make_detection("rgbd", POS),
            make_detection("laser", POS + [0.005, 0.0, 0.0], num_points=10),
            make_detection("lidar", POS + [0.0, 0.0, 0.005]),
        ]
        out = fusion.fuse_detections(dets)
        assert len(out) == 1
        assert out[0].sensors == ("rgbd", "laser", "lidar")
        assert out[0].num_points == sum(d.num_points for d in dets)

    def test_ambiguous_only_if_every_member_is(self):
        fusion = make_fusion()
        mixed = fusion.fuse_detections(
            [make_detection("rgbd", POS, ambiguous=True), make_detection("lidar", POS)]
        )
        both = fusion.fuse_detections(
            [make_detection("rgbd", POS, ambiguous=True), make_detection("lidar", POS, ambiguous=True)]
        )
        assert mixed[0].ambiguous is False
        assert both[0].ambiguous is True

    def test_color_is_kept_from_the_sensor_that_has_it(self):
        fusion = make_fusion()
        color = color_descriptor(np.tile(np.array([[30, 30, 200]], dtype=np.uint8), (10, 1)))
        out = fusion.fuse_detections(
            [make_detection("rgbd", POS, color=color), make_detection("lidar", POS)]
        )
        assert out[0].color is color

    def test_correlated_group_is_merged_with_covariance_intersection(self):
        fusion = make_fusion()
        rgbd = make_detection("rgbd", POS)
        laser = make_detection("laser", POS + [0.005, 0.0, 0.005])
        fused = fusion.fuse_detections([rgbd, laser])[0]
        m1 = fusion.measurement_from_detection(rgbd, inflate_correlated=False)
        m2 = fusion.measurement_from_detection(laser, inflate_correlated=False)
        _, independent = fuse_independent([m1.position, m2.position], [m1.covariance, m2.covariance])
        assert np.trace(fused.covariance) >= np.trace(independent) - 1e-12
        assert np.trace(fused.covariance) <= min(np.trace(m1.covariance), np.trace(m2.covariance)) + 1e-12

    def test_fuse_detections_matches_the_manual_pipeline(self):
        fusion = make_fusion()
        dets = [make_detection("rgbd", POS), make_detection("laser", POS + [0.005, 0.0, 0.0])]
        a = fusion.fuse_detections(dets)[0]
        b = fusion.fuse(fusion.measurements_from_detections(dets, inflate_correlated=False))[0]
        assert np.allclose(a.position, b.position) and np.allclose(a.covariance, b.covariance)

    def test_fuse_rejects_already_fused_input(self):
        fusion = make_fusion()
        fused = fusion.fuse_detections([make_detection("rgbd", POS), make_detection("lidar", POS)])[0]
        with pytest.raises(ValueError):
            fusion.fuse([fused])

    def test_empty_input(self):
        fusion = make_fusion()
        assert fusion.fuse([]) == []
        assert fusion.fuse_detections([]) == []


class TestCombine:
    def test_single_measurement_is_returned_as_is(self):
        m = Measurement(("rgbd",), 0.0, POS, np.eye(3) * 1e-4)
        assert make_fusion().combine([m]) is m

    def test_invalid_input_raises(self):
        fusion = make_fusion()
        m = Measurement(("rgbd",), 0.0, POS, np.eye(3) * 1e-4)
        with pytest.raises(ValueError):
            fusion.combine([])
        with pytest.raises(ValueError):
            fusion.combine([m, m])  # same sensor twice
        fused = Measurement(("rgbd", "lidar"), 0.0, POS, np.eye(3) * 1e-4)
        with pytest.raises(ValueError):
            fusion.combine([fused, Measurement(("laser",), 0.0, POS, np.eye(3) * 1e-4)])

    def test_independent_sensors_fuse_consistently_and_beat_a_single_sensor(self):
        rng = np.random.default_rng(0)
        truth = np.array([0.2, 0.5, 1.3])
        r1, r2 = np.eye(3) * 1e-4, np.eye(3) * 1.5e-4
        fusion = make_fusion()
        single_sq, fused_sq, nees = [], [], []
        for _ in range(500):
            z1 = truth + rng.multivariate_normal(np.zeros(3), r1)
            z2 = truth + rng.multivariate_normal(np.zeros(3), r2)
            m = fusion.combine(
                [Measurement(("rgbd",), 0.0, z1, r1), Measurement(("lidar",), 0.0, z2, r2)]
            )
            err = m.position - truth
            single_sq.append(np.sum((z1 - truth) ** 2))
            fused_sq.append(np.sum(err**2))
            nees.append(err @ np.linalg.solve(m.covariance, err))
        assert np.sqrt(np.mean(fused_sq)) < 0.9 * np.sqrt(np.mean(single_sq))
        assert abs(np.mean(nees) - 3.0) < 0.4  # statistically consistent

    def test_covariance_intersection_stays_consistent_with_correlated_errors(self):
        rng = np.random.default_rng(1)
        truth = np.array([0.2, 0.5, 1.3])
        r = np.eye(3) * 1e-4
        fusion = make_fusion()  # rgbd and laser form a correlation group
        nees_ci, nees_independent = [], []
        for _ in range(500):
            common = rng.multivariate_normal(np.zeros(3), 0.5 * r)  # shared camera error
            z1 = truth + common + rng.multivariate_normal(np.zeros(3), 0.5 * r)
            z2 = truth + common + rng.multivariate_normal(np.zeros(3), 0.5 * r)
            ci = fusion.combine(
                [Measurement(("rgbd",), 0.0, z1, r), Measurement(("laser",), 0.0, z2, r)]
            )
            mean_i, cov_i = fuse_independent([z1, z2], [r, r])
            err_ci, err_i = ci.position - truth, mean_i - truth
            nees_ci.append(err_ci @ np.linalg.solve(ci.covariance, err_ci))
            nees_independent.append(err_i @ np.linalg.solve(cov_i, err_i))
        assert np.mean(nees_ci) < 3.0  # conservative (consistent)
        assert np.mean(nees_independent) > 3.5  # overconfident when correlation is ignored