"""Unit tests for the estimation package."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from estimation.common import (
    check_dt,
    gaussian_log_likelihood,
    initial_state,
    make_psd,
    matrix_sqrt,
    nees,
    symmetrize,
)
from estimation.ekf import EKF
from estimation.imm import IMM, build_imm
from estimation.measurement_models import (
    POSITION_MODEL,
    PositionMeasurementModel,
    SphericalMeasurementModel,
    wrap_angle,
)
from estimation.motion_models import ConstantVelocity2D, ConstantVelocity3D
from estimation.ukf import UKF, UKFConfig, sigma_points, sigma_weights

SENSOR = np.array([0.0, -3.0, 1.5])
MOUNT = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # sensor x -> world +Y


# ------------------------------------------------------------------ helpers
def polar_model():
    return SphericalMeasurementModel(SENSOR, MOUNT)


def numeric_jacobian(model, x, eps=1e-6):
    jac = np.zeros((model.dim, 6))
    for i in range(6):
        xp, xm = np.array(x, dtype=float), np.array(x, dtype=float)
        xp[i] += eps
        xm[i] -= eps
        jac[:, i] = model.residual(model.h(xp), model.h(xm)) / (2.0 * eps)
    return jac


def reference_kf(model, x0, p0, zs, r, dt):
    """Textbook Kalman filter with a linear model and position measurements."""
    x, p = np.array(x0, dtype=float), np.array(p0, dtype=float)
    f = model.transition_matrix(dt)
    q = model.process_noise(dt)
    h = np.hstack([np.eye(3), np.zeros((3, 3))])
    for z in zs:
        x = f @ x
        p = f @ p @ f.T + q
        s = h @ p @ h.T + r
        k = p @ h.T @ np.linalg.inv(s)
        x = x + k @ (z - h @ x)
        p = (np.eye(6) - k @ h) @ p
    return x, p


# ------------------------------------------------------------------- common
class TestCommon:
    def test_symmetrize(self):
        m = np.array([[1.0, 2.0], [0.0, 1.0]])
        assert np.allclose(symmetrize(m), [[1.0, 1.0], [1.0, 1.0]])

    def test_make_psd_clips_negative_eigenvalues(self):
        m = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and -1
        fixed = make_psd(m)
        assert np.linalg.eigvalsh(fixed).min() >= -1e-12
        assert np.allclose(fixed, fixed.T)

    def test_matrix_sqrt_squares_back(self):
        a = np.random.default_rng(0).normal(size=(4, 4))
        p = a @ a.T
        root = matrix_sqrt(p)
        assert np.allclose(root @ root, p, atol=1e-9)
        assert np.allclose(root, root.T)

    def test_matrix_sqrt_handles_singular_matrices(self):
        v = np.array([1.0, 2.0, 3.0])
        p = np.outer(v, v)  # rank one
        root = matrix_sqrt(p)
        assert np.allclose(root @ root, p, atol=1e-9)

    def test_check_dt(self):
        assert check_dt(0.1) == 0.1
        assert check_dt(0) == 0.0
        for bad in (-0.1, np.nan, np.inf):
            with pytest.raises(ValueError):
                check_dt(bad)

    def test_nees(self):
        assert nees([1.0, 2.0], np.diag([1.0, 4.0])) == pytest.approx(2.0)

    def test_initial_state(self):
        x, p = initial_state([1.0, 2.0, 3.0], np.eye(3) * 0.01, velocity_std=0.5)
        assert np.allclose(x, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        assert np.allclose(p[:3, :3], 0.01 * np.eye(3))
        assert np.allclose(p[3:, 3:], 0.25 * np.eye(3))
        assert np.allclose(p[:3, 3:], 0.0)

    def test_initial_state_invalid(self):
        with pytest.raises(ValueError):
            initial_state([0, 0, 0], np.eye(3), velocity_std=0.0)
        with pytest.raises(ValueError):
            initial_state([0, 0, 0], np.eye(2))

    def test_gaussian_log_likelihood_matches_scipy(self):
        s = np.array([[2.0, 0.3, 0.0], [0.3, 1.0, 0.1], [0.0, 0.1, 0.5]])
        y = np.array([0.4, -0.2, 0.1])
        ll, nis = gaussian_log_likelihood(y, s)
        assert ll == pytest.approx(multivariate_normal(np.zeros(3), s).logpdf(y))
        assert nis == pytest.approx(y @ np.linalg.solve(s, y))

    def test_gaussian_log_likelihood_rejects_non_positive_definite(self):
        with pytest.raises(ValueError):
            gaussian_log_likelihood([0.0, 0.0], -np.eye(2))
        with pytest.raises(ValueError):
            gaussian_log_likelihood([0.0, 0.0], np.zeros((2, 2)))


# ------------------------------------------------------------- motion models
class TestMotionModels:
    def test_cv3d_transition_and_noise(self):
        model = ConstantVelocity3D(accel_std=0.5)
        dt = 0.1
        f = model.transition_matrix(dt)
        expected = np.eye(6)
        expected[:3, 3:] = dt * np.eye(3)
        assert np.allclose(f, expected)

        q = model.process_noise(dt)
        k = 0.25
        assert q[0, 0] == pytest.approx(k * dt**4 / 4)
        assert q[0, 3] == pytest.approx(k * dt**3 / 2)
        assert q[3, 3] == pytest.approx(k * dt**2)
        assert q[0, 1] == 0.0  # axes are independent
        assert np.allclose(q, q.T)
        assert np.linalg.eigvalsh(q).min() > -1e-15

    def test_cv3d_predict_state_and_jacobian(self):
        model = ConstantVelocity3D()
        x = np.array([1.0, 2.0, 3.0, 0.5, -0.5, 0.2])
        assert np.allclose(model.predict_state(x, 2.0), [2.0, 1.0, 3.4, 0.5, -0.5, 0.2])
        assert np.allclose(model.jacobian(x, 2.0), model.transition_matrix(2.0))

    def test_cv2d_holds_z_and_zeroes_vz(self):
        model = ConstantVelocity2D()
        x = np.array([1.0, 2.0, 3.0, 0.5, -0.5, 0.7])
        assert np.allclose(model.predict_state(x, 1.0), [1.5, 1.5, 3.0, 0.5, -0.5, 0.0])
        f = model.transition_matrix(0.1)
        assert f[5, 5] == 0.0 and f[2, 5] == 0.0
        assert f[0, 3] == pytest.approx(0.1) and f[1, 4] == pytest.approx(0.1)

    def test_cv2d_noise_structure(self):
        model = ConstantVelocity2D(accel_std=0.5, z_walk_std=0.01)
        dt = 0.1
        q = model.process_noise(dt)
        assert q[0, 0] == pytest.approx(0.25 * dt**4 / 4)
        assert q[1, 4] == pytest.approx(0.25 * dt**3 / 2)
        assert q[2, 2] == pytest.approx(0.01**2 * dt)
        assert np.all(q[5, :] == 0.0) and np.all(q[:, 5] == 0.0)  # no vertical velocity noise
        assert q[0, 1] == 0.0
        assert np.allclose(q, q.T)
        assert np.linalg.eigvalsh(q).min() > -1e-15

    def test_zero_dt(self):
        assert np.allclose(ConstantVelocity3D().transition_matrix(0.0), np.eye(6))
        assert np.allclose(ConstantVelocity3D().process_noise(0.0), 0.0)

    def test_negative_dt_and_noise_raise(self):
        with pytest.raises(ValueError):
            ConstantVelocity3D().transition_matrix(-0.1)
        with pytest.raises(ValueError):
            ConstantVelocity2D().process_noise(-0.1)
        with pytest.raises(ValueError):
            ConstantVelocity3D(accel_std=-1.0)
        with pytest.raises(ValueError):
            ConstantVelocity2D(z_walk_std=-1.0)
        with pytest.raises(ValueError):
            ConstantVelocity3D(name="")

    def test_planar_flag_and_names(self):
        assert ConstantVelocity2D().planar and not ConstantVelocity3D().planar
        assert ConstantVelocity2D().name == "CV2D" and ConstantVelocity3D().name == "CV3D"


# ------------------------------------------------------- measurement models
class TestMeasurementModels:
    def test_wrap_angle(self):
        assert wrap_angle(3.0 * np.pi / 2) == pytest.approx(-np.pi / 2)
        assert wrap_angle(-3.0 * np.pi / 2) == pytest.approx(np.pi / 2)
        assert wrap_angle(0.3) == pytest.approx(0.3)
        assert np.allclose(wrap_angle(np.array([2 * np.pi + 0.1, -0.1])), [0.1, -0.1])

    def test_position_model(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        model = PositionMeasurementModel()
        assert np.allclose(model.h(x), [1.0, 2.0, 3.0])
        h = model.jacobian(x)
        assert h.shape == (3, 6)
        assert np.allclose(h[:, :3], np.eye(3)) and np.allclose(h[:, 3:], 0.0)
        assert POSITION_MODEL.dim == 3

    def test_spherical_known_values(self):
        model = SphericalMeasurementModel([0.0, 0.0, 0.0])
        assert np.allclose(model.measure([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0])
        assert np.allclose(model.measure([0.0, 2.0, 0.0]), [2.0, np.pi / 2, 0.0])
        assert np.allclose(model.measure([1.0, 1.0, np.sqrt(2.0)]), [2.0, np.pi / 4, np.pi / 4])

    def test_spherical_with_sensor_pose(self):
        model = SphericalMeasurementModel([1.0, 1.0, 0.0], MOUNT)
        # 2 m along the sensor's forward axis (world +Y) from the sensor position.
        assert np.allclose(model.measure([1.0, 3.0, 0.0]), [2.0, 0.0, 0.0])
        # 2 m along the sensor's left axis (world -X).
        assert np.allclose(model.measure([-1.0, 1.0, 0.0]), [2.0, np.pi / 2, 0.0])

    def test_measure_and_to_world_roundtrip(self):
        model = polar_model()
        p = np.array([0.4, 0.3, 1.2])
        assert np.allclose(model.to_world(model.measure(p)), p)

    def test_jacobian_matches_finite_differences(self):
        model = polar_model()
        for x in ([0.4, 0.2, 1.1, 0.1, 0.2, 0.3], [-1.0, 1.5, 0.3, 0.0, 0.0, 0.0]):
            assert np.allclose(model.jacobian(x), numeric_jacobian(model, x), atol=1e-6)

    def test_residual_wraps_azimuth(self):
        model = SphericalMeasurementModel([0.0, 0.0, 0.0])
        y = model.residual([1.0, np.pi - 0.01, 0.0], [1.0, -np.pi + 0.01, 0.0])
        assert np.allclose(y, [0.0, -0.02, 0.0])

    def test_circular_mean_of_azimuth(self):
        model = SphericalMeasurementModel([0.0, 0.0, 0.0])
        zs = np.array([[1.0, np.pi - 0.1, 0.0], [3.0, -np.pi + 0.1, 0.2]])
        mean = model.mean(zs, [0.5, 0.5])
        assert mean[0] == pytest.approx(2.0) and mean[2] == pytest.approx(0.1)
        assert abs(abs(mean[1]) - np.pi) < 1e-9

    def test_singularities_raise(self):
        model = SphericalMeasurementModel([0.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            model.measure([0.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            model.jacobian([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])  # on the vertical axis

    def test_invalid_rotation_raises(self):
        with pytest.raises(ValueError):
            SphericalMeasurementModel([0, 0, 0], np.diag([1.0, 1.0, 2.0]))
        with pytest.raises(ValueError):
            SphericalMeasurementModel([0, 0, 0], np.diag([1.0, 1.0, -1.0]))
        with pytest.raises(ValueError):
            SphericalMeasurementModel([0, 0, 0], np.eye(2))


# --------------------------------------------------------------------- EKF
class TestEKF:
    X0 = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 3.0])

    def test_predict(self):
        model = ConstantVelocity3D(accel_std=0.5)
        p0 = np.diag([0.01, 0.02, 0.03, 0.1, 0.2, 0.3])
        ekf = EKF(model, self.X0, p0)
        ekf.predict(0.5)
        f, q = model.transition_matrix(0.5), model.process_noise(0.5)
        assert np.allclose(ekf.x, f @ self.X0)
        assert np.allclose(ekf.P, f @ p0 @ f.T + q)
        assert ekf.time == pytest.approx(0.5)

    def test_predict_to_and_backwards(self):
        ekf = EKF(ConstantVelocity3D(), self.X0, np.eye(6) * 0.01, time=1.0)
        ekf.predict_to(1.5)
        assert ekf.time == pytest.approx(1.5)
        ekf.predict_to(1.5)  # zero step is allowed
        with pytest.raises(ValueError):
            ekf.predict_to(1.0)

    def test_linear_case_matches_the_textbook_kalman_filter(self):
        rng = np.random.default_rng(0)
        model = ConstantVelocity3D(accel_std=0.4)
        p0 = np.diag([0.04] * 3 + [0.5] * 3)
        r = np.eye(3) * 0.02**2
        zs = [np.array([0.1 * k, 0.05 * k, 1.0]) + 0.02 * rng.standard_normal(3) for k in range(20)]
        ekf = EKF(model, self.X0, p0)
        for z in zs:
            ekf.predict(0.1)
            ekf.update(z, r)
        x_ref, p_ref = reference_kf(model, self.X0, p0, zs, r, 0.1)
        assert np.allclose(ekf.x, x_ref, atol=1e-9)
        assert np.allclose(ekf.P, p_ref, atol=1e-9)

    def test_update_pulls_the_state_toward_the_measurement_and_shrinks_covariance(self):
        ekf = EKF(ConstantVelocity3D(), self.X0, np.eye(6) * 0.1)
        z = np.array([0.5, 0.0, 1.0])
        trace_before = np.trace(ekf.P)
        before = np.linalg.norm(ekf.x[:3] - z)
        ekf.update(z, np.eye(3) * 0.01)
        assert np.linalg.norm(ekf.x[:3] - z) < before
        assert np.trace(ekf.P) < trace_before

    def test_update_result_for_a_perfect_prediction(self):
        ekf = EKF(ConstantVelocity3D(), self.X0, np.eye(6) * 0.05)
        r = np.eye(3) * 0.01
        _, s = ekf.predict_measurement(r)
        result = ekf.update(ekf.x[:3], r)
        assert result.nis == pytest.approx(0.0, abs=1e-12)
        assert np.allclose(result.innovation, 0.0)
        assert result.gain.shape == (6, 3)
        expected = -0.5 * (3 * np.log(2 * np.pi) + np.linalg.slogdet(s)[1])
        assert result.log_likelihood == pytest.approx(expected)

    def test_gating_distance_equals_the_nis(self):
        ekf = EKF(ConstantVelocity3D(), self.X0, np.eye(6) * 0.05)
        z, r = np.array([0.1, -0.2, 1.1]), np.eye(3) * 0.01
        d2 = ekf.gating_distance(z, r)
        assert ekf.update(z, r).nis == pytest.approx(d2)

    def test_predict_measurement(self):
        ekf = EKF(ConstantVelocity3D(), self.X0, np.eye(6) * 0.05)
        z_pred, s = ekf.predict_measurement(np.eye(3) * 0.01)
        assert np.allclose(z_pred, self.X0[:3])
        assert np.allclose(s, 0.06 * np.eye(3))

    def test_joseph_form_keeps_the_covariance_valid(self):
        rng = np.random.default_rng(1)
        ekf = EKF(ConstantVelocity3D(accel_std=0.01), self.X0, np.eye(6))
        r = np.eye(3) * 1e-8
        for _ in range(500):
            ekf.predict(0.1)
            ekf.update(np.array([0.0, 0.0, 1.0]) + 1e-4 * rng.standard_normal(3), r)
        p = ekf.P
        assert np.allclose(p, p.T)
        assert np.linalg.eigvalsh(p).min() > -1e-12
        assert np.isfinite(ekf.x).all()

    def test_update_measurement_accepts_a_measurement_like_object(self):
        ekf1 = EKF(ConstantVelocity3D(), self.X0, np.eye(6) * 0.05)
        ekf2 = EKF(ConstantVelocity3D(), self.X0, np.eye(6) * 0.05)
        z, r = np.array([0.1, 0.0, 1.0]), np.eye(3) * 0.01
        ekf1.update_measurement(SimpleNamespace(position=z, covariance=r))
        ekf2.update(z, r)
        assert np.allclose(ekf1.x, ekf2.x) and np.allclose(ekf1.P, ekf2.P)

    def test_tracks_a_constant_velocity_target(self):
        rng = np.random.default_rng(3)
        v = np.array([0.4, -0.2, 0.1])
        p_start = np.array([0.0, 0.0, 1.0])
        r = np.eye(3) * 0.02**2
        x0, p0 = initial_state(p_start, r, velocity_std=1.0)
        ekf = EKF(ConstantVelocity3D(accel_std=0.05), x0, p0)
        t = 0.0
        for _ in range(100):
            t += 0.05
            ekf.predict(0.05)
            ekf.update(p_start + v * t + 0.02 * rng.standard_normal(3), r)
        assert np.linalg.norm(ekf.x[3:] - v) < 0.1
        assert np.linalg.norm(ekf.x[:3] - (p_start + v * t)) < 0.05

    def test_nees_is_consistent_for_a_matched_model(self):
        rng = np.random.default_rng(0)
        model = ConstantVelocity3D(accel_std=0.5)
        dt = 0.1
        f = model.transition_matrix(dt)
        q_root = matrix_sqrt(model.process_noise(dt))
        r = np.eye(3) * 0.02**2
        p0 = np.diag([0.05**2] * 3 + [0.3**2] * 3)
        x0 = np.array([0.0, 0.0, 1.0, 0.5, -0.3, 0.1])
        p0_std = np.sqrt(np.diag(p0))
        values = []
        for _ in range(150):
            truth = x0 + p0_std * rng.standard_normal(6)
            ekf = EKF(model, x0, p0)
            for _ in range(40):
                truth = f @ truth + q_root @ rng.standard_normal(6)
                ekf.predict(dt)
                ekf.update(truth[:3] + 0.02 * rng.standard_normal(3), r)
            values.append(nees(ekf.x - truth, ekf.P))
        assert abs(np.mean(values) - 6.0) < 1.0

    def test_nonlinear_update_with_spherical_measurements(self):
        model = polar_model()
        truth = np.array([0.5, 0.3, 1.2])
        x0 = np.concatenate([truth + [0.08, -0.05, 0.06], np.zeros(3)])
        p0 = np.diag([0.15**2] * 3 + [1.0] * 3)
        ekf = EKF(ConstantVelocity3D(), x0, p0)
        r = np.diag([0.01**2, 0.002**2, 0.002**2])
        before = np.linalg.norm(ekf.x[:3] - truth)
        ekf.update(model.measure(truth), r, model)
        assert np.linalg.norm(ekf.x[:3] - truth) < 0.5 * before

    def test_azimuth_wrap_is_handled(self):
        model = SphericalMeasurementModel([0.0, 0.0, 0.0])
        truth = np.array([-3.0, 0.01, 0.5])
        x0 = np.array([-3.0, -0.01, 0.5, 0.0, 0.0, 0.0])
        p0 = np.diag([0.05**2] * 3 + [0.5**2] * 3)
        r = np.diag([0.01**2, 0.001**2, 0.001**2])
        ekf = EKF(ConstantVelocity3D(), x0, p0)
        ekf.update(model.measure(truth), r, model)
        assert np.linalg.norm(ekf.x[:3] - truth) < 0.005

    def test_invalid_inputs_raise(self):
        model = ConstantVelocity3D()
        p = np.eye(6) * 0.01
        with pytest.raises(ValueError):
            EKF(model, np.zeros(5), p)
        with pytest.raises(ValueError):
            EKF(model, np.zeros(6), np.eye(5))
        with pytest.raises(ValueError):
            EKF(model, np.zeros(6), -np.eye(6))
        with pytest.raises(ValueError):
            EKF(model, np.full(6, np.nan), p)
        asymmetric = p.copy()
        asymmetric[0, 1] = 0.5
        with pytest.raises(ValueError):
            EKF(model, np.zeros(6), asymmetric)

        ekf = EKF(model, np.zeros(6), p)
        with pytest.raises(ValueError):
            ekf.update(np.zeros(3), np.eye(2))
        with pytest.raises(ValueError):
            ekf.update(np.zeros(3), np.zeros((3, 3)))  # not positive definite
        with pytest.raises(ValueError):
            ekf.update(np.zeros(2), np.eye(3))
        with pytest.raises(ValueError):
            ekf.predict(-0.1)


# --------------------------------------------------------------------- UKF
class TestUKF:
    X0 = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 3.0])

    def test_sigma_weights(self):
        wm, wc, lam = sigma_weights(6, UKFConfig())
        assert len(wm) == 13 and len(wc) == 13
        assert wm.sum() == pytest.approx(1.0)
        assert np.all(wc >= 0.0)
        assert lam == pytest.approx(0.0)

    def test_sigma_points_reproduce_mean_and_covariance(self):
        a = np.random.default_rng(0).normal(size=(6, 6))
        p = a @ a.T
        for cov in (p, np.diag([1.0, 2.0, 0.0, 0.5, 0.1, 0.0])):  # also a singular covariance
            wm, wc, lam = sigma_weights(6, UKFConfig(alpha=0.8, kappa=1.0))
            pts = sigma_points(self.X0, cov, lam)
            assert pts.shape == (13, 6)
            mean = wm @ pts
            d = pts - mean
            assert np.allclose(mean, self.X0, atol=1e-9)
            assert np.allclose((wc[:, None] * d).T @ d, cov, atol=1e-9)

    def test_matches_the_ekf_for_linear_models(self):
        rng = np.random.default_rng(1)
        model = ConstantVelocity3D(accel_std=0.4)
        p0 = np.diag([0.04] * 3 + [0.5] * 3)
        r = np.eye(3) * 0.02**2
        ekf, ukf = EKF(model, self.X0, p0), UKF(model, self.X0, p0)
        for k in range(15):
            z = np.array([0.1 * k, 0.05 * k, 1.0]) + 0.02 * rng.standard_normal(3)
            ekf.predict(0.1)
            ukf.predict(0.1)
            ekf.update(z, r)
            ukf.update(z, r)
        assert np.allclose(ukf.x, ekf.x, atol=1e-8)
        assert np.allclose(ukf.P, ekf.P, atol=1e-8)

    def test_predicted_measurement_matches_the_ekf_for_small_covariance(self):
        model = polar_model()
        x = np.array([0.4, 0.2, 1.1, 0.1, 0.2, 0.3])
        p = np.diag([0.01**2] * 3 + [0.1**2] * 3)
        r = np.diag([0.01**2, 0.002**2, 0.002**2])
        ekf, ukf = EKF(ConstantVelocity3D(), x, p), UKF(ConstantVelocity3D(), x, p)
        z_e, s_e = ekf.predict_measurement(r, model)
        z_u, s_u = ukf.predict_measurement(r, model)
        assert np.allclose(z_u, z_e, atol=1e-4)
        assert np.allclose(s_u, s_e, rtol=0.02, atol=1e-9)

    def test_nonlinear_update_with_spherical_measurements(self):
        model = polar_model()
        truth = np.array([0.5, 0.3, 1.2])
        x0 = np.concatenate([truth + [0.08, -0.05, 0.06], np.zeros(3)])
        p0 = np.diag([0.15**2] * 3 + [1.0] * 3)
        ukf = UKF(ConstantVelocity3D(), x0, p0)
        r = np.diag([0.01**2, 0.002**2, 0.002**2])
        before = np.linalg.norm(ukf.x[:3] - truth)
        ukf.update(model.measure(truth), r, model)
        assert np.linalg.norm(ukf.x[:3] - truth) < 0.5 * before

    def test_azimuth_wrap_is_handled(self):
        model = SphericalMeasurementModel([0.0, 0.0, 0.0])
        truth = np.array([-3.0, 0.01, 0.5])
        x0 = np.array([-3.0, -0.01, 0.5, 0.0, 0.0, 0.0])
        p0 = np.diag([0.05**2] * 3 + [0.5**2] * 3)
        r = np.diag([0.01**2, 0.001**2, 0.001**2])
        ukf = UKF(ConstantVelocity3D(), x0, p0)
        ukf.update(model.measure(truth), r, model)
        assert np.linalg.norm(ukf.x[:3] - truth) < 0.005

    def test_singular_covariance_with_the_planar_model(self):
        p0 = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.0])  # zero vz variance
        ukf = UKF(ConstantVelocity2D(), self.X0, p0)
        for _ in range(20):
            ukf.predict(0.1)
            ukf.update(np.array([0.0, 0.0, 1.0]), np.eye(3) * 1e-4)
        assert np.isfinite(ukf.x).all()
        assert np.linalg.eigvalsh(ukf.P).min() > -1e-12
        assert ukf.x[5] == pytest.approx(0.0, abs=1e-6)

    def test_gating_distance_equals_the_nis(self):
        ukf = UKF(ConstantVelocity3D(), self.X0, np.eye(6) * 0.05)
        z, r = np.array([0.1, -0.2, 1.1]), np.eye(3) * 0.01
        d2 = ukf.gating_distance(z, r)
        assert ukf.update(z, r).nis == pytest.approx(d2)

    def test_tracks_with_spherical_measurements(self):
        rng = np.random.default_rng(4)
        model = polar_model()
        r = np.diag([0.01**2, 0.003**2, 0.003**2])
        p_start = np.array([-0.5, 0.0, 1.2])
        v = np.array([0.3, 0.2, 0.1])
        x0 = np.concatenate([p_start + 0.1, np.zeros(3)])
        p0 = np.diag([0.2**2] * 3 + [1.0] * 3)
        ukf = UKF(ConstantVelocity3D(accel_std=0.1), x0, p0)
        sigma = np.sqrt(np.diag(r))
        t = 0.0
        for _ in range(40):
            t += 0.1
            ukf.predict(0.1)
            ukf.update(model.measure(p_start + v * t) + sigma * rng.standard_normal(3), r, model)
        assert np.linalg.norm(ukf.x[:3] - (p_start + v * t)) < 0.05
        assert np.linalg.norm(ukf.x[3:] - v) < 0.1

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError):
            UKFConfig(alpha=0.0)
        with pytest.raises(ValueError):
            UKFConfig(alpha=1.5)
        with pytest.raises(ValueError):
            UKFConfig(beta=-1.0)
        with pytest.raises(ValueError):
            UKFConfig(kappa=-6.0)


class TestEkfOnSphericalTracking:
    def test_ekf_tracks_with_spherical_measurements(self):
        rng = np.random.default_rng(4)
        model = polar_model()
        r = np.diag([0.01**2, 0.003**2, 0.003**2])
        p_start = np.array([-0.5, 0.0, 1.2])
        v = np.array([0.3, 0.2, 0.1])
        x0 = np.concatenate([p_start + 0.1, np.zeros(3)])
        p0 = np.diag([0.2**2] * 3 + [1.0] * 3)
        ekf = EKF(ConstantVelocity3D(accel_std=0.1), x0, p0)
        sigma = np.sqrt(np.diag(r))
        t = 0.0
        for _ in range(40):
            t += 0.1
            ekf.predict(0.1)
            ekf.update(model.measure(p_start + v * t) + sigma * rng.standard_normal(3), r, model)
        assert np.linalg.norm(ekf.x[:3] - (p_start + v * t)) < 0.05
        assert np.linalg.norm(ekf.x[3:] - v) < 0.1


# --------------------------------------------------------------------- IMM
def two_cv3d(p=None, x_a=None, x_b=None):
    p = np.eye(6) * 0.01 if p is None else p
    x_a = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]) if x_a is None else x_a
    x_b = x_a if x_b is None else x_b
    a = EKF(ConstantVelocity3D(0.5, name="A"), x_a, p)
    b = EKF(ConstantVelocity3D(0.5, name="B"), x_b, p)
    return a, b


class TestIMM:
    def test_identical_models_reduce_to_a_single_filter(self):
        rng = np.random.default_rng(0)
        x0, p0 = initial_state([0.0, 0.0, 1.0], np.eye(3) * 0.01, 1.0)
        a = EKF(ConstantVelocity3D(0.5, name="A"), x0, p0)
        b = EKF(ConstantVelocity3D(0.5, name="B"), x0, p0)
        imm = IMM([a, b], [[0.9, 0.1], [0.1, 0.9]])
        single = EKF(ConstantVelocity3D(0.5), x0, p0)
        r = np.eye(3) * 0.02**2
        for k in range(30):
            z = np.array([0.01 * k, 0.0, 1.0]) + 0.02 * rng.standard_normal(3)
            imm.predict(0.1)
            single.predict(0.1)
            imm.update(z, r)
            single.update(z, r)
        assert np.allclose(imm.x, single.x, atol=1e-9)
        assert np.allclose(imm.P, single.P, atol=1e-10)
        assert np.allclose(imm.probabilities, [0.5, 0.5], atol=1e-9)

    def test_predict_applies_the_transition_matrix(self):
        a, b = two_cv3d()
        imm = IMM([a, b], [[0.9, 0.1], [0.2, 0.8]], initial_probabilities=[0.7, 0.3])
        imm.predict(0.1)
        assert np.allclose(imm.probabilities, [0.69, 0.31])
        assert imm.time == pytest.approx(0.1)
        assert imm.probabilities_by_name == pytest.approx({"A": 0.69, "B": 0.31})

    def test_combined_covariance_includes_the_spread_of_the_means(self):
        p = np.eye(6) * 0.01
        a, b = two_cv3d(p, x_a=np.array([0.0, 0, 0, 1, 0, 0]), x_b=np.array([1.0, 0, 0, 1, 0, 0]))
        imm = IMM([a, b], np.eye(2))
        imm.predict(0.0)
        assert np.allclose(imm.x, [0.5, 0.0, 0.0, 1.0, 0.0, 0.0])
        expected = p.copy()
        expected[0, 0] += 0.25
        assert np.allclose(imm.P, expected)

    def test_vertical_motion_selects_the_3d_model(self):
        rng = np.random.default_rng(2)
        r = np.eye(3) * 0.01**2
        dt = 0.05
        v = np.array([0.4, 0.2, 0.8])
        p_start = np.array([-0.5, 0.0, 0.5])
        x0, p0 = initial_state(p_start, r, velocity_std=1.0)
        imm = build_imm("ekf", x0, p0, accel_std_2d=0.1, accel_std_3d=0.5)
        for k in range(1, 41):
            imm.predict(dt)
            imm.update(p_start + v * (k * dt) + 0.01 * rng.standard_normal(3), r)
        assert imm.most_likely_model == "CV3D"
        assert imm.probability("CV3D") > 0.95

    def test_planar_motion_selects_the_2d_model(self):
        rng = np.random.default_rng(5)
        r = np.eye(3) * 0.01**2
        dt = 0.05
        v = np.array([0.4, 0.2, 0.0])
        p_start = np.array([-0.5, 0.0, 0.3])
        x0, p0 = initial_state(p_start, r, velocity_std=1.0)
        imm = build_imm("ekf", x0, p0, accel_std_2d=0.05, accel_std_3d=1.0)
        for k in range(1, 61):
            imm.predict(dt)
            imm.update(p_start + v * (k * dt) + 0.01 * rng.standard_normal(3), r)
        assert imm.most_likely_model == "CV2D"
        assert imm.probability("CV2D") > 0.9
        assert imm.probability("CV3D") >= 1e-4 - 1e-12  # probability floor

    def test_update_result(self):
        a, b = two_cv3d()
        imm = IMM([a, b], [[0.9, 0.1], [0.1, 0.9]])
        result = imm.update(np.array([0.1, 0.0, 0.0]), np.eye(3) * 0.01)
        assert result.probabilities.sum() == pytest.approx(1.0)
        assert result.model_log_likelihoods.shape == (2,)
        assert result.innovation.shape == (3,)
        assert result.innovation_cov.shape == (3, 3)
        assert result.nis >= 0.0 and np.isfinite(result.log_likelihood)

    def test_update_measurement_accepts_a_measurement_like_object(self):
        a1, b1 = two_cv3d()
        a2, b2 = two_cv3d()
        imm1 = IMM([a1, b1], np.eye(2))
        imm2 = IMM([a2, b2], np.eye(2))
        z, r = np.array([0.1, 0.0, 0.0]), np.eye(3) * 0.01
        imm1.update_measurement(SimpleNamespace(position=z, covariance=r))
        imm2.update(z, r)
        assert np.allclose(imm1.x, imm2.x)

    def test_gating_distance_of_identical_models_equals_the_single_filter(self):
        x0, p0 = initial_state([0.0, 0.0, 1.0], np.eye(3) * 0.01, 1.0)
        a = EKF(ConstantVelocity3D(0.5, name="A"), x0, p0)
        b = EKF(ConstantVelocity3D(0.5, name="B"), x0, p0)
        imm = IMM([a, b], [[0.9, 0.1], [0.1, 0.9]])
        single = EKF(ConstantVelocity3D(0.5), x0, p0)
        imm.predict(0.1)
        single.predict(0.1)
        z, r = np.array([0.1, 0.05, 1.0]), np.eye(3) * 0.01
        assert imm.gating_distance(z, r) == pytest.approx(single.gating_distance(z, r))

    def test_works_with_ukf_and_spherical_measurements(self):
        rng = np.random.default_rng(6)
        model = polar_model()
        r = np.diag([0.01**2, 0.003**2, 0.003**2])
        p_start = np.array([-0.5, 0.0, 1.2])
        v = np.array([0.3, 0.2, 0.1])
        x0 = np.concatenate([p_start + 0.1, np.zeros(3)])
        p0 = np.diag([0.2**2] * 3 + [1.0] * 3)
        imm = build_imm("ukf", x0, p0, accel_std_2d=0.1, accel_std_3d=0.2)
        sigma = np.sqrt(np.diag(r))
        t = 0.0
        for _ in range(40):
            t += 0.1
            imm.predict(0.1)
            imm.update(model.measure(p_start + v * t) + sigma * rng.standard_normal(3), r, model)
        assert np.linalg.norm(imm.x[:3] - (p_start + v * t)) < 0.1
        assert imm.probabilities.sum() == pytest.approx(1.0)

    def test_build_imm(self):
        x0, p0 = initial_state([0.0, 0.0, 1.0], np.eye(3) * 0.01, 1.0)
        for kind, cls in (("ekf", EKF), ("UKF", UKF)):
            imm = build_imm(kind, x0, p0)
            assert imm.model_names == ["CV2D", "CV3D"]
            assert np.allclose(imm.probabilities, [0.5, 0.5])
            assert all(isinstance(f, cls) for f in imm.filters)
        with pytest.raises(ValueError):
            build_imm("particle", x0, p0)
        with pytest.raises(ValueError):
            build_imm("ekf", x0, p0, stay_probability=1.0)

    def test_predict_to_and_backwards(self):
        a, b = two_cv3d()
        imm = IMM([a, b], np.eye(2))
        imm.predict_to(0.5)
        assert imm.time == pytest.approx(0.5)
        with pytest.raises(ValueError):
            imm.predict_to(0.1)

    def test_invalid_construction_raises(self):
        a, b = two_cv3d()
        p = np.eye(6) * 0.01
        with pytest.raises(ValueError):
            IMM([], np.eye(1))
        dup = EKF(ConstantVelocity3D(0.5, name="A"), np.zeros(6), p)
        with pytest.raises(ValueError):
            IMM([a, dup], np.eye(2))  # duplicate names
        with pytest.raises(ValueError):
            IMM([a, b], np.eye(3))
        with pytest.raises(ValueError):
            IMM([a, b], [[0.5, 0.4], [0.1, 0.9]])  # first row sums to 0.9
        with pytest.raises(ValueError):
            IMM([a, b], [[1.2, -0.2], [0.1, 0.9]])
        with pytest.raises(ValueError):
            IMM([a, b], [[1.0, 0.0], [1.0, 0.0]])  # model B is unreachable
        with pytest.raises(ValueError):
            IMM([a, b], np.eye(2), initial_probabilities=[0.5, 0.6])
        with pytest.raises(ValueError):
            IMM([a, b], np.eye(2), min_probability=0.6)
        late = EKF(ConstantVelocity3D(0.5, name="C"), np.zeros(6), p, time=1.0)
        with pytest.raises(ValueError):
            IMM([a, late], np.eye(2))