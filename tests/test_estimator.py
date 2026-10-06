"""Estimator-in-the-loop: UKF tracking, determinism, sensor models."""

from __future__ import annotations

import numpy as np
import pytest

from drone6dof.config import CONTROL_LIMIT, INIT_STATE, SENSOR, build_scene
from drone6dof.control import DSGuidanceController
from drone6dof.dynamics import NumpyPlantBackend
from drone6dof.estimator import ErrorStateUKF
from drone6dof.sensors import SensorConfig, SensorSuite
from drone6dof.sim import Simulation


def _truth():
    # Level hover: 4 rotors at the hover speed produce specific force +g along
    # body z (an ideal accelerometer cannot measure gravity).
    return {
        "p": np.array([0.0, 0.0, 1.0]),
        "v": np.array([0.2, 0.0, 0.0]),
        "R": np.eye(3),
        "omega": np.zeros(3),
        "omega_m": np.full(4, 286.0),
        "mass": 0.5,
        "C_T": 1.5e-5,
        "specific_force_body": np.array([0.0, 0.0, 9.81]),
    }


def test_ukf_predict_update_are_finite_and_shaped():
    ukf = ErrorStateUKF()
    ukf.reset(np.zeros(3), np.zeros(3))
    meas = SensorSuite(SensorConfig(seed=1), lidar=SENSOR).measure(1, _truth())
    view = ukf.step(meas, np.full(4, 286.0))
    assert view.shape == (6,)
    assert np.all(np.isfinite(view))
    assert np.all(np.isfinite(ukf.P))
    assert np.allclose(np.linalg.norm(ukf.q), 1.0)


def test_sensor_suite_rates_and_dropout():
    suite = SensorSuite(SensorConfig(gps_dropout_p=1.0, seed=0), lidar=SENSOR, dt=0.02)
    m0 = suite.measure(0, _truth())
    assert m0.gps_pos is None and m0.gps_ok is False       # forced dropout
    assert m0.accel is not None and m0.gyro is not None     # IMU at every step
    assert m0.scan is not None and m0.scan_angles is not None
    assert len(m0.scan) == len(m0.scan_angles)

    clean = SensorSuite(SensorConfig(gps_dropout_p=0.0, seed=0), lidar=SENSOR, dt=0.02)
    mc = clean.measure(0, _truth())
    assert mc.gps_ok and np.allclose(mc.gps_pos, [0, 0, 1], atol=1.0)


def test_estimator_tracks_the_hidden_plant():
    scene = build_scene()
    backend = NumpyPlantBackend(heading_target=scene.goal_np)
    ctrl = DSGuidanceController(action_limit=CONTROL_LIMIT, scene=scene)
    sim = Simulation(backend, ctrl, scene, steps=200, initial_state=INIT_STATE,
                     use_estimator=True)
    sim.run()
    est = np.asarray(sim.history["estimate"], dtype=float)
    true = np.asarray(sim.history["state"], dtype=float)
    assert est.shape[1] == 6
    rmse = float(np.sqrt(np.mean(np.sum((true[:, :3] - est[:, :3]) ** 2, axis=1))))
    assert rmse < 0.5, rmse


def test_estimator_is_deterministic():
    scene = build_scene()

    def run():
        backend = NumpyPlantBackend(heading_target=scene.goal_np)
        ctrl = DSGuidanceController(action_limit=CONTROL_LIMIT, scene=scene)
        sim = Simulation(backend, ctrl, scene, steps=60, initial_state=INIT_STATE,
                         use_estimator=True)
        sim.run()
        return np.asarray(sim.history["estimate"], dtype=float)

    assert np.array_equal(run(), run())


def test_ukf_covariance_stays_positive_definite():
    scene = build_scene()
    backend = NumpyPlantBackend(heading_target=scene.goal_np)
    ctrl = DSGuidanceController(action_limit=CONTROL_LIMIT, scene=scene)
    sim = Simulation(backend, ctrl, scene, steps=120, initial_state=INIT_STATE,
                     use_estimator=True)
    for _ in range(120):
        sim.step()
        assert sim.estimator.min_eig >= -1e-9, sim.estimator.min_eig
    assert np.all(np.isfinite(sim.estimator.P))


def test_ukf_zero_bias_initialisation_is_stable():
    # The shipped default initialises b_a/b_g to the *true* simulated biases
    # (privileged).  With zero initial bias the filter must still stay finite and
    # PSD while it learns the bias.
    from drone6dof.config import SENSOR_SUITE

    truth = _truth()
    suite = SensorSuite(SensorConfig(seed=3), lidar=SENSOR, dt=0.02)
    ukf = ErrorStateUKF(sensor=SENSOR_SUITE)
    ukf.reset(truth["p"], truth["v"])
    ukf.b_a = np.zeros(3)
    ukf.b_g = np.zeros(3)
    for step in range(1, 60):
        meas = suite.measure(step, truth)
        view = ukf.step(meas, truth["omega_m"])
        assert np.all(np.isfinite(view))
        assert ukf.min_eig >= -1e-9
    assert np.linalg.norm(ukf.q) == pytest.approx(1.0, abs=1e-9)
