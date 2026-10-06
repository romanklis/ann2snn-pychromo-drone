"""IMU model: an ideal accelerometer measures body-frame specific force.

Regression guard for the gravity double-count bug: with the old model the level
hover reading was ~2g (thrust plus a spurious gravity term); it must now be ~1g.
"""

from __future__ import annotations

import numpy as np
import pytest

from drone6dof.config import SENSOR
from drone6dof.params import QuadParams
from drone6dof.plant import Quad6DoF
from drone6dof.sensors import SensorConfig, SensorSuite


def test_hover_specific_force_is_one_g_not_two_g():
    p = QuadParams()
    plant = Quad6DoF(p)
    plant(np.zeros(6), np.zeros(3), dt=0.02, limit=12.0, gain=1.0)
    sf = plant.specific_force_body
    assert sf[0] == pytest.approx(0.0, abs=1e-6)
    assert sf[1] == pytest.approx(0.0, abs=1e-6)
    assert sf[2] == pytest.approx(p.g, rel=0.05)
    assert sf[2] < 1.5 * p.g          # the old model read ~2g here


def test_specific_force_includes_aerodynamic_drag():
    p = QuadParams()
    plant = Quad6DoF(p)
    # 1.5 m/s forward with a level body: in-plane H-drag + fuselage drag appear
    # in the specific force, so it differs from the thrust-only model.
    state = np.array([0.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    plant(state, np.zeros(3), dt=0.02, limit=12.0, gain=1.0)
    sf = plant.specific_force_body
    thrust_only = np.array([0.0, 0.0, plant.last_telemetry["thrust_n"] / p.m])
    assert np.linalg.norm(sf - thrust_only) > 1e-3
    assert np.all(np.isfinite(sf))


def _truth(sf):
    return {
        "p": np.zeros(3), "v": np.zeros(3), "R": np.eye(3),
        "omega": np.zeros(3), "omega_m": np.full(4, 286.0),
        "mass": 0.5, "C_T": 1.5e-5, "specific_force_body": np.asarray(sf, dtype=float),
    }


def test_sensor_reads_the_supplied_specific_force():
    cfg = SensorConfig(accel_bias=(0.0, 0.0, 0.0), accel_sigma=0.0,
                       imu_outlier_p=0.0, seed=0)
    suite = SensorSuite(cfg, lidar=SENSOR, dt=0.02)
    meas = suite.measure(0, _truth([5.0, -2.0, 11.0]))
    assert np.allclose(meas.accel, [5.0, -2.0, 11.0])


def test_sensor_fallback_is_thrust_only_at_hover():
    cfg = SensorConfig(accel_bias=(0.0, 0.0, 0.0), accel_sigma=0.0,
                       imu_outlier_p=0.0, seed=0)
    suite = SensorSuite(cfg, lidar=SENSOR, dt=0.02)
    truth = _truth([0.0, 0.0, 0.0])
    del truth["specific_force_body"]          # exercise the minimal-truth fallback
    meas = suite.measure(0, truth)
    assert meas.accel[2] == pytest.approx(9.81, rel=0.02)
