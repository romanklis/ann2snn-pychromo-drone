"""Sensor models and observations for the estimator-in-the-loop stack.

The estimator and the policies never see the true plant state: they only receive
noisy, rate-limited, occasionally dropped measurements produced here from the
hidden truth.  Two things are produced each frame:

* a :class:`Measurement` for the UKF (GPS position, IMU specific force + gyro,
  INS velocity/attitude, barometer altitude, compass heading, LiDAR scan);
* an :class:`Observation` for the controller (the fused state estimate plus the
  measured LiDAR scan).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from .params import DT
from .sensor import SensorConfig as LidarConfig
from .sensor import scan as lidar_scan
from .so3 import rpy_from_R

__all__ = ["SensorConfig", "Measurement", "Observation", "SensorSuite"]

_EPS = 1e-9


@dataclass(frozen=True)
class SensorConfig:
    """Rates, noise and failure modes for the onboard sensor suite."""

    imu_hz: float = 50.0
    gps_hz: float = 10.0
    ins_hz: float = 50.0
    baro_hz: float = 25.0
    mag_hz: float = 25.0
    gps_sigma: float = 0.25
    gps_dropout_p: float = 0.05
    accel_sigma: float = 0.12
    gyro_sigma: float = 0.015
    accel_bias: Tuple[float, float, float] = (0.05, -0.04, 0.06)
    gyro_bias: Tuple[float, float, float] = (0.012, -0.010, 0.008)
    imu_outlier_p: float = 0.01
    imu_outlier_scale: float = 8.0
    vel_sigma: float = 0.08
    att_sigma: float = 0.02
    baro_sigma: float = 0.25
    mag_sigma: float = 0.04
    seed: int = 0

    def period(self, hz: float, dt: float) -> int:
        return max(1, int(round((1.0 / max(hz, _EPS)) / dt)))

    def to_dict(self) -> dict:
        return {
            "imu_hz": float(self.imu_hz), "gps_hz": float(self.gps_hz),
            "ins_hz": float(self.ins_hz), "baro_hz": float(self.baro_hz),
            "mag_hz": float(self.mag_hz),
            "gps_sigma": float(self.gps_sigma), "gps_dropout_p": float(self.gps_dropout_p),
            "accel_sigma": float(self.accel_sigma), "gyro_sigma": float(self.gyro_sigma),
            "imu_outlier_p": float(self.imu_outlier_p),
            "vel_sigma": float(self.vel_sigma), "att_sigma": float(self.att_sigma),
            "baro_sigma": float(self.baro_sigma), "mag_sigma": float(self.mag_sigma),
        }


@dataclass
class Measurement:
    step: int = 0
    t: float = 0.0
    gps_pos: Optional[np.ndarray] = None
    gps_ok: bool = False
    accel: Optional[np.ndarray] = None    # body-frame specific force [m/s^2]
    gyro: Optional[np.ndarray] = None     # body-frame angular rate [rad/s]
    vel: Optional[np.ndarray] = None      # INS world velocity
    attitude: Optional[np.ndarray] = None  # INS roll/pitch/yaw
    baro: Optional[float] = None
    heading: Optional[float] = None
    scan: Optional[np.ndarray] = None
    scan_angles: Optional[np.ndarray] = None


@dataclass
class Observation:
    """What a controller is allowed to see: the fused estimate + LiDAR scan."""

    state: np.ndarray                      # (6,) estimated [p, v]
    estimate: Optional[np.ndarray] = None  # full estimator state, if available
    scan: Optional[np.ndarray] = None
    scan_angles: Optional[np.ndarray] = None
    gps_ok: bool = True


class SensorSuite:
    def __init__(
        self,
        config: Optional[SensorConfig] = None,
        lidar: Optional[LidarConfig] = None,
        obstacles: Sequence = (),
        dt: float = DT,
    ) -> None:
        self.cfg = config or SensorConfig()
        self.lidar = lidar or LidarConfig()
        self.obstacles = tuple(obstacles)
        self.dt = float(dt)
        self.rng = np.random.default_rng(self.cfg.seed)
        gz = 9.81
        self.gravity = np.array([0.0, 0.0, -gz])
        self.accel_bias = np.asarray(self.cfg.accel_bias, dtype=np.float64)
        self.gyro_bias = np.asarray(self.cfg.gyro_bias, dtype=np.float64)
        self._p_imu = self.cfg.period(self.cfg.imu_hz, dt)
        self._p_gps = self.cfg.period(self.cfg.gps_hz, dt)
        self._p_ins = self.cfg.period(self.cfg.ins_hz, dt)
        self._p_baro = self.cfg.period(self.cfg.baro_hz, dt)
        self._p_mag = self.cfg.period(self.cfg.mag_hz, dt)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.cfg.seed)

    def measure(self, step: int, truth: dict) -> Measurement:
        p = np.asarray(truth["p"], dtype=np.float64)
        v = np.asarray(truth["v"], dtype=np.float64)
        R = np.asarray(truth["R"], dtype=np.float64)
        omega = np.asarray(truth["omega"], dtype=np.float64)
        omega_m = np.asarray(truth["omega_m"], dtype=np.float64)
        m = float(truth.get("mass", 0.5))
        C_T = float(truth.get("C_T", 1.5e-5))
        out = Measurement(step=step, t=step * self.dt)

        def spike(x, scale):
            return x + self.rng.normal(0.0, scale, size=np.shape(x))

        if step % self._p_imu == 0:
            # accelerometer measures specific force f = R^T (a_body - gravity)
            thrust = C_T * float(np.sum(omega_m ** 2))
            a_world = (thrust / m) * R[:, 2]
            f_body = R.T @ (a_world - self.gravity)
            acc = f_body + self.accel_bias + self.rng.normal(0.0, self.cfg.accel_sigma, 3)
            gyr = omega + self.gyro_bias + self.rng.normal(0.0, self.cfg.gyro_sigma, 3)
            if self.rng.random() < self.cfg.imu_outlier_p:
                acc = spike(acc, self.cfg.accel_sigma * self.cfg.imu_outlier_scale)
            if self.rng.random() < self.cfg.imu_outlier_p:
                gyr = spike(gyr, self.cfg.gyro_sigma * self.cfg.imu_outlier_scale)
            out.accel, out.gyro = acc, gyr

        if step % self._p_gps == 0:
            if self.rng.random() >= self.cfg.gps_dropout_p:
                out.gps_pos = p + self.rng.normal(0.0, self.cfg.gps_sigma, 3)
                out.gps_ok = True

        if step % self._p_ins == 0:
            out.vel = v + self.rng.normal(0.0, self.cfg.vel_sigma, 3)
            out.attitude = rpy_from_R(R) + self.rng.normal(0.0, self.cfg.att_sigma, 3)

        if step % self._p_baro == 0:
            out.baro = float(p[2] + self.rng.normal(0.0, self.cfg.baro_sigma))

        if step % self._p_mag == 0:
            out.heading = float(rpy_from_R(R)[2] + self.rng.normal(0.0, self.cfg.mag_sigma))

        if self.lidar is not None:
            out.scan = lidar_scan(self.obstacles, p[:2], self.lidar, self.rng)
            out.scan_angles = self.lidar.angles()

        return out
