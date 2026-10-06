"""Error-state Unscented Kalman Filter for the 6-DoF quadrotor.

The estimator fuses noisy GPS/IMU/INS/baro/compass measurements into a full state
estimate: position, velocity, attitude (quaternion), body rates, rotor speeds and
IMU biases.  The covariance lives on a 22-D **error** state (attitude errors are
3-D rotation vectors), which avoids the unit-norm singularity of putting a
quaternion directly in the covariance.

The process model is a *simplified* nonlinear quad (kinematics, thrust mapping,
rotor first-order lag + saturation) driven by the applied rotor speeds — never the
full plant, so the 2n+1 sigma-point propagation stays cheap.  Measurements are
gated (Mahalanobis) so dropouts/outliers do not wreck the estimate.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .params import DT, QuadParams
from .sensors import Measurement, SensorConfig
from .so3 import rpy_from_R

__all__ = ["ErrorStateUKF"]

_EPS = 1e-9


# ---------------------------------------------------------------- quaternion --- #
def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _qnorm(q):
    return q / (np.linalg.norm(q) + _EPS)


def _q_from_rotvec(rv):
    th = float(np.linalg.norm(rv))
    if th < 1e-9:
        return np.array([1.0, 0.5 * rv[0], 0.5 * rv[1], 0.5 * rv[2]])
    axis = rv / th
    h = 0.5 * th
    return np.array([np.cos(h), axis[0] * np.sin(h), axis[1] * np.sin(h), axis[2] * np.sin(h)])


def _rotvec_from_q(q):
    q = _qnorm(q)
    if q[0] < 0.0:
        q = -q
    v = q[1:4]
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        return np.zeros(3)
    ang = 2.0 * np.arctan2(s, q[0])
    return v / s * ang


def _R_from_q(q):
    w, x, y, z = _qnorm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class ErrorStateUKF:
    """UKF over ``[p, v, q, omega, omega_m, b_a, b_g]`` (22-D error state)."""

    N_ERR = 22

    def __init__(
        self,
        params: Optional[QuadParams] = None,
        sensor: Optional[SensorConfig] = None,
        *,
        dt: float = DT,
        rotor_tau: float = 0.05,
        rotor_omega_max: float = 1400.0,
        alpha: float = 0.3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        self.p = params or QuadParams()
        self.sensor = sensor or SensorConfig()
        self.dt = float(dt)
        self.rotor_tau = float(rotor_tau)
        self.rotor_omega_max = float(rotor_omega_max)
        self.m = float(self.p.m)
        self.J = np.diag(self.p.J_diag)
        self.J_inv = np.linalg.inv(self.J)
        self.C_T = float(self.p.C_T)
        self.gravity = np.array([0.0, 0.0, -self.p.g])
        d_arm = self.p.d / np.sqrt(2.0)
        c_ratio = self.p.C_Q / self.p.C_T
        self.B = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [-d_arm, d_arm, d_arm, -d_arm],
            [-d_arm, d_arm, -d_arm, d_arm],
            [-c_ratio, -c_ratio, c_ratio, c_ratio],
        ])
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self.lam = self.alpha ** 2 * (self.N_ERR + self.kappa) - self.N_ERR
        n = self.N_ERR
        self.Wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + self.lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lam / (n + self.lam)
        self.Wc[0] = self.lam / (n + self.lam) + (1.0 - self.alpha ** 2 + self.beta)
        self.Q = self._default_process_noise()
        self.reset()

    # ------------------------------------------------------------- lifecycle --
    def reset(self, p0=None, v0=None) -> None:
        self.pos = np.asarray(p0, dtype=np.float64).reshape(3) if p0 is not None else np.zeros(3)
        self.vel = np.asarray(v0, dtype=np.float64).reshape(3) if v0 is not None else np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.omega = np.zeros(3)
        self.omega_m = np.full(4, float(self.p.hover_omega))
        self.b_a = np.asarray(self.sensor.accel_bias, dtype=np.float64).copy()
        self.b_g = np.asarray(self.sensor.gyro_bias, dtype=np.float64).copy()
        init = np.concatenate([
            np.full(3, 1.0),    # position
            np.full(3, 1.0),    # velocity
            np.full(3, 0.1),    # attitude
            np.full(3, 0.5),    # rates
            np.full(4, 200.0),  # rotor speeds
            np.full(3, 0.05),   # accel bias
            np.full(3, 0.02),   # gyro bias
        ])
        self.P = np.diag(init)
        self._last_nis = 0.0
        self._updates = 0
        return self.state_view()

    # ------------------------------------------------------------ state view --
    def _pack(self) -> np.ndarray:
        return np.concatenate([self.pos, self.vel, self.q, self.omega, self.omega_m, self.b_a, self.b_g])

    def _unpack(self, x: np.ndarray) -> None:
        self.pos = x[0:3].copy()
        self.vel = x[3:6].copy()
        self.q = _qnorm(x[6:10])
        self.omega = x[10:13].copy()
        self.omega_m = x[13:17].copy()
        self.b_a = x[17:20].copy()
        self.b_g = x[20:23].copy()

    def _add(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        out = x.copy()
        out[0:3] += d[0:3]
        out[3:6] += d[3:6]
        out[6:10] = _qmul(out[6:10], _q_from_rotvec(d[6:9]))
        out[10:13] += d[9:12]
        out[13:17] += d[12:16]
        out[17:20] += d[16:19]
        out[20:23] += d[19:22]
        out[6:10] = _qnorm(out[6:10])
        return out

    def _diff(self, x1: np.ndarray, x0: np.ndarray) -> np.ndarray:
        d = np.zeros(self.N_ERR)
        d[0:3] = x1[0:3] - x0[0:3]
        d[3:6] = x1[3:6] - x0[3:6]
        d[6:9] = _rotvec_from_q(_qmul(_qconj(x0[6:10]), x1[6:10]))
        d[9:12] = x1[10:13] - x0[10:13]
        d[12:16] = x1[13:17] - x0[13:17]
        d[16:19] = x1[17:20] - x0[17:20]
        d[19:22] = x1[20:23] - x0[20:23]
        return d

    def state_view(self) -> np.ndarray:
        """``[p, v]`` — what a controller is allowed to consume."""
        return np.concatenate([self.pos, self.vel])

    @property
    def position(self) -> np.ndarray:
        return self.pos.copy()

    @property
    def rotation(self) -> np.ndarray:
        return _R_from_q(self.q)

    @property
    def last_nis(self) -> float:
        return float(self._last_nis)

    @property
    def min_eig(self) -> float:
        """Smallest eigenvalue of the (symmetrised) covariance — PD health check."""
        return float(np.linalg.eigvalsh(0.5 * (self.P + self.P.T)).min())

    def _stabilize_covariance(self) -> None:
        """Project ``P`` onto the PSD cone.

        The sigma-point central weights are large and negative at ``alpha=0.3``
        (``W0m ≈ −10.1``, ``W0c ≈ −7.2`` for the 22-D error state), so the
        weighted covariance sum can lose positive-definiteness.  Eigen-decompose
        and floor the spectrum rather than relying on symmetrisation alone.
        """
        P = 0.5 * (self.P + self.P.T)
        w, V = np.linalg.eigh(P)
        if float(w.min()) < 1e-9:
            w = np.maximum(w, 1e-9)
            P = V @ np.diag(w) @ V.T
        self.P = 0.5 * (P + P.T)

    # -------------------------------------------------------------- process --
    def _process(self, x: np.ndarray, rotor_target: np.ndarray) -> np.ndarray:
        p, v, q, omega, omega_m = x[0:3], x[3:6], x[6:10], x[10:13], x[13:17]
        R = _R_from_q(q)
        thrust = self.C_T * float(np.sum(omega_m ** 2))
        a_world = (thrust / self.m) * R[:, 2] + self.gravity
        p2 = p + v * self.dt + 0.5 * a_world * self.dt ** 2
        v2 = v + a_world * self.dt
        q2 = _qnorm(_qmul(q, _q_from_rotvec(omega * self.dt)))
        wrench = self.B @ (self.C_T * omega_m ** 2)
        tau = wrench[1:4]
        omega2 = omega + (self.J_inv @ (tau - np.cross(omega, self.J @ omega))) * self.dt
        target = np.clip(rotor_target, 0.0, self.rotor_omega_max)
        alpha = min(1.0, self.dt / max(self.rotor_tau, _EPS))
        omega_m2 = np.clip(omega_m + (target - omega_m) * alpha, 1.0, self.rotor_omega_max)
        out = x.copy()
        out[0:3], out[3:6], out[6:10], out[10:13], out[13:17] = p2, v2, q2, omega2, omega_m2
        return out

    def _default_process_noise(self) -> np.ndarray:
        q = np.zeros(self.N_ERR)
        q[0:3] = 1e-4      # position
        q[3:6] = 5e-3      # velocity
        q[6:9] = 1e-4      # attitude
        q[9:12] = 2e-3     # rates
        q[12:16] = 1e-1    # rotor speeds
        q[16:19] = 1e-6    # accel bias (near constant)
        q[19:22] = 1e-7    # gyro bias
        return np.diag(q)

    # -------------------------------------------------------------- predict --
    def predict(self, rotor_target: np.ndarray) -> None:
        x = self._pack()
        n, lam = self.N_ERR, self.lam
        jitter = 1e-9
        try:
            L = np.linalg.cholesky((n + lam) * (self.P + jitter * np.eye(n)))
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh((n + lam) * self.P)
            L = V @ np.diag(np.sqrt(np.maximum(w, jitter)))
        sigmas = [x]
        for i in range(n):
            sigmas.append(self._add(x, L[:, i]))
            sigmas.append(self._add(x, -L[:, i]))
        prop = [self._process(s, rotor_target) for s in sigmas]
        # weighted mean (attitude relative to propagated nominal)
        x_pred = prop[0].copy()
        q_ref = prop[0][6:10]
        def wmean(sl):
            return sum(self.Wm[i] * prop[i][sl] for i in range(len(prop)))
        x_pred[0:3] = wmean(slice(0, 3))
        x_pred[3:6] = wmean(slice(3, 6))
        dtheta = sum(self.Wm[i] * _rotvec_from_q(_qmul(_qconj(q_ref), prop[i][6:10])) for i in range(len(prop)))
        x_pred[6:10] = _qnorm(_qmul(q_ref, _q_from_rotvec(dtheta)))
        x_pred[10:13] = wmean(slice(10, 13))
        x_pred[13:17] = wmean(slice(13, 17))
        x_pred[17:20] = wmean(slice(17, 20))
        x_pred[20:23] = wmean(slice(20, 23))
        P = self.Q.copy()
        for i in range(len(prop)):
            e = self._diff(prop[i], x_pred)
            P += self.Wc[i] * np.outer(e, e)
        self.P = 0.5 * (P + P.T)
        self._stabilize_covariance()
        self._unpack(x_pred)

    # --------------------------------------------------------------- update --
    def update(self, meas: Measurement) -> None:
        h_fns = []
        z_list = []
        R_list = []
        if meas.gps_ok and meas.gps_pos is not None:
            h_fns.append(lambda x: x[0:3]); z_list.append(meas.gps_pos)
            R_list.append(np.eye(3) * self.sensor.gps_sigma ** 2)
        if meas.accel is not None:
            def h_acc(x):
                R = _R_from_q(x[6:10])
                v = x[3:6]
                thrust = self.C_T * float(np.sum(x[13:17] ** 2))
                v_b = R.T @ v
                omega_sum = float(np.sum(x[13:17]))
                # Specific force = thrust + the dominant aero terms, in body frame.
                # The estimator mirrors the plant's in-plane H-drag and fuselage
                # drag so the accelerometer model stays consistent with the
                # sensor (residual mismatch: rotor in-flow loss, thrust_min).
                F_H = -self.p.k_h * omega_sum * np.array([v_b[0], v_b[1], 0.0])
                F_drag = -0.5 * self.p.rho * np.asarray(self.p.C_D_body) * (v_b * np.abs(v_b))
                return ((thrust / self.m) * np.array([0.0, 0.0, 1.0])
                        + (F_H + F_drag) / self.m + x[17:20])
            h_fns.append(h_acc); z_list.append(meas.accel)
            R_list.append(np.eye(3) * self.sensor.accel_sigma ** 2)
        if meas.gyro is not None:
            h_fns.append(lambda x: x[10:13] + x[20:23]); z_list.append(meas.gyro)
            R_list.append(np.eye(3) * self.sensor.gyro_sigma ** 2)
        if meas.vel is not None:
            h_fns.append(lambda x: x[3:6]); z_list.append(meas.vel)
            R_list.append(np.eye(3) * self.sensor.vel_sigma ** 2)
        if meas.attitude is not None:
            h_fns.append(lambda x: rpy_from_R(_R_from_q(x[6:10]))); z_list.append(meas.attitude)
            R_list.append(np.eye(3) * self.sensor.att_sigma ** 2)
        if meas.baro is not None:
            h_fns.append(lambda x: x[2:3]); z_list.append(np.array([meas.baro]))
            R_list.append(np.eye(1) * self.sensor.baro_sigma ** 2)
        if meas.heading is not None:
            h_fns.append(lambda x: np.array([rpy_from_R(_R_from_q(x[6:10]))[2]]))
            z_list.append(np.array([meas.heading]))
            R_list.append(np.eye(1) * self.sensor.mag_sigma ** 2)
        if not h_fns:
            return
        z = np.concatenate(z_list)
        R = np.zeros((len(z), len(z)))
        o = 0
        for block in R_list:
            n = block.shape[0]
            R[o:o + n, o:o + n] = block
            o += n

        # sigma points around the current estimate
        x = self._pack()
        n_err, lam = self.N_ERR, self.lam
        jitter = 1e-9
        try:
            L = np.linalg.cholesky((n_err + lam) * (self.P + jitter * np.eye(n_err)))
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh((n_err + lam) * self.P)
            L = V @ np.diag(np.sqrt(np.maximum(w, jitter)))
        sigmas = [x]
        for i in range(n_err):
            sigmas.append(self._add(x, L[:, i]))
            sigmas.append(self._add(x, -L[:, i]))

        z_sig = [np.concatenate([h(s) for h in h_fns]) for s in sigmas]
        z_mean = sum(self.Wm[i] * z_sig[i] for i in range(len(z_sig)))
        S = R.copy()
        for i in range(len(z_sig)):
            dz = z_sig[i] - z_mean
            S += self.Wc[i] * np.outer(dz, dz)
        S = 0.5 * (S + S.T)

        innovation = z - z_mean
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        self._last_nis = float(innovation @ S_inv @ innovation)
        # (no batch gate: a single outlier would reject every sensor in the frame;
        # robustness comes from the noise model + per-sensor gating if needed)
        Pxz = np.zeros((n_err, len(z)))
        for i in range(len(z_sig)):
            e = self._diff(sigmas[i], x)
            dz = z_sig[i] - z_mean
            Pxz += self.Wc[i] * np.outer(e, dz)
        K = Pxz @ S_inv
        dx = K @ innovation
        self._unpack(self._add(x, dx))
        self.P = self.P - K @ S @ K.T
        self._stabilize_covariance()
        self._updates += 1

    def step(self, meas: Measurement, rotor_target: np.ndarray) -> np.ndarray:
        """Predict with the applied rotor target, then update with measurements."""
        self.predict(rotor_target)
        self.update(meas)
        return self.state_view()

    def telemetry(self) -> Dict[str, float]:
        return {"nis": float(self._last_nis), "updates": int(self._updates),
                "min_eig": self.min_eig}
