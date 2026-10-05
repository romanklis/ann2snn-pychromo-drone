"""6-DoF quadcopter plant: attitude, rotor/fuselage aero, motors and battery.

Numpy port of ``sim_engine/plants/quad6dof.py`` from the ANN2SNN
``drone-example`` branch (MIT); see ``NOTICE``.  Every equation, clip, ZOH
update and the 500 Hz inner substepping are preserved.  Only the torch boundary
is replaced with numpy: :meth:`Quad6DoF.__call__` takes and returns plain
``float64`` arrays.

The translational state ``[p, v]`` is what the caller sees; the rotation ``R``,
body rates ``ω``, four motor speeds and currents, and the state of charge live
inside the object as hidden state and are exposed through
:attr:`Quad6DoF.attitude_rpy` and :attr:`Quad6DoF.last_telemetry`.

Gravity lives here: the command is an *acceleration demand* ``u`` (m/s²) that
excludes gravity; the plant closes the loop as ``F_des = m·(u + g·ẑ)``.  Hover is
therefore ``u = 0``.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .params import QuadParams
from .so3 import reorthonormalize, rpy_from_R, rodrigues

__all__ = ["Quad6DoF", "QuadParams"]

_EPS = 1e-9


class Quad6DoF:
    """Stateful 6-DoF plant step; call it once per pipeline frame.

    A fresh instance is created per run so that the hidden attitude/motor state
    is isolated between controllers.
    """

    def __init__(self, params: Optional[QuadParams] = None) -> None:
        self.p = params or QuadParams()
        self._g = float(self.p.g)
        self.gravity_up = np.array([0.0, 0.0, self._g], dtype=np.float64)
        #: Constant world heading used to build the desired attitude.
        self.heading_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        #: Optional world point the vehicle keeps its nose pointed at (the
        #: prototype's heading blend).  ``None`` keeps the fixed ``heading_dir``.
        self.heading_target: Optional[np.ndarray] = None
        self.heading_d_min: float = 0.15
        self.heading_d_max: float = 0.60
        self.J = np.diag(self.p.J_diag)
        self.J_inv = np.linalg.inv(self.J)
        # Quad-X mixer [thrust, roll, pitch, yaw] -> 4 rotor thrusts
        d_arm = self.p.d / np.sqrt(2.0)
        c_ratio = self.p.C_Q / self.p.C_T
        self.B = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [-d_arm, d_arm, d_arm, -d_arm],
                [-d_arm, d_arm, -d_arm, d_arm],
                [-c_ratio, -c_ratio, c_ratio, c_ratio],
            ],
            dtype=np.float64,
        )
        self.B_inv = np.linalg.inv(self.B)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        p = self.p
        self.R = np.eye(3)
        self.omega = np.zeros(3)
        self.omega_m = np.full(4, float(p.initial_omega_m))
        self.current = np.full(4, float(p.initial_current))
        self.tau_att = np.zeros(3)
        self.soc = float(p.soc0)
        self._last: Dict[str, float] = {}
        self._last_omega_target = np.full(4, float(p.initial_omega_m))
        self.frames = 0
        self.substeps = 0

    # -------------------------------------------------------------- reporting
    @property
    def attitude_rpy(self) -> np.ndarray:
        """Current (roll, pitch, yaw) in radians."""
        return np.asarray(rpy_from_R(self.R))

    @property
    def last_telemetry(self) -> Dict[str, float]:
        return dict(self._last)

    @property
    def last_rotor_target(self) -> np.ndarray:
        """Applied/commanded rotor speeds (rad/s) from the last substep."""
        return self._last_omega_target.copy()

    # ------------------------------------------------------------------- step
    def __call__(
        self,
        state: np.ndarray,
        command: np.ndarray,
        *,
        dt: float,
        limit: float,
        gain: float,
        damping: float = 0.0,
        disturbance: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        p = self.p
        state_np = np.asarray(state, dtype=np.float64).reshape(-1)
        pos = state_np[:3].copy()
        vel = state_np[3:6].copy()

        cmd = np.asarray(command, dtype=np.float64).reshape(-1)
        u = np.clip(cmd[:3], -float(limit), float(limit))
        authority = max(float(gain), 1e-6)           # rotor-efficiency multiplier
        if disturbance is None:
            dist = np.zeros(3)
        else:
            dist = np.asarray(disturbance, dtype=np.float64).reshape(-1)[:3]
        damping = float(damping)                     # extra linear drag [1/s]

        n = max(1, int(round(float(dt) * float(p.inner_hz))))
        h = float(dt) / n
        gravity = np.array([0.0, 0.0, -self._g])
        dist_force = self.p.m * dist

        for _ in range(n):
            # ---- outer-loop command (zero-order hold across the substeps) ---- #
            F_des = p.m * (u + self.gravity_up)
            f_norm = math.sqrt(F_des[0] * F_des[0] + F_des[1] * F_des[1] + F_des[2] * F_des[2])
            z_b_des = F_des / (f_norm + _EPS)
            # Heading: blend a look-at-target direction with a fixed forward when
            # the example sets ``heading_target``.
            if self.heading_target is not None:
                to_goal = self.heading_target - pos
                dist_goal = math.sqrt(float(to_goal @ to_goal))
                if dist_goal > self.heading_d_min:
                    look = to_goal / dist_goal
                else:
                    look = np.array([1.0, 0.0, 0.0])
                sblend = min(
                    max(
                        (dist_goal - self.heading_d_min)
                        / (self.heading_d_max - self.heading_d_min),
                        0.0,
                    ),
                    1.0,
                )
                wblend = sblend * sblend * (3.0 - 2.0 * sblend)
                blended = wblend * look + (1.0 - wblend) * np.array([1.0, 0.0, 0.0])
                cam = blended / (math.sqrt(float(blended @ blended)) + _EPS)
            else:
                cam = self.heading_dir
            c0, c1, c2 = cam
            y0 = z_b_des[1] * c2 - z_b_des[2] * c1
            y1 = z_b_des[2] * c0 - z_b_des[0] * c2
            y2 = z_b_des[0] * c1 - z_b_des[1] * c0
            y_norm = math.sqrt(y0 * y0 + y1 * y1 + y2 * y2)
            y0, y1, y2 = y0 / (y_norm + _EPS), y1 / (y_norm + _EPS), y2 / (y_norm + _EPS)
            x0 = y1 * z_b_des[2] - y2 * z_b_des[1]
            x1 = y2 * z_b_des[0] - y0 * z_b_des[2]
            x2 = y0 * z_b_des[1] - y1 * z_b_des[0]
            R_d = np.array(
                [
                    [x0, y0, z_b_des[0]],
                    [x1, y1, z_b_des[1]],
                    [x2, y2, z_b_des[2]],
                ]
            )

            # SO(3) attitude error ``e_R = ½·vee(R_dᵀ R − Rᵀ R_d)`` (Lee et al.)
            S = R_d.T @ self.R - self.R.T @ R_d
            e_R0, e_R1, e_R2 = 0.5 * S[2, 1], 0.5 * S[0, 2], 0.5 * S[1, 0]

            raw0 = -p.att_kp * e_R0
            raw1 = -p.att_kp * e_R1
            raw2 = -p.att_kp_yaw * e_R2
            blend = 1.0 - math.exp(-h / p.att_filter_tau)
            self.tau_att[0] += (raw0 - self.tau_att[0]) * blend
            self.tau_att[1] += (raw1 - self.tau_att[1]) * blend
            self.tau_att[2] += (raw2 - self.tau_att[2]) * blend
            Jw = self.J @ self.omega
            gyro_x = self.omega[1] * Jw[2] - self.omega[2] * Jw[1]
            gyro_y = self.omega[2] * Jw[0] - self.omega[0] * Jw[2]
            gyro_z = self.omega[0] * Jw[1] - self.omega[1] * Jw[0]
            tau0 = self.tau_att[0] - p.att_kd * (self.omega[0] + p.att_rate_ref * e_R0) + gyro_x
            tau1 = self.tau_att[1] - p.att_kd * (self.omega[1] + p.att_rate_ref * e_R1) + gyro_y
            tau2 = (
                self.tau_att[2]
                + float(
                    np.clip(
                        -p.att_kd_yaw * (self.omega[2] + p.att_rate_ref * e_R2),
                        -p.tau_yaw_limit,
                        p.tau_yaw_limit,
                    )
                )
                + gyro_z
            )
            # Demanded achieved collective thrust, and the rotor thrust that
            # produces it at the current efficiency (clipped in rotor terms, so a
            # derated body saturates physically without acquiring a vertical bias).
            T_demand = float(F_des @ self.R[:, 2])
            T_rotor_cmd = float(
                np.clip(
                    T_demand / authority,
                    p.thrust_cmd_min,
                    p.thrust_cmd_max_scale * p.m * self._g,
                )
            )

            targets = self.B_inv @ np.array([T_rotor_cmd, tau0, tau1, tau2])
            targets = p.mixer_center + p.mixer_scale * np.tanh(
                (targets - p.mixer_center) / p.mixer_scale
            )
            omega_target = np.sqrt(np.maximum(targets, 0.0) / p.C_T)
            self._last_omega_target = omega_target.copy()
            tau_ff = p.d_m * omega_target + p.C_Q * omega_target ** 2
            i_target = np.clip(
                (tau_ff + 0.003 * (omega_target - self.omega_m)) / p.K_t,
                0.0,
                p.current_max,
            )

            # ---- battery ------------------------------------------------------ #
            i_total = float(np.sum(self.current)) + p.I_avionics
            v_ocv = (
                p.V_min_ocv
                + 3.0 * np.clip(self.soc, 0.0, 1.0)
                + 0.6 * np.clip(self.soc, 0.0, 1.0) ** 2
            )
            v_term = v_ocv - i_total * p.R_int
            self.soc -= (i_total / (p.capacity_Ah * 3600.0)) * h
            omega_max_v = np.maximum(
                (v_term - self.current * p.R_coil) / p.K_e, p.omega_voltage_floor
            )

            # ---- motor --------------------------------------------------------- #
            # Current loop uses its exact ZOH update; the mechanical speed loop is
            # slow enough that explicit Euler is stable (ported verbatim).
            self.current += (i_target - self.current) * (1.0 - math.exp(-h / p.tau_e))
            np.clip(self.current, 0.0, p.current_max, out=self.current)
            tau_motor = p.K_t * self.current
            tau_drag = p.d_m * self.omega_m + p.C_Q * self.omega_m ** 2
            self.omega_m += ((tau_motor - tau_drag) / p.J_m) * h
            np.clip(self.omega_m, p.omega_min, omega_max_v, out=self.omega_m)

            # ---- rotor + fuselage aerodynamics ------------------------------- #
            v_body = self.R.T @ vel
            delta_f = p.k_vz * self.omega_m * v_body[2]
            f_actual = (
                np.maximum(p.C_T * self.omega_m ** 2 - delta_f, p.thrust_min) * authority
            )
            F_H = -p.k_h * float(np.sum(self.omega_m)) * np.array([v_body[0], v_body[1], 0.0])
            F_drag = -0.5 * p.rho * np.asarray(p.C_D_body) * (v_body * np.abs(v_body))

            wrench = self.B @ f_actual
            tau_aero = wrench[1:4] - np.asarray(p.D_rot_aero) * self.omega
            tau_gyro = -np.cross(self.omega, self.J @ self.omega)

            # ---- rigid body --------------------------------------------------- #
            self.omega = self.omega + (self.J_inv @ (tau_aero + tau_gyro)) * h
            self.R = self.R @ rodrigues(self.omega * h)

            T_actual = float(wrench[0])
            F_thrust_world = T_actual * self.R[:, 2]
            F_damp = -p.m * damping * vel            # extra linear drag (embodiment)
            F_net = (
                F_thrust_world
                + self.R @ (F_H + F_drag)
                + gravity * p.m
                + dist_force
                + F_damp
            )
            vel = vel + (F_net / p.m) * h
            pos = pos + vel * h
            self.substeps += 1

        # ---- telemetry, once per pipeline frame (as the prototype logs it) ----- #
        v_in_plane = float(np.hypot(v_body[0], v_body[1]))
        omega_mean = float(np.mean(self.omega_m))
        margin = (T_actual / T_demand - 1.0) if abs(T_demand) > _EPS else 0.0
        saturated = (
            1.0
            if (
                margin < -0.02
                or T_demand / authority > p.thrust_cmd_max_scale * p.m * self._g * (1.0 + 1e-9)
            )
            else 0.0
        )
        self._last = {
            "g_force": float(np.linalg.norm(F_thrust_world) / (p.m * self._g)),
            "thrust_n": T_actual,
            "power_w": float(i_total * v_term),
            "soc_pct": float(self.soc * 100.0),
            "v_term": float(v_term),
            "v_ocv": float(v_ocv),
            "advance_ratio": float(v_in_plane / (omega_mean * p.R_prop + _EPS)),
            "omega_mean": omega_mean,
            "current_mean": float(np.mean(self.current)),
            "tau_aero_norm": float(np.linalg.norm(tau_aero)),
            "tau_gyro_norm": float(np.linalg.norm(tau_gyro)),
            "inflow_loss": float(np.sum(delta_f)),
            "thrust_demand_n": float(T_demand),
            "thrust_achieved_n": float(T_actual),
            "thrust_margin_frac": float(margin),
            "thrust_saturated": float(saturated),
        }
        for i in range(4):
            self._last[f"rpm{i + 1}"] = float(self.omega_m[i] * 60.0 / (2.0 * np.pi))
            self._last[f"current{i + 1}"] = float(self.current[i])

        self.R = reorthonormalize(self.R)
        self.frames += 1
        return np.concatenate([pos, vel])

    # ------------------------------------------------------------- plumbing
    def step_fn(self) -> Callable:
        """Convenience: the callable bound method (matches the source API shape)."""
        return self.__call__
