"""Physical constants and quadcopter parameters (numpy port).

Constants are ported from ``sim_engine/physics.py`` and the defaults from
``sim_engine/plants/quad6dof.py`` on the ANN2SNN ``drone-example`` branch
(MIT); see ``NOTICE``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = ["GRAVITY", "DT", "MAX_THRUST", "DRONE_HALF", "QuadParams"]

GRAVITY: float = 9.81
"""Gravitational acceleration [m/s^2]."""

DT: float = 0.02
"""Outer (control / visualization) timestep [s] — 50 Hz."""

MAX_THRUST: float = 3.0
"""Nominal actuator saturation for the point-mass drone [m/s^2]."""

DRONE_HALF: float = 1.0
"""Half-extent of the point-mass drone's corridor [m]."""


@dataclass
class QuadParams:
    """Physical parameters (defaults are the prototype's 5-inch quad)."""

    # -- body --------------------------------------------------------------- #
    m: float = 0.5
    g: float = GRAVITY
    J: Tuple[float, float, float] = (2.3e-3, 2.3e-3, 4.0e-3)
    d: float = 0.15                      # centre-to-motor arm length [m]
    # -- fuselage aerodynamics ---------------------------------------------- #
    rho: float = 1.225
    C_D_body: Tuple[float, float, float] = (0.022, 0.022, 0.048)
    D_rot_aero: Tuple[float, float, float] = (0.0012, 0.0012, 0.0025)
    # -- rotor blade-element coefficients ----------------------------------- #
    C_T: float = 1.5e-5
    C_Q: float = 2.5e-7
    R_prop: float = 0.0635
    k_vz: float = 1.4e-4                 # axial climb inflow penalty
    k_h: float = 1.6e-6                  # rotor in-plane ("H") drag
    # -- motor -------------------------------------------------------------- #
    J_m: float = 1.2e-5
    K_t: float = 0.015
    K_e: float = 0.015
    d_m: float = 1.5e-5
    R_coil: float = 0.18
    tau_e: float = 0.002                 # electrical time constant [s]
    # -- battery (4S LiPo equivalent circuit) -------------------------------- #
    capacity_Ah: float = 1.3
    R_int: float = 0.035
    V_min_ocv: float = 13.2
    V_max_ocv: float = 16.8
    I_avionics: float = 0.25
    soc0: float = 0.95
    # -- limits (from the prototype) ---------------------------------------- #
    thrust_cmd_min: float = 1.0
    thrust_cmd_max_scale: float = 2.0    # clip to 2·m·g
    tau_yaw_limit: float = 0.015
    current_max: float = 6.5
    omega_min: float = 10.0
    omega_voltage_floor: float = 50.0
    thrust_min: float = 0.05
    mixer_center: float = 1.35
    mixer_scale: float = 1.15
    # -- attitude autopilot (the prototype's outer attitude law) ------------- #
    att_kp: float = 0.08                 # roll/pitch proportional torque
    att_kp_yaw: float = 0.015
    att_kd: float = 0.022                # roll/pitch rate loop
    att_kd_yaw: float = 0.008
    att_rate_ref: float = 6.0            # ω_des = −att_rate_ref · e_R
    att_filter_tau: float = 0.015        # first-order filter on the PD torque
    # -- integration -------------------------------------------------------- #
    inner_hz: float = 500.0              # inner substep rate (validated)
    initial_omega_m: float = 286.0
    initial_current: float = 1.65

    @property
    def J_diag(self) -> np.ndarray:
        return np.asarray(self.J, dtype=np.float64)

    @property
    def hover_omega(self) -> float:
        """Rotor speed whose collective thrust balances gravity [rad/s]."""
        return float(np.sqrt(self.m * self.g / (4.0 * self.C_T)))
