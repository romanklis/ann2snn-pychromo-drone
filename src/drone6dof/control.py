"""Controllers (numpy ports of the ANN2SNN ``ds_guidance`` and ``classical``).

Ported from ``sim_engine/controllers/ds_guidance.py`` and
``sim_engine/controllers/classical.py`` on the ``drone-example`` branch (MIT);
see ``NOTICE``.  Both act on the **true** plant state (no Kalman estimate),
which matches the vendored prototype oracle.

The command is an acceleration demand ``u`` (m/s²) excluding gravity: the plant
closes ``F_des = m·(u + g·ẑ)``, so hover is ``u = 0``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from .params import QuadParams
from .scene import Scene
from .task import ObstacleGoalTask

__all__ = ["DSGuidanceController", "ClassicalPDController"]

_EPS = 1e-6


def _scene_of(ref, fallback: Optional[Scene]) -> Scene:
    meta = getattr(ref, "meta", None) or {}
    scene = meta.get("scene")
    if scene is not None:
        return scene
    if fallback is not None:
        return fallback
    return Scene()


class DSGuidanceController:
    """Obstacle-avoiding dynamical-system guidance (the navigation teacher).

    Billard-modulated vector field followed by a passive impedance law.  The
    output is an acceleration demand in m/s².
    """

    name = "ds_guidance"
    description = "Billard dynamical-system guidance + impedance (teacher)"

    def __init__(
        self,
        mass: float = QuadParams().m,
        speed_cap: float = 2.2,
        damping_along: float = 1.2,
        damping_across: float = 4.5,
        ds_gain: float = 1.35,
        ds_radial_gain: float = 0.35,
        ds_climb_gain: float = 0.45,
        action_limit: float = 20.0,
        pos_dim: int = 3,
        scene: Optional[Scene] = None,
        task: Optional[ObstacleGoalTask] = None,
    ) -> None:
        self.mass = float(mass)
        self.speed_cap = float(speed_cap)
        self.damping_along = float(damping_along)
        self.damping_across = float(damping_across)
        self.ds_gain = float(ds_gain)
        self.ds_radial_gain = float(ds_radial_gain)
        self.ds_climb_gain = float(ds_climb_gain)
        self.action_limit = float(action_limit)
        self.pos_dim = int(pos_dim)
        self.scene = scene
        self.task = task or ObstacleGoalTask()
        self._last: dict = {}

    # ------------------------------------------------------------------ scene
    def _features_from_state(self, state: np.ndarray, scene: Scene) -> np.ndarray:
        p = np.asarray(state, dtype=np.float64).reshape(-1)[:3]
        return self.task._from_positions(p.reshape(1, 3), scene)[0]

    # ----------------------------------------------------------------- core
    def _core(self, e: np.ndarray, edot: np.ndarray, position: np.ndarray, scene: Scene):
        """The guidance law for a single ``(3,)`` sample. Returns ``u`` (m/s²)."""
        goal_dist = float(np.linalg.norm(e))
        # ---- nominal DS velocity field (rotational + radial attraction) ---- #
        vx = -self.ds_radial_gain * e[0] - self.ds_gain * e[1]
        vy = -self.ds_radial_gain * e[1] + self.ds_gain * e[0]
        vz = -self.ds_climb_gain * e[2]
        v_nom = np.array([vx, vy, vz], dtype=np.float64)
        speed = float(np.linalg.norm(v_nom))
        v_nom = v_nom * min(self.speed_cap / (speed + _EPS), 1.0)

        # ---- obstacle modulation (matrix-free M_i = I + w_i(E_i Λ_i E_iᵀ − I)),
        # composed over the scene's obstacles (v ← M_k(…M_1 f)).  Each obstacle
        # uses its closest-point outward normal, so boxes and cylinders share the
        # same law; a cylinder's core_offset reproduces the ported pillar exactly.
        v_des = v_nom.copy()
        p_xy = np.asarray(position, dtype=np.float64)[:2]
        active = 0.0
        for obstacle in getattr(scene, "obstacles", ()) or ():
            _, n_xy, sdf = obstacle.closest_point_normal(p_xy)
            dist_ref = float(sdf) + float(obstacle.core_offset)
            influence = float(obstacle.influence_radius)
            dead = float(obstacle.dead_radius)
            if dist_ref >= influence:
                continue
            n = np.array([n_xy[0], n_xy[1], 0.0])
            t = np.array([-n[1], n[0], 0.0])
            if float(v_des @ t) < 0.0:
                t = -t
            gamma = (dist_ref / dead) ** 2
            lam_r = 1.0 - 1.0 / max(gamma, 0.05)
            lam_t = 1.0 + 0.8 / max(gamma, 0.05)
            s = float(np.clip((influence - dist_ref) / (influence - dead), 0.0, 1.0))
            w = math.sin(math.pi * 0.5 * s) ** 2
            v_des = v_des + w * (
                (lam_r - 1.0) * float(v_des @ n) * n
                + (lam_t - 1.0) * float(v_des @ t) * t
            )
            active = 1.0

        # ---- passive impedance toward the DS velocity --------------------- #
        dv = edot - v_des
        vh = v_des / (float(np.linalg.norm(v_des)) + _EPS)
        along = float(dv @ vh)
        accel = -(
            self.damping_along * along * vh
            + self.damping_across * (dv - along * vh)
        )
        u = accel / self.mass

        cos_deflect = float(v_nom @ v_des) / (
            float(np.linalg.norm(v_nom)) * float(np.linalg.norm(v_des)) + _EPS
        )
        self._last = {
            "ds_deflection_deg": float(
                np.degrees(np.arccos(np.clip(cos_deflect, -1.0, 1.0)))
            ),
            "ds_goal_dist": goal_dist,
            "ds_active": active,
            "ds_v_des": float(np.linalg.norm(v_des)),
        }
        return u

    # ------------------------------------------------------------------ act
    def raw_act(self, state: np.ndarray, ref) -> np.ndarray:
        if self.pos_dim != 3:
            raise ValueError(
                "ds_guidance is a 3-D navigation teacher; it does not apply to "
                f"pos_dim={self.pos_dim} examples"
            )
        scene = _scene_of(ref, self.scene)
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        e = state[:3] - np.asarray(ref.pos, dtype=np.float64)
        edot = state[3:6] - np.asarray(ref.vel, dtype=np.float64)
        return self._core(e, edot, state[:3], scene)

    def act(self, state: np.ndarray, ref) -> np.ndarray:
        return np.clip(self.raw_act(state, ref), -self.action_limit, self.action_limit)

    def reset(self) -> None:
        self._last = {}

    @property
    def last_telemetry(self) -> dict:
        return dict(self._last)


class ClassicalPDController:
    """Classical PD with acceleration feed-forward (the baseline).

    ``u_i = kp·e_i + kd·ė_i + a_ref_i / G`` with ``kp = ω_n²/|G|`` and
    ``kd = 2ζω_n/|G|`` from a target natural frequency and damping ratio.
    """

    name = "pid"
    description = "Classical PD with acceleration feed-forward"

    def __init__(
        self,
        omega_n: float = 3.5,
        zeta: float = 0.85,
        plant_gain: float = 1.0,
        action_limit: float = 12.0,
        use_feedforward: bool = True,
        pos_dim: int = 3,
    ) -> None:
        self.omega_n = float(omega_n)
        self.zeta = float(zeta)
        self.plant_gain = float(plant_gain)
        self.action_limit = float(action_limit)
        self.use_feedforward = bool(use_feedforward)
        self.pos_dim = int(pos_dim)
        mag = abs(self.plant_gain)
        self.kp = (self.omega_n ** 2) / mag
        self.kd = (2.0 * self.zeta * self.omega_n) / mag

    def raw_act(self, state: np.ndarray, ref) -> np.ndarray:
        d = self.pos_dim
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        e = state[:d] - np.asarray(ref.pos, dtype=np.float64)[:d]
        edot = state[d:2 * d] - np.asarray(ref.vel, dtype=np.float64)[:d]
        accel = -(self.omega_n ** 2 * e + 2.0 * self.zeta * self.omega_n * edot)
        if self.use_feedforward:
            accel = accel + np.asarray(ref.acc, dtype=np.float64)[:d]
        return accel / self.plant_gain

    def act(self, state: np.ndarray, ref) -> np.ndarray:
        return np.clip(self.raw_act(state, ref), -self.action_limit, self.action_limit)

    def reset(self) -> None:
        pass

    @property
    def last_telemetry(self) -> dict:
        return {}
