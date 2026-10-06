"""Global path guidance and the privileged local supervisor.

Separation of roles (plan D):

* **global** — :class:`PathTracker` follows the A* route with a path-tracking DS
  ``v_nom = ṙ_ref + k(r_ref − p)`` (look-ahead, capped);
* **local** — the teacher's per-obstacle modulation supplies the *labels* the
  learned LiDAR field is distilled from (:func:`supervisor_v_des`), and the
  runtime field controller reproduces it without oracle geometry.

:class:`SupervisorController` is the privileged rollout policy used to generate
training trajectories/targets; it is not used online.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .field import FieldConfig, modulate_ds, teacher_grad
from .params import DT
from .planner import PlannerConfig
from .reference import path_reference

__all__ = ["PathTracker", "teacher_field", "supervisor_v_des", "SupervisorController"]

_EPS = 1e-6


def teacher_field(scene, point, config: Optional[FieldConfig] = None) -> np.ndarray:
    """Obstacle field ``F* = −∇U*`` from privileged geometry."""
    return -teacher_grad(scene, point, config)


def supervisor_v_des(
    scene, point, v_nom: np.ndarray, config: Optional[FieldConfig] = None
) -> np.ndarray:
    """Privileged local avoidance: the DS modulated by the teacher field."""
    return modulate_ds(v_nom, teacher_field(scene, point, config), config)


class PathTracker:
    """Track a planned route with a look-ahead path-tracking DS."""

    def __init__(
        self,
        scene,
        start,
        goal,
        *,
        speed: float = 1.4,
        k_path: float = 1.0,
        lookahead: float = 0.5,
        brake: float = 0.8,
        speed_cap: Optional[float] = None,
        planner_config: Optional[PlannerConfig] = None,
        steps: int = 1500,
        dt: float = DT,
        way: Optional[list] = None,
    ) -> None:
        self.scene = scene
        self.goal = np.asarray(goal, dtype=np.float64).reshape(3)
        self.start = np.asarray(start, dtype=np.float64).reshape(3)
        self.k_path = float(k_path)
        self.lookahead = float(lookahead)
        self.speed_cap = float(speed_cap if speed_cap is not None else speed)
        self._speed = float(speed)
        self._brake = float(brake)
        self._steps = int(steps)
        self._dt = float(dt)
        self._planner_config = planner_config
        self.ref = path_reference(
            scene, self.start, self.goal, steps=steps, dt=dt, speed=speed,
            brake=brake, planner_config=planner_config, way=way,
        )
        self.pos = np.asarray(self.ref.pos, dtype=np.float64)
        self.vel = np.asarray(self.ref.vel, dtype=np.float64)
        self._idx = 0

    @property
    def waypoints(self):
        return list(self.ref.meta.get("waypoints", []))

    @property
    def planned(self) -> bool:
        return bool(self.ref.meta.get("planned", False))

    def reset(self) -> None:
        self._idx = 0

    def set_path(self, way, start=None) -> None:
        """Replace the tracked route (replanning) and reset the progress index."""
        if start is not None:
            self.start = np.asarray(start, dtype=np.float64).reshape(3)
        self.ref = path_reference(
            self.scene, self.start, self.goal, steps=self._steps, dt=self._dt,
            speed=self._speed, brake=self._brake, planner_config=self._planner_config,
            way=way,
        )
        self.pos = np.asarray(self.ref.pos, dtype=np.float64)
        self.vel = np.asarray(self.ref.vel, dtype=np.float64)
        self._idx = 0

    def v_nom(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64).reshape(-1)[:3]
        n = len(self.pos)
        if n == 0:
            return np.zeros(3)
        lo = max(0, self._idx - 30)
        hi = min(n - 1, self._idx + 200)
        seg = self.pos[lo:hi + 1]
        i = lo + int(np.argmin(np.sum((seg - p) ** 2, axis=1)))
        self._idx = i
        acc = 0.0
        j = i
        while j < n - 1 and acc < self.lookahead:
            acc += float(np.linalg.norm(self.pos[j + 1] - self.pos[j]))
            j += 1
        v = self.vel[i] + self.k_path * (self.pos[j] - p)
        sp = float(np.linalg.norm(v))
        if sp > self.speed_cap:
            v = v * (self.speed_cap / sp)
        return v


class SupervisorController:
    """Privileged path-tracking + teacher-modulated policy (training labels)."""

    name = "ds_path"
    description = "A* path tracking + teacher obstacle modulation (supervisor)"

    def __init__(
        self,
        scene,
        start,
        goal,
        *,
        field_config: Optional[FieldConfig] = None,
        guidance: Optional[dict] = None,
        action_limit: float = 12.0,
        mass: float = 0.5,
        damping_along: float = 1.2,
        damping_across: float = 4.5,
        planner_config: Optional[PlannerConfig] = None,
    ) -> None:
        cfg = guidance or {}
        self.scene = scene
        self.cfg = field_config or FieldConfig()
        self.action_limit = float(action_limit)
        self.mass = float(mass)
        self.damping_along = float(damping_along)
        self.damping_across = float(damping_across)
        self.tracker = PathTracker(
            scene, start, goal,
            speed=float(cfg.get("speed", 1.4)),
            k_path=float(cfg.get("k_path", 1.0)),
            lookahead=float(cfg.get("lookahead", 0.5)),
            brake=float(cfg.get("brake", 0.8)),
            planner_config=planner_config,
        )

    def reset(self) -> None:
        self.tracker.reset()

    def act(self, state, ref=None) -> np.ndarray:
        state = np.asarray(getattr(state, "state", state), dtype=np.float64).reshape(-1)
        v_nom = self.tracker.v_nom(state[:3])
        v_des = supervisor_v_des(self.scene, state[:3], v_nom, self.cfg)
        edot = state[3:6]
        dv = edot - v_des
        vh = v_des / (float(np.linalg.norm(v_des)) + _EPS)
        along = float(dv @ vh)
        accel = -(self.damping_along * along * vh
                  + self.damping_across * (dv - along * vh))
        return np.clip(accel / self.mass, -self.action_limit, self.action_limit)
