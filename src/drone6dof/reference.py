"""Reference trajectories (numpy port of the parts the demo needs).

Ported from ``sim_engine/reference.py`` on the ANN2SNN ``drone-example`` branch
(MIT); see ``NOTICE``.  The navigation task is a constant-goal regulation
problem, so only :func:`goal_reference` and the :class:`Reference` /
:class:`RefPoint` containers are ported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .params import DT
from .planner import PlannerConfig, plan_path

__all__ = ["RefPoint", "Reference", "goal_reference", "path_reference"]


@dataclass
class RefPoint:
    """A single reference sample: position, velocity and acceleration."""

    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pos": [float(v) for v in self.pos],
            "vel": [float(v) for v in self.vel],
            "acc": [float(v) for v in self.acc],
        }


@dataclass
class Reference:
    """A whole reference trajectory (``(T, D)`` position/velocity/acceleration)."""

    pos: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    vel: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    acc: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    dt: float = DT
    radius: Optional[float] = None
    freq: Optional[float] = None
    meta: dict = field(default_factory=dict)

    @property
    def pos_dim(self) -> int:
        return int(self.pos.shape[1]) if self.pos.ndim == 2 else 0

    def __len__(self) -> int:
        return int(self.pos.shape[0])

    def at(self, k: int) -> RefPoint:
        """Reference sample at index ``k`` (clamped to the trajectory)."""
        k = max(0, min(int(k), len(self) - 1))
        return RefPoint(
            pos=self.pos[k], vel=self.vel[k], acc=self.acc[k], meta=self.meta
        )

    def to_dict(self) -> dict:
        return {
            "kind": self.meta.get("kind", "goal"),
            "steps": len(self),
            "pos_dim": self.pos_dim,
            "dt": self.dt,
            "goal": list(self.meta.get("goal", [])),
        }


def goal_reference(
    steps: int = 500,
    goal: Tuple[float, float, float] = (0.0, 0.0, 2.5),
    dt: float = DT,
    *,
    meta: Optional[dict] = None,
) -> Reference:
    """Constant-goal regulation reference: ``r(t) = goal``, ``ṙ = r̈ = 0``."""
    goal_arr = np.asarray(goal, dtype=np.float64).reshape(3)
    pos = np.tile(goal_arr, (int(steps), 1))
    zeros = np.zeros((int(steps), 3))
    base_meta = {
        "kind": "goal",
        "shape": "3d",
        "steps": int(steps),
        "goal": [float(v) for v in goal_arr],
    }
    base_meta.update(meta or {})
    return Reference(pos=pos, vel=zeros, acc=zeros.copy(), dt=float(dt), meta=base_meta)


def path_reference(
    scene,
    start,
    goal,
    *,
    steps: int = 500,
    dt: float = DT,
    speed: float = 1.4,
    brake: float = 0.8,
    planner_config: Optional[PlannerConfig] = None,
    way: Optional[list] = None,
    meta: Optional[dict] = None,
) -> Reference:
    """Time-parameterised reference along a route (falls back to a straight line
    if no route is found).  An explicit ``way`` polyline skips planning.
    ``r(t)``, ``ṙ(t)`` and ``r̈(t)`` come from integrating the trapezoidal speed
    profile along the polyline.
    """
    start = np.asarray(start, dtype=np.float64).reshape(3)
    goal = np.asarray(goal, dtype=np.float64).reshape(3)
    planned = way is not None
    if way is None and getattr(scene, "obstacles", ()):
        way = plan_path(scene, start[:2], goal[:2], planner_config)
        planned = way is not None
    if not way:
        way = [(float(start[0]), float(start[1])), (float(goal[0]), float(goal[1]))]

    xy = np.asarray(way, dtype=np.float64)[:, :2]
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1) if len(xy) > 1 else np.zeros(0)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    frac = cum / (cum[-1] + 1e-9)
    poly = np.column_stack([xy[:, 0], xy[:, 1], start[2] + frac * (goal[2] - start[2])])
    seg3 = np.linalg.norm(np.diff(poly, axis=0), axis=1) if len(poly) > 1 else np.zeros(0)
    cum3 = np.concatenate([[0.0], np.cumsum(seg3)])
    length = float(cum3[-1])

    def sample(s: float) -> Tuple[np.ndarray, np.ndarray]:
        if length < 1e-9:
            return poly[0].copy(), np.zeros(3)
        s = float(np.clip(s, 0.0, length))
        k = int(np.clip(np.searchsorted(cum3, s, side="right") - 1, 0, len(poly) - 2))
        denom = float(cum3[k + 1] - cum3[k])
        t = 0.0 if denom < 1e-9 else (s - cum3[k]) / denom
        seg_vec = poly[k + 1] - poly[k]
        tangent = seg_vec / (float(np.linalg.norm(seg_vec)) + 1e-9)
        return poly[k] + t * seg_vec, tangent

    pos = np.zeros((int(steps), 3))
    vel = np.zeros((int(steps), 3))
    s = 0.0
    for k in range(int(steps)):
        v = min(float(speed), float(np.sqrt(2.0 * brake * max(0.0, length - s))))
        if length - s <= 1e-9:
            v = 0.0
        p, tangent = sample(s)
        pos[k] = p
        vel[k] = v * tangent
        s += v * dt
    acc = np.zeros_like(vel)
    if int(steps) >= 2:
        acc[:-1] = np.diff(vel, axis=0) / dt

    base_meta = {
        "kind": "path",
        "shape": "3d",
        "steps": int(steps),
        "goal": [float(v) for v in goal],
        "planned": bool(planned),
        "waypoints": [[float(x), float(y)] for x, y in way],
        "scene": scene,
    }
    base_meta.update(meta or {})
    return Reference(pos=pos, vel=vel, acc=acc, dt=float(dt), meta=base_meta)
