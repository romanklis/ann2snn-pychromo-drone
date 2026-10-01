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

__all__ = ["RefPoint", "Reference", "goal_reference"]


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
