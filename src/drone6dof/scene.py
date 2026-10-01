"""Task scenes: the world a navigation task lives in (goal + obstacles).

Numpy port of ``sim_engine/scene.py`` from the ANN2SNN ``drone-example`` branch
(MIT); see ``NOTICE``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = ["Scene", "SceneSpec"]


@dataclass(frozen=True)
class Scene:
    """One instantiated scene: goal point and an optional vertical cylinder."""

    goal: Tuple[float, float, float] = (0.0, 0.0, 2.5)
    obstacle: Optional[Tuple[float, float, float]] = None
    core_radius: float = 0.35
    influence_radius: float = 1.05
    dead_radius: float = 0.62
    #: Visual height of the cylinder [m] (the clearance metric is horizontal only).
    obstacle_height: float = 2.8
    seed: int = 0

    @property
    def goal_np(self) -> np.ndarray:
        return np.asarray(self.goal, dtype=np.float64)

    @property
    def obstacle_np(self) -> Optional[np.ndarray]:
        return None if self.obstacle is None else np.asarray(self.obstacle, dtype=np.float64)

    def clearance(self, point) -> float:
        """Horizontal distance from ``point`` to the obstacle surface.

        Positive outside the core, negative inside it, ``inf`` when there is no
        obstacle.  Only the horizontal distance matters: the pillar is vertical.
        """
        obs = self.obstacle_np
        if obs is None:
            return float("inf")
        p = np.asarray(point, dtype=np.float64).reshape(-1)
        return float(np.hypot(p[0] - obs[0], p[1] - obs[1]) - self.core_radius)

    def clearance_series(self, trajectory) -> np.ndarray:
        traj = np.asarray(trajectory)
        if self.obstacle is None:
            return np.full(len(traj), np.inf)
        obs = self.obstacle_np
        return np.hypot(traj[:, 0] - obs[0], traj[:, 1] - obs[1]) - self.core_radius

    def to_dict(self) -> dict:
        return {
            "goal": [float(v) for v in self.goal],
            "obstacle": None if self.obstacle is None else [float(v) for v in self.obstacle],
            "core_radius": float(self.core_radius),
            "influence_radius": float(self.influence_radius),
            "dead_radius": float(self.dead_radius),
            "obstacle_height": float(self.obstacle_height),
            "seed": int(self.seed),
        }


@dataclass(frozen=True)
class SceneSpec:
    """Declarative scene template; ``instantiate`` freezes it for one run."""

    goal: Tuple[float, float, float] = (0.0, 0.0, 2.5)
    obstacle: Optional[Tuple[float, float, float]] = (1.35, 0.0, 1.5)
    core_radius: float = 0.35
    influence_radius: float = 1.05
    dead_radius: float = 0.62
    obstacle_height: float = 2.8
    #: Optional per-axis jitter applied to the obstacle position when ``randomize``.
    obstacle_jitter: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    randomize: bool = False

    def instantiate(self, seed: int = 0) -> Scene:
        obstacle = self.obstacle
        if self.randomize and obstacle is not None:
            rng = np.random.default_rng(seed)
            jitter = rng.uniform(-1.0, 1.0, size=3) * np.asarray(self.obstacle_jitter)
            obstacle = tuple(float(v) for v in (np.asarray(obstacle) + jitter))
        return Scene(
            goal=self.goal,
            obstacle=obstacle,
            core_radius=float(self.core_radius),
            influence_radius=float(self.influence_radius),
            dead_radius=float(self.dead_radius),
            obstacle_height=float(self.obstacle_height),
            seed=int(seed),
        )

    def to_dict(self) -> dict:
        return {
            "goal": [float(v) for v in self.goal],
            "obstacle": None if self.obstacle is None else [float(v) for v in self.obstacle],
            "core_radius": float(self.core_radius),
            "influence_radius": float(self.influence_radius),
            "dead_radius": float(self.dead_radius),
            "obstacle_height": float(self.obstacle_height),
            "obstacle_jitter": [float(v) for v in self.obstacle_jitter],
            "randomize": bool(self.randomize),
        }
