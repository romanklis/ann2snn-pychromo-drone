"""Task scenes: the world a navigation task lives in (goal + obstacles).

Numpy port of ``sim_engine/scene.py`` from the ANN2SNN ``drone-example`` branch
(MIT); see ``NOTICE``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .geometry import Cylinder

__all__ = ["Scene", "SceneSpec"]


@dataclass(frozen=True)
class Scene:
    """One instantiated scene: goal point and a set of vertical obstacles."""

    goal: Tuple[float, float, float] = (0.0, 0.0, 2.5)
    obstacle: Optional[Tuple[float, float, float]] = None
    core_radius: float = 0.35
    influence_radius: float = 1.05
    dead_radius: float = 0.62
    #: Visual height of the cylinder [m] (the clearance metric is horizontal only).
    obstacle_height: float = 2.8
    seed: int = 0
    #: Explicit obstacle geometries (boxes/cylinders).  The legacy single
    #: ``obstacle`` cylinder is materialised here when this is empty.
    obstacles: Tuple = ()

    def __post_init__(self) -> None:
        if not self.obstacles and self.obstacle is not None:
            cyl = Cylinder(
                center=(float(self.obstacle[0]), float(self.obstacle[1])),
                radius=float(self.core_radius),
                height=float(self.obstacle_height),
                core_offset=float(self.core_radius),
                # centre-based thresholds reproduce the ported behaviour exactly
                dead_radius=float(self.dead_radius),
                influence_radius=float(self.influence_radius),
            )
            object.__setattr__(self, "obstacles", (cyl,))

    @property
    def goal_np(self) -> np.ndarray:
        return np.asarray(self.goal, dtype=np.float64)

    @property
    def obstacle_np(self) -> Optional[np.ndarray]:
        return None if self.obstacle is None else np.asarray(self.obstacle, dtype=np.float64)

    def clearance(self, point) -> float:
        """Minimum signed distance to any obstacle surface (``inf`` with none)."""
        if not self.obstacles:
            return float("inf")
        return float(min(o.signed_distance(point) for o in self.obstacles))

    def clearance_series(self, trajectory) -> np.ndarray:
        traj = np.asarray(trajectory)
        if not self.obstacles:
            return np.full(len(traj), np.inf)
        return np.min(
            np.stack([o.signed_distance_batch(traj) for o in self.obstacles]), axis=0
        )

    def to_dict(self) -> dict:
        return {
            "goal": [float(v) for v in self.goal],
            "obstacle": None if self.obstacle is None else [float(v) for v in self.obstacle],
            "obstacles": [o.to_dict() for o in self.obstacles],
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
