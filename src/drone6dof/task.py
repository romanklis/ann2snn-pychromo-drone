"""Navigation task contract: obstacle features, telemetry and success mask.

Numpy port of the obstacle/goal parts of ``sim_engine/tasks.py`` on the ANN2SNN
``drone-example`` branch (MIT); see ``NOTICE``.  The policy-input sampling and
distillation-label machinery is training-only and has been omitted.

Features (``F = 4``): obstacle-relative position in the horizontal plane
(obstacle minus drone, i.e. the bearing *toward* the pillar), the clearance to
its surface, and its radius.  The DS guidance law is an exact function of these.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .scene import Scene

__all__ = ["NO_OBSTACLE_CLEARANCE", "ObstacleGoalTask"]

#: clearance reported when a scene has no obstacle at all
NO_OBSTACLE_CLEARANCE: float = 10.0


@dataclass
class ObstacleGoalTask:
    """Reach the goal without entering a vertical cylindrical obstacle."""

    #: flight-box half extents (kept for API parity / telemetry)
    bounds_high: Tuple[float, float, float] = (2.5, 2.5, 3.0)
    goal_tolerance: float = 0.25
    sample_speed: float = 0.8
    n_features: int = 4
    _scene: Optional[Scene] = None

    @staticmethod
    def _from_positions(points: np.ndarray, scene: Scene) -> np.ndarray:
        """``(B, 4)`` feature block for world positions ``(B, 3)``."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        out = np.zeros((len(pts), 4), dtype=np.float64)
        obs = scene.obstacle_np
        if obs is None:
            out[:, 2] = NO_OBSTACLE_CLEARANCE
            return out
        out[:, 0] = obs[0] - pts[:, 0]
        out[:, 1] = obs[1] - pts[:, 1]
        out[:, 2] = np.hypot(pts[:, 0] - obs[0], pts[:, 1] - obs[1]) - scene.core_radius
        out[:, 3] = scene.core_radius
        return out

    def features_from_state(self, state: np.ndarray, scene: Scene) -> np.ndarray:
        p = np.asarray(state, dtype=np.float64).reshape(-1)[:3]
        return self._from_positions(p.reshape(1, 3), scene)[0]

    def frame_telemetry(self, state: np.ndarray, scene: Scene) -> dict:
        p = np.asarray(state, dtype=np.float64).reshape(-1)[:3]
        v = np.asarray(state, dtype=np.float64).reshape(-1)[3:6]
        err = p - scene.goal_np
        return {
            "goal_dist": float(np.linalg.norm(err)),
            "clearance": float(scene.clearance(p)),
            "lyapunov_v": float(0.5 * err @ err),
            "lyapunov_vdot": float(err @ v),
        }

    def success_mask(self, trajectory: np.ndarray, scene: Scene) -> np.ndarray:
        """Success = the obstacle was never entered (the avoidance objective)."""
        return scene.clearance_series(np.asarray(trajectory)) >= 0.0

    def to_dict(self) -> dict:
        return {
            "kind": "obstacle_goal",
            "n_features": int(self.n_features),
            "goal_tolerance": float(self.goal_tolerance),
            "bounds_high": [float(v) for v in self.bounds_high],
            "sample_speed": float(self.sample_speed),
            "feature_names": ["obs_rel_x", "obs_rel_y", "clearance", "obs_radius"],
        }
