"""Policy input and reference-acceleration feed-forward.

Numpy port of ``BaseController.policy_input`` and ``ReferenceAccelEstimator``
from the ANN2SNN ``drone-example`` branch (MIT); see ``NOTICE``.

The policy input for the ``quad6dof`` example is

    [e (3), ė (3), u_ff (3), task features (4)]        (13,)

with ``e = p - r``, ``ė = v - ṙ``, ``u_ff = â_ref / plant_gain`` (the reference
acceleration reconstructed from the state, ``0`` for the constant goal), and the
four ``ObstacleGoalTask`` columns ``[obs_rel_x, obs_rel_y, clearance, core_radius]``.

Unlike upstream, the state here is the **true** plant state, not a Kalman
estimate (the same simplification the ported DS/PID controllers use).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .params import DT
from .scene import Scene
from .task import ObstacleGoalTask

__all__ = ["error_vector", "ReferenceAccelEstimator", "policy_input", "POLICY_IN_DIM"]


def error_vector(state: np.ndarray, ref, pos_dim: int = 3) -> np.ndarray:
    """``[p - r, v - ṙ]`` for the first ``pos_dim`` axes (dimension ``2·pos_dim``)."""
    d = int(pos_dim)
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    return np.concatenate([state[:d] - np.asarray(ref.pos, dtype=np.float64)[:d],
                           state[d:2 * d] - np.asarray(ref.vel, dtype=np.float64)[:d]])


class ReferenceAccelEstimator:
    """Reference acceleration from a 3-point second difference of ``r̂ = x - e``.

    For a constant goal the estimate settles at zero, so ``u_ff`` is zero; for a
    smooth moving reference it recovers ``r̈``.  ``plant_gain`` is the *signed*
    control-to-acceleration gain (``+1`` for the drone), so ``u_ff = â_ref/gain``.
    """

    def __init__(self, dt: float = DT, plant_gain: float = 1.0, pos_dim: int = 3) -> None:
        self.dt = float(dt)
        self.gain = float(plant_gain)
        self.pos_dim = int(pos_dim)
        self._hist: list = []

    def reset(self) -> None:
        self._hist = []

    def update(self, state: np.ndarray, ref) -> np.ndarray:
        d = self.pos_dim
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        err = error_vector(state, ref, d)
        r_hat = state[:2 * d] - err
        self._hist.append(r_hat.copy())
        if len(self._hist) > 3:
            self._hist.pop(0)
        if len(self._hist) == 3:
            a_hat = (
                self._hist[2][:d] - 2.0 * self._hist[1][:d] + self._hist[0][:d]
            ) / (self.dt ** 2)
        else:
            a_hat = np.zeros(d, dtype=np.float64)
        return a_hat / self.gain


#: policy-input width for the 3-D task: 3·pos_dim + n_features
POLICY_IN_DIM = 3 * 3 + ObstacleGoalTask().n_features


def policy_input(
    state: np.ndarray,
    ref,
    scene: Scene,
    estimator: ReferenceAccelEstimator,
    task: Optional[ObstacleGoalTask] = None,
    *,
    pos_dim: int = 3,
) -> np.ndarray:
    """Assemble the ``(3·pos_dim + F,)`` policy input for one frame."""
    task = task or ObstacleGoalTask()
    d = int(pos_dim)
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    err = error_vector(state, ref, d)
    u_ff = estimator.update(state, ref)
    feats = task._from_positions(state[:d].reshape(1, d), scene)[0]
    return np.concatenate([err, u_ff, feats]).astype(np.float64)
