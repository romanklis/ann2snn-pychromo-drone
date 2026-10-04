"""Policy input and reference-acceleration feed-forward.

Numpy port of ``BaseController.policy_input`` / ``ReferenceAccelEstimator`` from
the ANN2SNN ``drone-example`` branch (MIT); see ``NOTICE``.

The obstacle part of the input is now **sensor-derived** (a horizontal LiDAR
scan + cues) rather than oracle geometry: the policy sees

    [e (3), ė (3), u_ff (3), ranges (k), min_range, bearing (2), slope, curvature]

so it must infer the obstacle shape from the scan.  The reference-accel
feed-forward is ``u_ff = â_ref / plant_gain`` (zero for a constant goal).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .params import DT
from .sensor import SensorConfig, sensor_features

__all__ = [
    "error_vector",
    "ReferenceAccelEstimator",
    "policy_input",
    "policy_input_dim",
]


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


def policy_input_dim(sensor: SensorConfig, pos_dim: int = 3) -> int:
    """Width of :func:`policy_input`: ``3·pos_dim`` for ``err``+``u_ff`` plus the scan."""
    return 3 * int(pos_dim) + int(sensor.input_dim)


def policy_input(
    state: np.ndarray,
    ref,
    scene,
    estimator: ReferenceAccelEstimator,
    task=None,
    *,
    sensor: Optional[SensorConfig] = None,
    rng: Optional[np.random.Generator] = None,
    pos_dim: int = 3,
) -> np.ndarray:
    """Assemble the sensor-conditioned policy input for one frame."""
    sensor = sensor or SensorConfig()
    d = int(pos_dim)
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    err = error_vector(state, ref, d)
    u_ff = estimator.update(state, ref)
    scan_feats = sensor_features(state[:2], getattr(scene, "obstacles", ()), sensor, rng)
    return np.concatenate([err, u_ff, scan_feats]).astype(np.float64)
