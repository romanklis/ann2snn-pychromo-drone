"""LiDAR-like range sensing and shape cues for the sensor-conditioned policy.

The policy never sees obstacle identities or extents: it sees a horizontal scan
(``k`` beams over 360 deg, clipped at ``r_max``) plus a few cues estimated from
the profile (nearest distance, bearing to the nearest return, slope and
curvature).  A realistic noise model adds range noise, max-range/missed returns
and dropout.  The scan is a horizontal slice of the 2.5D geometry, so the same
code becomes a 3D slice later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .geometry import ray_cast

__all__ = ["SensorConfig", "scan", "sensor_features", "SENSOR_CUE_DIM"]

_EPS = 1e-9

#: cues appended after the ``k`` range beams: min range, bearing (2), slope, curvature
SENSOR_CUE_DIM = 5


@dataclass(frozen=True)
class SensorConfig:
    k: int = 32
    r_max: float = 3.0
    range_sigma: float = 0.02
    dropout_p: float = 0.02
    max_range_miss_p: float = 0.01
    seed: int = 0

    def angles(self) -> np.ndarray:
        return np.linspace(0.0, 2.0 * np.pi, int(self.k), endpoint=False)

    @property
    def input_dim(self) -> int:
        return int(self.k) + SENSOR_CUE_DIM

    def to_dict(self) -> dict:
        return {
            "k": int(self.k),
            "r_max": float(self.r_max),
            "range_sigma": float(self.range_sigma),
            "dropout_p": float(self.dropout_p),
            "max_range_miss_p": float(self.max_range_miss_p),
        }


def scan(
    obstacles: Sequence,
    position,
    config: SensorConfig,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Noisy horizontal range scan (``k`` beams, ``r_max`` on a miss/dropout)."""
    angles = config.angles()
    ranges = ray_cast(obstacles, position, angles, config.r_max)
    ranges = np.clip(ranges, 0.0, config.r_max)
    if rng is not None:
        ranges = ranges + rng.normal(0.0, float(config.range_sigma), size=ranges.shape)
        if config.max_range_miss_p > 0.0:
            ranges[rng.random(ranges.shape) < float(config.max_range_miss_p)] = config.r_max
        if config.dropout_p > 0.0:
            ranges[rng.random(ranges.shape) < float(config.dropout_p)] = config.r_max
    return np.clip(ranges, 0.0, config.r_max)


def sensor_features(
    position,
    obstacles: Sequence,
    config: SensorConfig,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """``(k + 5,)`` policy features: normalized ranges + min/bearing/slope/curvature."""
    angles = config.angles()
    k = int(config.k)
    ranges = scan(obstacles, position, config, rng)
    r_norm = ranges / config.r_max

    idx = int(np.argmin(ranges))
    min_range = float(ranges[idx]) / config.r_max
    bearing = np.array([np.cos(angles[idx]), np.sin(angles[idx])], dtype=np.float64)

    delta = 2.0 * np.pi / max(k, 1)
    i0, i1 = (idx - 1) % k, (idx + 1) % k
    slope = float(ranges[i1] - ranges[i0]) / (2.0 * delta * config.r_max)
    curvature = abs(float(ranges[i0]) - 2.0 * float(ranges[idx]) + float(ranges[i1])) / config.r_max

    cues = np.array([min_range, bearing[0], bearing[1], slope, curvature], dtype=np.float64)
    return np.concatenate([r_norm, cues])
