"""2.5D obstacle geometry: oriented boxes and cylinders.

Shapes live in the horizontal plane with a stored ``height``/``z0`` (so a future 3D
mode can use the vertical extent; the current teacher uses the 2D cross-section).
Each shape exposes the three primitives the DS teacher and the LiDAR sensor need:

* ``signed_distance(point_xy)`` — negative inside, positive outside;
* ``closest_point_normal(point_xy)`` — nearest boundary point and the outward unit
  normal (face- or corner-aware for boxes);
* ``ray_intersect(origin_xy, direction_xy)`` — entry distance for range sensing.

The reference distances the modulation uses are ``sdf + core_offset`` (a
cylinder's radius, 0 for a box), so the legacy single-pillar behaviour is
reproduced exactly: ``core_offset=radius``, ``dead_radius``/``influence_radius``
measured from the cylinder axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

__all__ = ["BoxObstacle", "Cylinder", "ray_cast", "clearance"]

_EPS = 1e-9


def _rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


@dataclass(frozen=True)
class BoxObstacle:
    """Axis- or angle-oriented vertical box (rectangle cross-section)."""

    center: Tuple[float, float] = (0.0, 0.0)
    half: Tuple[float, float] = (0.3, 0.3)
    angle: float = 0.0
    height: float = 2.8
    z0: float = 0.0
    core_offset: float = 0.0
    dead_radius: float = 0.5
    influence_radius: float = 1.2
    kind: str = field(default="box", init=False)

    @property
    def center_np(self) -> np.ndarray:
        return np.asarray(self.center, dtype=np.float64)

    @property
    def half_np(self) -> np.ndarray:
        return np.asarray(self.half, dtype=np.float64)

    def _to_local(self, p: np.ndarray) -> np.ndarray:
        return _rotation(-self.angle) @ (np.asarray(p, dtype=np.float64)[:2] - self.center_np)

    def _to_world(self, p_local: np.ndarray) -> np.ndarray:
        return self.center_np + _rotation(self.angle) @ np.asarray(p_local, dtype=np.float64)

    def signed_distance(self, point) -> float:
        loc = np.abs(self._to_local(np.asarray(point, dtype=np.float64)))
        q = loc - self.half_np
        outside = np.hypot(max(q[0], 0.0), max(q[1], 0.0))
        inside = min(max(q[0], q[1]), 0.0)
        return float(outside + inside)

    def signed_distance_batch(self, points) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)[:, :2]
        loc = (p - self.center_np) @ _rotation(-self.angle).T
        q = np.abs(loc) - self.half_np
        outside = np.hypot(np.maximum(q[:, 0], 0.0), np.maximum(q[:, 1], 0.0))
        inside = np.minimum(np.maximum(q[:, 0], q[:, 1]), 0.0)
        return outside + inside

    def closest_point_normal(self, point):
        """Return ``(closest_point_world, outward_normal, signed_distance)``."""
        p = np.asarray(point, dtype=np.float64)[:2]
        loc = self._to_local(p)
        q = np.abs(loc) - self.half_np
        if q[0] <= 0.0 and q[1] <= 0.0:
            # inside: push out through the nearest face
            if q[0] > q[1]:
                sx = 1.0 if loc[0] >= 0.0 else -1.0
                normal_local = np.array([sx, 0.0])
                cp_local = np.array([sx * self.half_np[0], loc[1]])
            else:
                sy = 1.0 if loc[1] >= 0.0 else -1.0
                normal_local = np.array([0.0, sy])
                cp_local = np.array([loc[0], sy * self.half_np[1]])
            normal_local = _rotation(self.angle) @ normal_local
            return self._to_world(cp_local), normal_local, self.signed_distance(p)
        cp_local = np.clip(loc, -self.half_np, self.half_np)
        cp_world = self._to_world(cp_local)
        d = p - cp_world
        dist = float(np.linalg.norm(d))
        if dist < _EPS:
            normal = _rotation(self.angle) @ np.array([1.0, 0.0])
        else:
            normal = d / dist
        return cp_world, normal, dist

    def ray_intersect(self, origin, direction, tmax: float = np.inf) -> float:
        o = self._to_local(np.asarray(origin, dtype=np.float64))
        d = _rotation(-self.angle) @ np.asarray(direction, dtype=np.float64)[:2]
        t_min, t_max = 0.0, float(tmax)
        for i in range(2):
            if abs(d[i]) < 1e-12:
                if abs(o[i]) > self.half_np[i]:
                    return float("inf")
            else:
                t1 = (-self.half_np[i] - o[i]) / d[i]
                t2 = (self.half_np[i] - o[i]) / d[i]
                if t1 > t2:
                    t1, t2 = t2, t1
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                if t_min > t_max:
                    return float("inf")
        return t_min

    def to_dict(self) -> dict:
        return {
            "kind": "box",
            "center": [float(v) for v in self.center],
            "half": [float(v) for v in self.half],
            "angle": float(self.angle),
            "height": float(self.height),
            "z0": float(self.z0),
        }


@dataclass(frozen=True)
class Cylinder:
    """Vertical cylinder (the legacy pillar shape)."""

    center: Tuple[float, float] = (0.0, 0.0)
    radius: float = 0.35
    height: float = 2.8
    z0: float = 0.0
    core_offset: float = -1.0  # sentinel; set to radius in __post_init__
    dead_radius: float = 0.62
    influence_radius: float = 1.05
    kind: str = field(default="cylinder", init=False)

    def __post_init__(self) -> None:
        if self.core_offset < 0.0:
            object.__setattr__(self, "core_offset", float(self.radius))

    @property
    def center_np(self) -> np.ndarray:
        return np.asarray(self.center, dtype=np.float64)

    def signed_distance(self, point) -> float:
        p = np.asarray(point, dtype=np.float64)[:2]
        return float(np.linalg.norm(p - self.center_np) - self.radius)

    def signed_distance_batch(self, points) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)[:, :2]
        return np.hypot(p[:, 0] - self.center_np[0], p[:, 1] - self.center_np[1]) - self.radius

    def closest_point_normal(self, point):
        p = np.asarray(point, dtype=np.float64)[:2]
        d = p - self.center_np
        dist = float(np.linalg.norm(d))
        if dist < _EPS:
            normal = np.array([1.0, 0.0])
            cp = self.center_np + self.radius * normal
        else:
            normal = d / dist
            cp = self.center_np + self.radius * normal
        return cp, normal, float(dist - self.radius)

    def ray_intersect(self, origin, direction, tmax: float = np.inf) -> float:
        o = np.asarray(origin, dtype=np.float64)[:2] - self.center_np
        d = np.asarray(direction, dtype=np.float64)[:2]
        a = float(d @ d)
        b = 2.0 * float(o @ d)
        c = float(o @ o) - self.radius ** 2
        disc = b * b - 4.0 * a * c
        if a < 1e-12 or disc < 0.0:
            return float("inf")
        sq = np.sqrt(disc)
        t0 = (-b - sq) / (2.0 * a)
        t1 = (-b + sq) / (2.0 * a)
        t = t0 if t0 >= 0.0 else t1
        if t < 0.0 or t > tmax:
            return float("inf")
        return float(t)

    def to_dict(self) -> dict:
        return {
            "kind": "cylinder",
            "center": [float(v) for v in self.center],
            "radius": float(self.radius),
            "height": float(self.height),
            "z0": float(self.z0),
        }


def clearance(obstacles: Sequence, point) -> float:
    """Minimum signed distance to any obstacle (``inf`` with none)."""
    if not obstacles:
        return float("inf")
    return float(min(o.signed_distance(point) for o in obstacles))


def ray_cast(
    obstacles: Sequence,
    origin,
    angles: np.ndarray,
    r_max: float,
) -> np.ndarray:
    """Distance to the nearest obstacle along each angle (``r_max`` on a miss)."""
    origin = np.asarray(origin, dtype=np.float64)[:2]
    ranges = np.full(len(angles), float(r_max), dtype=np.float64)
    if not obstacles:
        return ranges
    for i, angle in enumerate(np.asarray(angles, dtype=np.float64)):
        direction = np.array([np.cos(angle), np.sin(angle)])
        best = float(r_max)
        for obstacle in obstacles:
            t = obstacle.ray_intersect(origin, direction, tmax=best)
            if t < best:
                best = t
        ranges[i] = best
    return ranges
