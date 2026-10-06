"""Grid A* global planner over the 2.5-D obstacle cross-section.

The route is global and analytic: obstacles are inflated by a drone radius on a
regular grid, A* finds an 8-connected path, and a line-of-sight string-pull
simplifies it.  The planner consumes the privileged scene (a map), consistent
with the existing privileged teacher; the learned LiDAR field only corrects
locally on top of the resulting guidance (see :mod:`drone6dof.guidance`).
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .geometry import clearance

__all__ = ["PlannerConfig", "plan_path", "plan_path_grid", "path_clearance"]

_SQRT2 = float(np.sqrt(2.0))


@dataclass(frozen=True)
class PlannerConfig:
    """Occupancy resolution, obstacle inflation and the planning window."""

    res: float = 0.1
    inflate: float = 0.30
    inflate_goal: float = 0.10
    x_min: float = -3.5
    x_max: float = 3.5
    y_min: float = -3.5
    y_max: float = 3.5

    def to_dict(self) -> dict:
        return {
            "res": float(self.res), "inflate": float(self.inflate),
            "inflate_goal": float(self.inflate_goal),
            "x": [float(self.x_min), float(self.x_max)],
            "y": [float(self.y_min), float(self.y_max)],
        }


def _axes(cfg: PlannerConfig) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.arange(cfg.x_min, cfg.x_max + 1e-9, cfg.res)
    ys = np.arange(cfg.y_min, cfg.y_max + 1e-9, cfg.res)
    return xs, ys


def _cell(point_xy, xs, ys) -> Tuple[int, int]:
    i = int(round((float(point_xy[0]) - xs[0]) / (xs[1] - xs[0])))
    j = int(round((float(point_xy[1]) - ys[0]) / (ys[1] - ys[0])))
    return i, j


def _in_bounds(cell, shape) -> bool:
    return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]


def _blocked_grid(scene, xs, ys, inflate: float) -> np.ndarray:
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    if getattr(scene, "obstacles", ()):
        sd = np.min(np.stack([o.signed_distance_batch(pts) for o in scene.obstacles]), axis=0)
    else:
        sd = np.full(pts.shape[0], np.inf)
    return (sd < inflate).reshape(X.shape)


def _nearest_free(blocked: np.ndarray, cell: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """BFS outward from ``cell`` to the closest unblocked grid cell."""
    if not _in_bounds(cell, blocked.shape):
        return None
    if not blocked[cell]:
        return cell
    seen = {cell}
    q = deque([cell])
    while q:
        i, j = q.popleft()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                n = (i + di, j + dj)
                if n in seen or not _in_bounds(n, blocked.shape):
                    continue
                if not blocked[n]:
                    return n
                seen.add(n)
                q.append(n)
    return None


def _astar(blocked: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]):
    if blocked[start] or blocked[goal]:
        return None
    nx, ny = blocked.shape
    neighbours = [(-1, -1, _SQRT2), (-1, 0, 1.0), (-1, 1, _SQRT2),
                  (0, -1, 1.0), (0, 1, 1.0),
                  (1, -1, _SQRT2), (1, 0, 1.0), (1, 1, _SQRT2)]

    def h(c):
        di = abs(c[0] - goal[0]); dj = abs(c[1] - goal[1])
        return (di + dj) + (_SQRT2 - 2.0) * min(di, dj)

    open_heap = [(h(start), 0.0, start)]
    g = {start: 0.0}
    came = {}
    while open_heap:
        _, gc, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if gc > g.get(cur, float("inf")) + 1e-12:
            continue
        for di, dj, cost in neighbours:
            n = (cur[0] + di, cur[1] + dj)
            if not _in_bounds(n, blocked.shape) or blocked[n]:
                continue
            ng = gc + cost
            if ng + 1e-12 < g.get(n, float("inf")):
                g[n] = ng
                came[n] = cur
                heapq.heappush(open_heap, (ng + h(n), ng, n))
    return None


def _line_of_sight(a, b, scene, margin: float = 0.0) -> bool:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    length = float(np.linalg.norm(b - a))
    n = max(2, int(length / 0.05) + 1)
    for t in np.linspace(0.0, 1.0, n):
        p = a + t * (b - a)
        if clearance(getattr(scene, "obstacles", ()), p) < margin:
            return False
    return True


def _string_pull(points: List[Tuple[float, float]], scene, margin: float = 0.0):
    if len(points) <= 2:
        return list(points)
    out = [points[0]]
    i = 0
    last = len(points) - 1
    while i < last:
        j = last
        while j > i + 1 and not _line_of_sight(points[i], points[j], scene, margin):
            j -= 1
        out.append(points[j])
        i = j
    return out


def plan_path(
    scene,
    start_xy: Sequence[float],
    goal_xy: Sequence[float],
    config: Optional[PlannerConfig] = None,
) -> Optional[List[Tuple[float, float]]]:
    """Plan a collision-free ``[start, ..., goal]`` polyline, or ``None``."""
    cfg = config or PlannerConfig()
    xs, ys = _axes(cfg)
    blocked = _blocked_grid(scene, xs, ys, cfg.inflate)
    start = _cell(start_xy, xs, ys)
    goal = _cell(goal_xy, xs, ys)
    if not _in_bounds(start, blocked.shape) or not _in_bounds(goal, blocked.shape):
        return None
    start_free = _nearest_free(blocked, start)
    if start_free is None:
        return None
    if blocked[goal]:
        # goal nearer than `inflate` to a surface: plan to a reduced-inflation
        # free cell; the tracker's final approach + goal-capture fade close it.
        blocked_goal = _blocked_grid(scene, xs, ys, cfg.inflate_goal)
        goal_free = _nearest_free(blocked_goal, goal)
    else:
        goal_free = goal
    if goal_free is None:
        return None
    cells = _astar(blocked, start_free, goal_free)
    if cells is None:
        return None

    points = [(float(xs[i]), float(ys[j])) for i, j in cells]
    if goal_free != goal:
        points.append((float(goal_xy[0]), float(goal_xy[1])))
    points[0] = (float(start_xy[0]), float(start_xy[1]))
    points[-1] = (float(goal_xy[0]), float(goal_xy[1]))
    return _string_pull(points, scene, margin=0.0)


def _los_grid(blocked: np.ndarray, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    """True if the straight cell-to-cell segment crosses no blocked cell."""
    (i0, j0), (i1, j1) = a, b
    steps = max(abs(i1 - i0), abs(j1 - j0))
    if steps == 0:
        return not blocked[a]
    for s in range(steps + 1):
        t = s / steps
        i = int(round(i0 + t * (i1 - i0)))
        j = int(round(j0 + t * (j1 - j0)))
        if blocked[i, j]:
            return False
    return True


def _string_pull_grid(points, blocked: np.ndarray, xs, ys):
    if len(points) <= 2:
        return list(points)
    out = [points[0]]
    i = 0
    last = len(points) - 1
    while i < last:
        j = last
        while j > i + 1 and not _los_grid(blocked, _cell(points[i], xs, ys),
                                         _cell(points[j], xs, ys)):
            j -= 1
        out.append(points[j])
        i = j
    return out


def plan_path_grid(
    blocked: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    start_xy: Sequence[float],
    goal_xy: Sequence[float],
) -> Optional[List[Tuple[float, float]]]:
    """A* over a prebuilt occupancy grid (e.g. a live SLAM map), or ``None``.

    Unknown cells are expected to be encoded as *free* by the caller (optimistic
    planning); only blocked cells are treated as obstacles.
    """
    start = _cell(start_xy, xs, ys)
    goal = _cell(goal_xy, xs, ys)
    if not _in_bounds(start, blocked.shape) or not _in_bounds(goal, blocked.shape):
        return None
    start_free = _nearest_free(blocked, start)
    goal_free = _nearest_free(blocked, goal)
    if start_free is None or goal_free is None:
        return None
    cells = _astar(blocked, start_free, goal_free)
    if cells is None:
        return None
    points = [(float(xs[i]), float(ys[j])) for i, j in cells]
    points[0] = (float(start_xy[0]), float(start_xy[1]))
    points[-1] = (float(goal_xy[0]), float(goal_xy[1]))
    return _string_pull_grid(points, blocked, xs, ys)


def path_clearance(scene, path) -> float:
    """Minimum true clearance along a polyline (``inf`` with no obstacles)."""
    pts = np.asarray(path, dtype=np.float64)
    if len(pts) == 0:
        return float("inf")
    if len(pts) == 1:
        return clearance(getattr(scene, "obstacles", ()), pts[0])
    best = float("inf")
    for a, b in zip(pts[:-1], pts[1:]):
        length = float(np.linalg.norm(b - a))
        n = max(2, int(length / 0.05) + 1)
        for t in np.linspace(0.0, 1.0, n):
            best = min(best, clearance(getattr(scene, "obstacles", ()), a + t * (b - a)))
    return float(best)
