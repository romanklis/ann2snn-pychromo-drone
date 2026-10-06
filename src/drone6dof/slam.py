"""SLAM-lite occupancy mapping.

Localization is provided by the existing UKF (the estimate); mapping is built
online from the horizontal LiDAR scan.  There is **no loop closure** — this is a
deliberately light mapper that turns the privileged scene into a map the planner
consumes, so the pipeline no longer reads ground truth online.

The grid stores log-odds; ``occupancy()`` thresholds it to
``0 = unknown, 1 = free, 2 = occupied`` for compact transport / plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .planner import plan_path_grid

__all__ = ["SlamConfig", "OccupancyMap", "truth_occupancy", "truth_shell"]

_EPS = 1e-9


@dataclass
class SlamConfig:
    """Occupancy grid + inverse-sensor-model parameters."""

    res: float = 0.1
    bounds: Tuple[float, float, float, float] = (-3.5, 3.5, -3.5, 3.5)
    log_odds_free: float = -0.4
    log_odds_occ: float = 0.85
    log_odds_clamp: float = 4.0
    occ_thresh: float = 0.6
    free_thresh: float = 0.4
    inflate: float = 0.30
    refresh: int = 40
    path_check_stride: int = 2
    viz_stride: int = 20
    viz_shape: int = 32

    def to_dict(self) -> dict:
        return {
            "res": float(self.res), "bounds": [float(v) for v in self.bounds],
            "log_odds_free": float(self.log_odds_free),
            "log_odds_occ": float(self.log_odds_occ),
            "log_odds_clamp": float(self.log_odds_clamp),
            "occ_thresh": float(self.occ_thresh),
            "free_thresh": float(self.free_thresh),
            "inflate": float(self.inflate),
            "refresh": int(self.refresh),
            "path_check_stride": int(self.path_check_stride),
            "viz_stride": int(self.viz_stride), "viz_shape": int(self.viz_shape),
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "SlamConfig":
        if not data:
            return cls()
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class OccupancyMap:
    """Log-odds occupancy grid with an inverse sensor model for a 2-D scan."""

    def __init__(self, config: Optional[SlamConfig] = None) -> None:
        self.cfg = config or SlamConfig()
        x0, x1, y0, y1 = self.cfg.bounds
        self.xs = np.arange(x0, x1 + 1e-9, self.cfg.res)
        self.ys = np.arange(y0, y1 + 1e-9, self.cfg.res)
        self.log_odds = np.zeros((len(self.xs), len(self.ys)), dtype=np.float64)
        # bookkeeping for the report
        self.replans = 0
        self.goal_replans = 0
        self.frontier_replans = 0
        self.replan_steps: list = []
        self.updates = 0
        self.r_max = 3.0
        self.path_found_step: Optional[int] = None

    # -- mapping ------------------------------------------------------------ #
    def _cell(self, xy) -> Optional[Tuple[int, int]]:
        i = int(round((float(xy[0]) - self.xs[0]) / self.cfg.res))
        j = int(round((float(xy[1]) - self.ys[0]) / self.cfg.res))
        if 0 <= i < len(self.xs) and 0 <= j < len(self.ys):
            return i, j
        return None

    def update(self, xy, ranges, angles, r_max: float) -> None:
        """Fuse one scan taken at pose ``xy`` (world-fixed beams)."""
        xy = np.asarray(xy, dtype=np.float64).reshape(2)
        ranges = np.asarray(ranges, dtype=np.float64).reshape(-1)
        angles = np.asarray(angles, dtype=np.float64).reshape(-1)
        self.r_max = float(r_max)
        step = self.cfg.res * 0.5
        lo_free = float(self.cfg.log_odds_free)
        lo_occ = float(self.cfg.log_odds_occ)
        for r, a in zip(ranges, angles):
            if not np.isfinite(r) or r <= 0.0:
                continue
            hit = r < (float(r_max) - 1e-3)
            d = 0.0
            while d <= r + _EPS:
                cell = self._cell((xy[0] + d * np.cos(a), xy[1] + d * np.sin(a)))
                if cell is not None:
                    if hit and d >= r - step:
                        self.log_odds[cell] += lo_occ
                    else:
                        self.log_odds[cell] += lo_free
                d += step
        np.clip(self.log_odds, -self.cfg.log_odds_clamp,
                self.cfg.log_odds_clamp, out=self.log_odds)
        self.updates += 1

    # -- queries ------------------------------------------------------------ #
    def occupancy(self) -> np.ndarray:
        """Thresholded grid: ``0`` unknown, ``1`` free, ``2`` occupied (uint8)."""
        p = _sigmoid(self.log_odds)
        out = np.zeros(p.shape, dtype=np.uint8)
        out[p <= self.cfg.free_thresh] = 1
        out[p >= self.cfg.occ_thresh] = 2
        return out

    def blocked(self, inflate: Optional[float] = None,
                conservative: bool = False) -> np.ndarray:
        """Boolean grid of occupied cells dilated by ``inflate`` (metres).

        ``conservative=True`` also blocks **unknown** cells, so the planner never
        routes through space it has not seen (the frontier fallback drives
        exploration).  Default is optimistic / unknown-passable.
        """
        occ = self.occupancy()
        blocked = occ == 2
        if conservative:
            blocked = blocked | (occ == 0)
        rad = 0 if inflate is None else int(round(float(inflate) / self.cfg.res))
        for _ in range(max(0, rad)):
            grown = blocked.copy()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    grown |= np.roll(np.roll(blocked, di, 0), dj, 1)
            blocked = grown
        return blocked

    def explored_frac(self) -> float:
        return float(np.mean(np.abs(self.log_odds) > 1e-6))

    def occupied_cells(self) -> int:
        return int(np.sum(self.occupancy() == 2))

    def entropy_bits(self) -> float:
        p = _sigmoid(self.log_odds)
        known = np.abs(self.log_odds) > 1e-6
        if not np.any(known):
            return 0.0
        q = p[known]
        h = -q * np.log2(q + _EPS) - (1 - q) * np.log2(1 - q + _EPS)
        return float(np.mean(h))

    def dense(self, shape: Optional[int] = None) -> np.ndarray:
        """Downsample the occupancy grid to ``shape``x``shape`` (occupied wins)."""
        occ = self.occupancy()
        n = int(shape or self.cfg.viz_shape)
        ni, nj = occ.shape
        ii = np.linspace(0, ni, n + 1).astype(int)
        jj = np.linspace(0, nj, n + 1).astype(int)
        out = np.zeros((n, n), dtype=np.uint8)
        for a in range(n):
            for b in range(n):
                block = occ[ii[a]:ii[a + 1], jj[b]:jj[b + 1]]
                if block.size == 0:
                    continue
                out[a, b] = 2 if np.any(block == 2) else (1 if np.any(block == 1) else 0)
        return out

    def iou(self, truth: np.ndarray) -> float:
        """IoU of the discovered occupied set against a truth occupancy grid.

        The mapper's occupied cells are dilated by one cell first, to tolerate the
        half-cell registration error of a surface-observing scan.
        """
        mine = self.occupancy() == 2
        grown = mine.copy()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                grown |= np.roll(np.roll(mine, di, 0), dj, 1)
        theirs = np.asarray(truth) == 2
        union = int(np.sum(grown | theirs))
        if union == 0:
            return 1.0
        return float(np.sum(grown & theirs) / union)

    def surface_recall(self, shell: np.ndarray) -> float:
        """Fraction of the (visible) truth shell the map has discovered.

        This is the fair "map discovery" score: recall against the obstacle
        surface cells the sensor could actually observe, not the whole structure.
        """
        mine = self.occupancy() == 2
        grown = mine.copy()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                grown |= np.roll(np.roll(mine, di, 0), dj, 1)
        target = np.asarray(shell) == 2
        n = int(np.sum(target))
        if n == 0:
            return 1.0
        return float(np.sum(grown & target) / n)

    # -- planning ----------------------------------------------------------- #
    def plan(self, start_xy, goal_xy, inflate: Optional[float] = None,
             conservative: bool = True):
        blocked = self.blocked(inflate if inflate is not None else self.cfg.inflate,
                               conservative=conservative)
        way = plan_path_grid(blocked, self.xs, self.ys, start_xy, goal_xy)
        if way is not None:
            self.replans += 1
        return way

    def frontier(self) -> Optional[np.ndarray]:
        """Nearest free cell adjacent to unknown, as ``(x, y)`` (or ``None``)."""
        occ = self.occupancy()
        free = occ == 1
        unknown = occ == 0
        adj = np.zeros_like(free)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                adj |= np.roll(np.roll(unknown, di, 0), dj, 1)
        cand = np.argwhere(free & adj)
        if cand.shape[0] == 0:
            return None
        i, j = cand[len(cand) // 2]
        return np.array([float(self.xs[i]), float(self.ys[j])])

    def nearest_frontier(self, goal_xy) -> Optional[np.ndarray]:
        occ = self.occupancy()
        free = occ == 1
        unknown = occ == 0
        adj = np.zeros_like(free)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                adj |= np.roll(np.roll(unknown, di, 0), dj, 1)
        cand = np.argwhere(free & adj)
        if cand.shape[0] == 0:
            return None
        pts = np.column_stack([self.xs[cand[:, 0]], self.ys[cand[:, 1]]])
        k = int(np.argmin(np.sum((pts - np.asarray(goal_xy).reshape(2)) ** 2, axis=1)))
        return pts[k]


def truth_occupancy(scene, map_: OccupancyMap) -> np.ndarray:
    """Ground-truth occupancy on the map grid (for the IoU metric only)."""
    X, Y = np.meshgrid(map_.xs, map_.ys, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    obs = getattr(scene, "obstacles", ())
    if obs:
        sd = np.min(np.stack([o.signed_distance_batch(pts) for o in obs]), axis=0)
    else:
        sd = np.full(pts.shape[0], np.inf)
    sd = sd.reshape(X.shape)
    out = np.ones(X.shape, dtype=np.uint8)          # free
    out[sd < 0.0] = 2                               # inside an obstacle
    return out


def truth_shell(scene, map_: OccupancyMap) -> np.ndarray:
    """Truth obstacle **shell** cells (inside, within ~1.5 cells of the surface).

    A horizontal LiDAR observes surfaces, not obstacle interiors, so the shell is
    the fair comparison target for the discovered occupied set.
    """
    X, Y = np.meshgrid(map_.xs, map_.ys, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    obs = getattr(scene, "obstacles", ())
    if obs:
        sd = np.min(np.stack([o.signed_distance_batch(pts) for o in obs]), axis=0)
    else:
        sd = np.full(pts.shape[0], np.inf)
    sd = sd.reshape(X.shape)
    out = np.zeros(X.shape, dtype=np.uint8)
    out[(sd <= 0.0) & (sd >= -1.5 * map_.cfg.res)] = 2
    return out
