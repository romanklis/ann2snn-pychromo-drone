"""Compact, trajectory-anchored obstacle-field representation + DS modulation.

The SNN does **not** output a control law.  It outputs ``K`` non-negative
coefficients ``a`` of a fixed local potential basis:

    U_obs(x) = Σ_k a_k φ_k(x − c_k)          (normalised Gaussian RBFs)

The basis centres ``c_k`` are anchored to the **planned nominal trajectory**
(a short preview of the stable nominal DS), so the field is trajectory-relative.
The obstacle vector field is the analytic gradient ``F_obs = −∇U_obs``
(conservative — never learned separately), and it drives a Billard-style
modulation of the nominal DS.  The teacher potential (privileged geometry) is
used only to produce training targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "FieldConfig",
    "PotentialBasis",
    "nominal_ds",
    "nominal_preview",
    "teacher_potential",
    "teacher_grad",
    "teacher_coeffs",
    "fit_field_coeffs",
    "gate",
    "goal_fade",
    "modulate_ds",
    "trajectory_cost",
    "eval_grid",
]

_EPS = 1e-9

#: nominal DS gains matching the example teacher (``DS_TEACHER_KWARGS`` + defaults)
DEFAULT_GAINS: Dict[str, float] = {
    "radial": 0.6,   # ds_radial_gain
    "curl": 1.35,    # ds_gain
    "climb": 0.45,   # ds_climb_gain
    "cap": 1.4,      # speed_cap
}


@dataclass(frozen=True)
class FieldConfig:
    """Trajectory-anchored RBF basis + modulation parameters."""

    h: int = 8                # preview points
    m: int = 3                # centres per preview point (on + both lateral)
    preview_dt: float = 0.05
    lateral: float = 0.4
    sigma: float = 0.5
    r_influence: float = 1.5  # teacher barrier influence [m]
    field_limit: float = 2.0
    f_ref: float = 1.0        # |F_obs| that gives full modulation weight
    lam_t: float = 0.8        # tangential stretch at full weight
    gains: Tuple[Tuple[str, float], ...] = tuple(sorted(DEFAULT_GAINS.items()))

    @property
    def k(self) -> int:
        return int(self.h) * int(self.m)

    def gains_dict(self) -> Dict[str, float]:
        return dict(self.gains)

    def to_dict(self) -> dict:
        return {
            "h": int(self.h), "m": int(self.m), "k": self.k,
            "preview_dt": float(self.preview_dt), "lateral": float(self.lateral),
            "sigma": float(self.sigma), "r_influence": float(self.r_influence),
            "field_limit": float(self.field_limit), "f_ref": float(self.f_ref),
            "lam_t": float(self.lam_t), "gains": {k: float(v) for k, v in self.gains},
        }


def nominal_ds(e: np.ndarray, gains: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Stable nominal DS velocity field ``f_DS(e)`` (radial + curl + climb)."""
    g = dict(DEFAULT_GAINS)
    if gains:
        g.update(gains)
    vx = -g["radial"] * e[0] - g["curl"] * e[1]
    vy = -g["radial"] * e[1] + g["curl"] * e[0]
    vz = -g["climb"] * e[2]
    v = np.array([vx, vy, vz], dtype=np.float64)
    speed = float(np.linalg.norm(v))
    return v * min(g["cap"] / (speed + _EPS), 1.0)


def nominal_preview(
    state: np.ndarray, goal, config: FieldConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Short nominal-DS preview: ``(H, 3)`` points and ``(H, 3)`` unit headings."""
    g = config.gains_dict()
    p = np.asarray(state, dtype=np.float64).reshape(-1)[:3].copy()
    goal = np.asarray(goal, dtype=np.float64).reshape(3)
    pts = [p.copy()]
    heads = []
    for _ in range(max(1, int(config.h)) - 1):
        v = nominal_ds(p - goal, g)
        h = v / (float(np.linalg.norm(v)) + _EPS)
        heads.append(h)
        p = p + v * config.preview_dt
        pts.append(p.copy())
    heads.append(heads[-1] if heads else np.array([1.0, 0.0, 0.0]))
    return np.asarray(pts), np.asarray(heads)


class PotentialBasis:
    """Trajectory-anchored normalised Gaussian RBF basis (analytic gradient)."""

    def __init__(self, config: Optional[FieldConfig] = None) -> None:
        self.cfg = config or FieldConfig()
        self.sigma = float(self.cfg.sigma)

    def centers(self, state: np.ndarray, goal) -> np.ndarray:
        pts, heads = nominal_preview(state, goal, self.cfg)
        centers = []
        for pt, h in zip(pts, heads):
            perp = np.array([-h[1], h[0], 0.0])
            centers.append(pt)
            centers.append(pt + self.cfg.lateral * perp)
            centers.append(pt - self.cfg.lateral * perp)
        return np.asarray(centers, dtype=np.float64)  # (K, 3)

    def _weights(self, point: np.ndarray, centers: np.ndarray):
        d = centers[:, :2] - np.asarray(point, dtype=np.float64)[:2]
        phi = np.exp(-np.sum(d * d, axis=1) / (2.0 * self.sigma ** 2))
        s = float(phi.sum())
        if s < 1e-12:
            return phi, s, d
        return phi, s, d

    def eval_U(self, coeffs: np.ndarray, point: np.ndarray, centers: np.ndarray) -> float:
        phi, s, _ = self._weights(point, centers)
        if s < 1e-12:
            return 0.0
        return float(np.asarray(coeffs, dtype=np.float64) @ (phi / s))

    def grad_U(self, coeffs: np.ndarray, point: np.ndarray, centers: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(coeffs, dtype=np.float64)
        phi, s, d = self._weights(point, centers)
        if s < 1e-12:
            return np.zeros(3)
        # d/dx of phi_k = -phi_k (x - c_k)/sigma^2 ; d = c_k - x so (x-c_k) = -d
        dphi = phi[:, None] * (d / self.sigma ** 2)   # (K,2), dphi_k/dx = -phi*(-d)/s2 = phi*d/s2
        A = coeffs @ phi
        dA = dphi.T @ coeffs                          # (2,)
        dS = dphi.sum(axis=0)                         # (2,)
        dU = dA / s - (A * dS) / (s * s)
        return np.array([dU[0], dU[1], 0.0])

    def eval_batch(self, coeffs: np.ndarray, points: np.ndarray, centers: np.ndarray) -> np.ndarray:
        return np.asarray([self.eval_U(coeffs, p, centers) for p in np.asarray(points)])

    def grad_matrix(self, point: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """``(3, K)`` matrix ``M`` with ``grad_U(a) = M @ a`` (linear in ``a``)."""
        phi, s, d = self._weights(point, centers)
        k = len(centers)
        if s < 1e-12:
            return np.zeros((3, k))
        dphi = phi[:, None] * (d / self.sigma ** 2)          # (K, 2) = d(phi_k)/dx
        dS = dphi.sum(axis=0)                                # (2,)
        m = dphi / s - phi[:, None] * dS[None, :] / (s * s)  # (K, 2) = grad(phi_k/s)
        out = np.zeros((3, k))
        out[:2, :] = m.T
        return out


def _barrier(d: float, r_influence: float) -> float:
    return 0.5 * max(0.0, r_influence - d) ** 2


def teacher_potential(scene, point, config: Optional[FieldConfig] = None) -> float:
    """Privileged positive barrier potential ``U*`` (superposition over obstacles)."""
    config = config or FieldConfig()
    p = np.asarray(point, dtype=np.float64).reshape(-1)
    total = 0.0
    for obstacle in getattr(scene, "obstacles", ()) or ():
        total += _barrier(float(obstacle.signed_distance(p)), config.r_influence)
    return float(total)


def teacher_grad(scene, point, config: Optional[FieldConfig] = None) -> np.ndarray:
    """``∇U*`` from privileged geometry (for checks/targets)."""
    config = config or FieldConfig()
    p = np.asarray(point, dtype=np.float64).reshape(-1)
    grad = np.zeros(3)
    for obstacle in getattr(scene, "obstacles", ()) or ():
        _, n_xy, sdf = obstacle.closest_point_normal(p[:2])
        d = float(sdf)
        if d < config.r_influence:
            # U = 0.5 (r-d)^2, dU/dx = -(r-d) * dd/dx = -(r-d) * n_out
            grad[:2] += -(config.r_influence - d) * np.asarray(n_xy, dtype=np.float64)
    return grad


def teacher_coeffs(scene, state, config: Optional[FieldConfig] = None) -> np.ndarray:
    """Nodal teacher-potential values at the trajectory-anchored centres (target)."""
    config = config or FieldConfig()
    goal = np.asarray(getattr(scene, "goal_np", np.zeros(3)), dtype=np.float64)
    centers = PotentialBasis(config).centers(state, goal)
    return np.asarray([teacher_potential(scene, c, config) for c in centers])


def fit_field_coeffs(
    scene, point, centers, config: Optional[FieldConfig] = None
) -> np.ndarray:
    """Non-negative coefficients whose gradient best matches ``∇U*`` at ``point``.

    Gradient matching (not value matching): the controller consumes ``−∇U``, so
    the target is the teacher gradient ``∇U*`` at the evaluation point.  The
    normal equations are under-determined (``K`` coefficients, 2 equations), so
    the min-norm least-squares solution is used and clipped to be non-negative.
    """
    config = config or FieldConfig()
    basis = PotentialBasis(config)
    m = basis.grad_matrix(np.asarray(point, dtype=np.float64), centers)
    target = teacher_grad(scene, point, config)
    if not np.any(target[:2]):
        return np.zeros(len(centers))
    a, *_ = np.linalg.lstsq(m[:2], target[:2], rcond=None)
    return np.clip(a, 0.0, config.field_limit)


def gate(min_range: float, config: Optional[FieldConfig] = None) -> float:
    """Distance gate: 0 in free space (``min_range ≥ r_influence``), 1 close in."""
    config = config or FieldConfig()
    return float(np.clip((config.r_influence - float(min_range))
                         / (config.r_influence + _EPS), 0.0, 1.0))


def goal_fade(dist_goal: float, snap: float = 0.30, hold: float = 0.80) -> float:
    """Goal-capture fade: 0 inside the goal tolerance, 1 beyond ``hold``."""
    return float(np.clip((float(dist_goal) - snap) / (hold - snap + _EPS), 0.0, 1.0))


def modulate_ds(v_nom: np.ndarray, f_obs: np.ndarray, config: Optional[FieldConfig] = None) -> np.ndarray:
    """Billard-style modulation from the learned field (``M→I`` when ``F→0``)."""
    config = config or FieldConfig()
    v_nom = np.asarray(v_nom, dtype=np.float64).reshape(3)
    f = np.asarray(f_obs, dtype=np.float64).reshape(3)
    mag = float(np.linalg.norm(f[:2]))
    if mag < 1e-9:
        return v_nom.copy()
    n = np.array([f[0] / mag, f[1] / mag, 0.0])   # outward (away from obstacle)
    t = np.array([-n[1], n[0], 0.0])
    if float(v_nom @ t) < 0.0:
        t = -t
    w = float(np.clip(mag / config.f_ref, 0.0, 1.0))
    vr = float(v_nom @ n)
    vt = float(v_nom @ t)
    return v_nom - w * vr * n + config.lam_t * w * vt * t


def trajectory_cost(basis: PotentialBasis, coeffs, path, centers) -> float:
    """``C = ∫ U(x(s)) ds`` along a candidate/local path."""
    pts = np.asarray(path, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    u = basis.eval_batch(coeffs, pts, centers)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return float(np.sum(0.5 * (u[:-1] + u[1:]) * seg))


def eval_grid(basis: PotentialBasis, coeffs, centers, center_xy, extent: float = 2.5, n: int = 41):
    """Return ``(X, Y, U, Fx, Fy)`` grids for visualisation."""
    xs = np.linspace(center_xy[0] - extent, center_xy[0] + extent, n)
    ys = np.linspace(center_xy[1] - extent, center_xy[1] + extent, n)
    X, Y = np.meshgrid(xs, ys)
    U = np.zeros_like(X)
    Fx = np.zeros_like(X)
    Fy = np.zeros_like(X)
    for i in range(n):
        for j in range(n):
            p = np.array([X[i, j], Y[i, j], 0.0])
            U[i, j] = basis.eval_U(coeffs, p, centers)
            g = basis.grad_U(coeffs, p, centers)
            Fx[i, j] = -g[0]
            Fy[i, j] = -g[1]
    return X, Y, U, Fx, Fy
