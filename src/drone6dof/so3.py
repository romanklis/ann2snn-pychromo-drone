"""Rotation utilities for the 6-DoF plant (numpy port of ``plants/quad6dof.py``).

Ported from the ANN2SNN ``drone-example`` branch (MIT); see ``NOTICE``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["skew", "rodrigues", "reorthonormalize", "rpy_from_R"]


def skew(w: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix ``[w]_x`` such that ``[w]_x v = w x v``."""
    return np.array(
        [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]],
        dtype=np.float64,
    )


def rodrigues(w_dt: np.ndarray) -> np.ndarray:
    """``exp(skew(w·dt))`` — exact for a constant body rate over the substep."""
    th = float(np.linalg.norm(w_dt))
    if th < 1e-9:
        return np.eye(3) + skew(w_dt)
    K = skew(w_dt / th)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def reorthonormalize(R: np.ndarray) -> np.ndarray:
    """SVD projection onto SO(3) to remove long-horizon integration drift."""
    U, _, Vt = np.linalg.svd(R)
    out = U @ Vt
    if np.linalg.det(out) < 0.0:
        U = U.copy()
        U[:, -1] *= -1.0
        out = U @ Vt
    return out


def rpy_from_R(R: np.ndarray) -> Tuple[float, float, float]:
    """ZYX Euler angles (roll, pitch, yaw) in radians."""
    pitch = float(np.arcsin(-np.clip(R[2, 0], -1.0, 1.0)))
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw
