"""Structured SNN/ANN field controller: LiDAR -> compact field -> DS modulation.

The network outputs ``K`` non-negative coefficients of the trajectory-anchored
potential basis (never acceleration).  The obstacle vector field is the analytic
gradient ``F = −∇U`` and drives a Billard-style modulation of the **nominal DS**;
the existing passive impedance then produces the acceleration demand reused by
the plant.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np

from .connectome import ConnectomeANN, LosslessConnectomeSNN, SparseRecurrence
from .field import (
    FieldConfig,
    PotentialBasis,
    modulate_ds,
    nominal_ds,
)
from .policy import ReferenceAccelEstimator, policy_input_with_scan
from .sensor import SensorConfig

__all__ = ["FieldDSController"]

_EPS = 1e-6


class FieldDSController:
    """``kind`` is ``field_ann`` or ``field_snn`` (both share the field head)."""

    def __init__(
        self,
        kind: str,
        bundle: dict,
        *,
        dt: float = 0.02,
        action_limit: float = 12.0,
        sensor: Optional[SensorConfig] = None,
        field_config: Optional[FieldConfig] = None,
        mass: float = 0.5,
        damping_along: float = 1.2,
        damping_across: float = 4.5,
    ) -> None:
        kind = str(kind).lower()
        if kind not in ("field_ann", "field_snn"):
            raise ValueError(f"kind must be 'field_ann' or 'field_snn', got {kind!r}")
        self.kind = kind
        self.name = kind
        self.dt = float(dt)
        self.action_limit = float(action_limit)
        self.mass = float(mass)
        self.damping_along = float(damping_along)
        self.damping_across = float(damping_across)

        cfg_dict = bundle.get("field", {}) or {}
        if field_config is None:
            field_config = FieldConfig(
                h=int(cfg_dict.get("h", 8)), m=int(cfg_dict.get("m", 3)),
                preview_dt=float(cfg_dict.get("preview_dt", 0.05)),
                lateral=float(cfg_dict.get("lateral", 0.4)),
                sigma=float(cfg_dict.get("sigma", 0.5)),
                r_influence=float(cfg_dict.get("r_influence", 1.5)),
                field_limit=float(cfg_dict.get("field_limit", 2.0)),
                f_ref=float(cfg_dict.get("f_ref", 1.0)),
                lam_t=float(cfg_dict.get("lam_t", 0.8)),
            )
        self.cfg = field_config
        self.gains = self.cfg.gains_dict()
        self.basis = PotentialBasis(self.cfg)

        self.sensor = sensor or SensorConfig()
        self.estimator = ReferenceAccelEstimator(dt=self.dt, plant_gain=1.0, pos_dim=3)
        self._rng = np.random.default_rng(self.sensor.seed)
        self._last_scan = None
        self._last: Dict[str, float] = {}

        edges = np.asarray(bundle["edges"], dtype=np.int64)
        polarity = np.asarray(bundle["polarity"], dtype=np.float64)
        signed = np.abs(np.asarray(bundle["w_mag"], dtype=np.float64)) * polarity[edges[1]]
        rec = SparseRecurrence(edges, signed, int(bundle["n_neurons"]))
        if kind == "field_ann":
            self.net = ConnectomeANN(
                bundle["w_in"], bundle["w_out"], rec,
                steps_per_frame=int(bundle.get("connectome_steps", 3)),
                limit=float(self.cfg.field_limit),
            )
        else:
            self.net = LosslessConnectomeSNN(
                bundle["w_in"], bundle["w_out"], rec,
                micro_steps=int(bundle.get("micro_steps", 10)),
                v_th=float(bundle.get("v_th", 1.0)),
                limit=float(self.cfg.field_limit),
            )

    def reset(self) -> None:
        self.net.reset()
        self.estimator.reset()
        self._rng = np.random.default_rng(self.sensor.seed)
        self._last = {}
        self._last_scan = None

    def act(self, state, ref) -> np.ndarray:
        t0 = time.perf_counter()
        scene = (getattr(ref, "meta", None) or {}).get("scene")
        if scene is None:
            raise ValueError("the field controller needs ref.meta['scene']")
        state = np.asarray(getattr(state, "state", state), dtype=np.float64).reshape(-1)
        features, ranges = policy_input_with_scan(
            state, ref, scene, self.estimator, sensor=self.sensor,
            rng=self._rng, pos_dim=3,
        )
        self._last_scan = ranges

        raw = self.net.forward_input(features)
        coeffs = np.clip(np.asarray(raw, dtype=np.float64), 0.0, self.cfg.field_limit)

        goal = np.asarray(ref.pos, dtype=np.float64)
        centers = self.basis.centers(state, goal)
        grad = self.basis.grad_U(coeffs, state[:3], centers)
        f_obs = -grad                                   # conservative F = -grad U
        v_nom = nominal_ds(state[:3] - goal, self.gains)
        v_des = modulate_ds(v_nom, f_obs, self.cfg)
        self._last_coeffs = coeffs.copy()
        self._last_centers = centers.copy()
        self._last_v_nom = v_nom.copy()
        self._last_v_des = v_des.copy()

        # passive impedance -> acceleration demand (reused low-level controller)
        edot = state[3:6] - np.asarray(ref.vel, dtype=np.float64)
        dv = edot - v_des
        vh = v_des / (float(np.linalg.norm(v_des)) + _EPS)
        along = float(dv @ vh)
        accel = -(self.damping_along * along * vh
                  + self.damping_across * (dv - along * vh))
        u = accel / self.mass

        mag = float(np.linalg.norm(f_obs[:2]))
        base = {
            "field_norm": mag,
            "mod_weight": float(np.clip(mag / self.cfg.f_ref, 0.0, 1.0)),
            "coeff_norm": float(np.linalg.norm(coeffs)),
            "latency_ms": (time.perf_counter() - t0) * 1e3,
        }
        if self.kind == "field_ann":
            base.update(self.net.last_telemetry)
        else:
            base.update(self.net.last_telemetry(self.dt))
        self._last = base

        return np.clip(u, -self.action_limit, self.action_limit)

    @property
    def last_telemetry(self) -> Dict[str, float]:
        return dict(self._last)

    @property
    def last_spikes(self):
        return getattr(self.net, "last_spikes", None)

    @property
    def last_scan(self):
        return self._last_scan

    @property
    def scan_angles(self):
        return self.sensor.angles()

    @property
    def last_coeffs(self):
        return getattr(self, "_last_coeffs", None)

    @property
    def last_centers(self):
        return getattr(self, "_last_centers", None)

    @property
    def last_ds(self):
        return {
            "v_nom": getattr(self, "_last_v_nom", None),
            "v_des": getattr(self, "_last_v_des", None),
        }
