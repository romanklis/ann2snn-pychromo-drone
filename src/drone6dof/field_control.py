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

from .connectome import ConnectomeANN, RateCodedConnectomeSNN, SparseRecurrence
from .config import GUIDANCE, PLANNER
from .field import (
    FieldConfig,
    PotentialBasis,
    goal_fade,
    gate,
    modulate_ds,
)
from .guidance import PathTracker
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
        self.guidance = dict(GUIDANCE)
        self.planner_config = PLANNER
        self._tracker: Optional[PathTracker] = None
        self._tracker_goal = None
        self._frame = 0

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
            self.net = RateCodedConnectomeSNN(
                bundle["w_in"], bundle["w_out"], rec,
                micro_steps=int(bundle.get("micro_steps", 10)),
                v_th=float(bundle.get("v_th", 1.0)),
                limit=float(self.cfg.field_limit),
                readout_gain=float(bundle.get("readout_gain", 1.0)),
            )

    def reset(self) -> None:
        self.net.reset()
        self.estimator.reset()
        self._rng = np.random.default_rng(self.sensor.seed)
        self._last = {}
        self._last_scan = None
        self._tracker = None
        self._tracker_goal = None
        self._frame = 0
        self._last_plan_frame = -10**9
        self._min_replan_gap = 5

    def _make_tracker(self, scene, goal, start, way=None) -> PathTracker:
        g = self.guidance
        return PathTracker(
            scene, start, goal, way=way,
            speed=float(g.get("speed", 1.4)), k_path=float(g.get("k_path", 1.0)),
            lookahead=float(g.get("lookahead", 0.5)), brake=float(g.get("brake", 0.8)),
            planner_config=self.planner_config,
        )

    @staticmethod
    def _path_blocked(map_, waypoints, stride) -> bool:
        """True if any sampled current waypoint is inside a **known** obstacle.

        Unknown space does not invalidate a path (the planner is conservative
        only when searching); only newly-observed obstacles do.
        """
        if not waypoints:
            return False
        blocked = map_.blocked(map_.cfg.inflate, conservative=False)
        xs, ys = map_.xs, map_.ys
        ni, nj = blocked.shape
        for x, y in list(waypoints)[::max(1, int(stride))]:
            i = int(round((float(x) - xs[0]) / map_.cfg.res))
            j = int(round((float(y) - ys[0]) / map_.cfg.res))
            if 0 <= i < ni and 0 <= j < nj and blocked[i, j]:
                return True
        return False

    def _ensure_tracker(self, scene, goal, start, map_=None) -> None:
        goal = np.asarray(goal, dtype=np.float64).reshape(3)
        if map_ is not None:
            refresh = max(1, int(getattr(map_.cfg, "refresh", 40)))
            stride = max(1, int(getattr(map_.cfg, "path_check_stride", 2)))
            gap = self._frame - self._last_plan_frame
            fresh = self._tracker is None or not np.allclose(self._tracker_goal, goal)
            due = gap >= refresh
            invalid = (self._tracker is not None and gap >= self._min_replan_gap
                       and self._frame % 5 == 0
                       and self._path_blocked(map_, self._tracker.waypoints, stride))
            if fresh or due or invalid:
                way = map_.plan(start[:2], goal[:2])
                kind = "goal"
                if way is None:
                    frontier = map_.nearest_frontier(goal[:2])
                    way = map_.plan(start[:2], frontier) if frontier is not None else None
                    kind = "frontier"
                if way is not None:
                    map_.replan_steps.append(int(self._frame))
                    if kind == "goal":
                        map_.goal_replans += 1
                        if map_.path_found_step is None:
                            map_.path_found_step = int(self._frame)
                    else:
                        map_.frontier_replans += 1
                    if self._tracker is None:
                        self._tracker = self._make_tracker(scene, goal, start, way=way)
                    else:
                        self._tracker.set_path(way, start=start)
                    self._tracker_goal = goal.copy()
                elif self._tracker is None:
                    # no map route yet and nothing to track: go straight (the
                    # learned field is the safety layer).  Never fall back to the
                    # privileged scene planner while a map is in use.
                    straight = [(float(start[0]), float(start[1])),
                                (float(goal[0]), float(goal[1]))]
                    self._tracker = self._make_tracker(scene, goal, start, way=straight)
                    self._tracker_goal = goal.copy()
                # throttle attempts regardless of success
                self._last_plan_frame = self._frame
            self._frame += 1
            return
        if self._tracker is not None and np.allclose(self._tracker_goal, goal):
            return
        self._tracker = self._make_tracker(scene, goal, start)
        self._tracker_goal = goal.copy()

    def act(self, state, ref) -> np.ndarray:
        t0 = time.perf_counter()
        scene = (getattr(ref, "meta", None) or {}).get("scene")
        if scene is None:
            raise ValueError("the field controller needs ref.meta['scene']")
        state = np.asarray(getattr(state, "state", state), dtype=np.float64).reshape(-1)
        goal = np.asarray(ref.pos, dtype=np.float64).reshape(3)
        map_ = (getattr(ref, "meta", None) or {}).get("map")
        self._ensure_tracker(scene, goal, state[:3], map_)
        features, ranges = policy_input_with_scan(
            state, ref, scene, self.estimator, sensor=self.sensor,
            rng=self._rng, pos_dim=3,
        )
        self._last_scan = ranges

        raw = self.net.forward_input(features)
        coeffs = np.clip(np.asarray(raw, dtype=np.float64), 0.0, self.cfg.field_limit)

        centers = self.basis.centers(state, goal)
        grad = self.basis.grad_U(coeffs, state[:3], centers)
        f_obs = -grad                                   # conservative F = -grad U
        v_nom = self._tracker.v_nom(state[:3])          # global A* path tracking

        min_range = float(np.min(ranges)) if ranges is not None and len(ranges) else np.inf
        weight = gate(min_range, self.cfg) * goal_fade(float(np.linalg.norm(state[:3] - goal)),
                                                       snap=float(self.guidance.get("snap", 0.30)),
                                                       hold=float(self.guidance.get("hold", 0.80)))
        v_des = modulate_ds(v_nom, weight * f_obs, self.cfg)
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
            "gate": float(weight),
            "coeff_norm": float(np.linalg.norm(coeffs)),
            "planned": 1.0 if (self._tracker is not None and self._tracker.planned) else 0.0,
            "waypoints": float(len(self._tracker.waypoints)) if self._tracker is not None else 0.0,
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
