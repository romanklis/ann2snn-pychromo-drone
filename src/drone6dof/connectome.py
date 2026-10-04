"""Connectome network: sparse recurrent ANN and its spiking (IF) transfer.

Numpy port of ``sim_engine/controllers/{connectome,snn}.py`` from the ANN2SNN
``drone-example`` branch (MIT); see ``NOTICE``.  The topology and dynamics match
upstream:

* topology: ``N`` neurons, fan-in ``K`` (``N·K`` edges), 20 % inhibitory scaled
  ×4 (Dale's law), seeded;
* ANN: ``h <- ReLU(W_in·x + W_rec·h)``, one command per ``steps_per_frame``
  recurrent evaluations;
* SNN: integrate-and-fire, ``micro_steps`` sub-steps per control frame,
  non-negative membrane, soft reset at threshold ``v_th``, command = mean
  output rate;
* both saturate the command to ``[-limit, +limit]``.

Runtime is numpy only.  The recurrence is stored sparsely (edge list) and applied
with ``np.bincount`` so a frame costs ``O(N·K)`` instead of ``O(N²)``.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .policy import ReferenceAccelEstimator, policy_input
from .sensor import SensorConfig
from .task import ObstacleGoalTask

__all__ = [
    "ConnectomeTopology",
    "SparseRecurrence",
    "ConnectomeANN",
    "LosslessConnectomeSNN",
    "ConnectomeController",
    "DEFAULTS",
]

DEFAULTS = {
    "n_neurons": 1000,
    "k": 40,
    "inhibitory_fraction": 0.20,
    "excitatory_weight": 1.0,
    "inhibitory_weight": -4.0,
    "seed": 42,
    "connectome_steps": 3,
    "micro_steps": 10,
    "v_th": 1.0,
}


class ConnectomeTopology:
    """Immutable random anatomy with Dale's law (E/I polarity per neuron)."""

    def __init__(
        self,
        n_neurons: int = DEFAULTS["n_neurons"],
        k: int = DEFAULTS["k"],
        inhibitory_fraction: float = DEFAULTS["inhibitory_fraction"],
        excitatory_weight: float = DEFAULTS["excitatory_weight"],
        inhibitory_weight: float = DEFAULTS["inhibitory_weight"],
        seed: int = DEFAULTS["seed"],
    ) -> None:
        self.n_neurons = int(n_neurons)
        self.k = int(k)
        self.seed = int(seed)
        self.total = self.n_neurons * self.k

        rng = np.random.default_rng(self.seed)
        is_inhibitory = rng.random(self.n_neurons) < float(inhibitory_fraction)
        self.polarity = np.where(
            is_inhibitory, float(inhibitory_weight), float(excitatory_weight)
        ).astype(np.float64)
        # edges[0] = postsynaptic (dst), edges[1] = presynaptic (src)
        src = rng.integers(0, self.n_neurons, size=self.total)
        dst = rng.integers(0, self.n_neurons, size=self.total)
        self.edges = np.stack([dst, src]).astype(np.int32)
        self.init_mag = (
            np.abs(rng.standard_normal(self.total)) * (0.25 / np.sqrt(self.k))
        ).astype(np.float64)

    def signed_weights(self, magnitudes: np.ndarray) -> np.ndarray:
        """``|w| · polarity[src]`` for every edge."""
        return np.abs(np.asarray(magnitudes, dtype=np.float64)) * self.polarity[self.edges[1]]


class SparseRecurrence:
    """``W_rec @ x`` for a fixed signed edge list, via ``bincount``."""

    def __init__(self, edges: np.ndarray, signed_weights: np.ndarray, n_neurons: int) -> None:
        self.dst = np.asarray(edges[0], dtype=np.int64)
        self.src = np.asarray(edges[1], dtype=np.int64)
        self.w = np.asarray(signed_weights, dtype=np.float64)
        self.n_neurons = int(n_neurons)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        return np.bincount(
            self.dst, weights=self.w * x[self.src], minlength=self.n_neurons
        )


class ConnectomeANN:
    """Sparse recurrent ReLU network (upstream ``ConnectomeANNController``)."""

    def __init__(
        self,
        w_in: np.ndarray,
        w_out: np.ndarray,
        recurrence: SparseRecurrence,
        *,
        steps_per_frame: int = DEFAULTS["connectome_steps"],
        limit: float = 12.0,
    ) -> None:
        self.w_in = np.asarray(w_in, dtype=np.float64)
        self.w_out = np.asarray(w_out, dtype=np.float64)
        self.rec = recurrence
        self.n_neurons = int(self.rec.n_neurons)
        self.n_out = int(self.w_out.shape[0])
        self.steps_per_frame = max(1, int(steps_per_frame))
        self.limit = float(limit)
        self.reset()

    def reset(self) -> None:
        self.h = np.zeros(self.n_neurons, dtype=np.float64)

    def forward_input(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        drive = self.w_in @ x
        for _ in range(self.steps_per_frame):
            self.h = np.maximum(drive + self.rec.matvec(self.h), 0.0)
        return np.clip(self.w_out @ self.h, -self.limit, self.limit)

    @property
    def last_telemetry(self) -> Dict[str, float]:
        return {"h_norm": float(np.linalg.norm(self.h))}


class LosslessConnectomeSNN:
    """Integrate-and-fire transfer of :class:`ConnectomeANN` (numpy)."""

    def __init__(
        self,
        w_in: np.ndarray,
        w_out: np.ndarray,
        recurrence: SparseRecurrence,
        *,
        micro_steps: int = DEFAULTS["micro_steps"],
        v_th: float = DEFAULTS["v_th"],
        limit: float = 12.0,
    ) -> None:
        self.w_in = np.asarray(w_in, dtype=np.float64)
        self.w_out = np.asarray(w_out, dtype=np.float64)
        self.rec = recurrence
        self.n_neurons = int(self.rec.n_neurons)
        self.n_out = int(self.w_out.shape[0])
        self.micro_steps = max(1, int(micro_steps))
        self.v_th = float(v_th)
        self.limit = float(limit)
        self.reset()

    def reset(self) -> None:
        self.v = np.zeros(self.n_neurons, dtype=np.float64)
        self.s = np.zeros(self.n_neurons, dtype=np.float64)
        self.last_spikes = np.zeros(self.n_neurons, dtype=np.uint8)

    def forward_input(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        i_input = self.w_in @ x
        motor = np.zeros(self.n_out, dtype=np.float64)
        for _ in range(self.micro_steps):
            i_syn = i_input + self.rec.matvec(self.s)
            self.v = np.clip(self.v + i_syn, 0.0, None)   # non-negative membrane
            self.s = (self.v >= self.v_th).astype(np.float64)
            self.v = self.v - self.s * self.v_th          # soft reset
            motor += self.w_out @ self.s
        self.last_spikes = self.s.astype(np.uint8)
        return np.clip(motor / self.micro_steps, -self.limit, self.limit)

    def last_telemetry(self, dt: float = 0.02) -> Dict[str, float]:
        rate_hz = float(np.mean(self.s)) * self.micro_steps / float(dt)
        return {
            "spike_rate_hz": rate_hz,
            "active_frac": float(np.mean(self.s)),
            "spikes_this_frame": float(np.sum(self.s)),
            "mean_v": float(np.mean(self.v)),
        }


class ConnectomeController:
    """Adapter implementing the demo's ``act(state, ref)`` controller contract.

    ``kind="ann"`` runs the non-spiking connectome ANN; ``kind="snn"`` runs the
    integrate-and-fire transfer.  Both consume the 13-D policy input and return a
    3-D acceleration demand saturated to ``action_limit``.
    """

    def __init__(
        self,
        kind: str,
        bundle: dict,
        *,
        dt: float = 0.02,
        action_limit: float = 12.0,
        task: Optional[ObstacleGoalTask] = None,
        sensor: Optional[SensorConfig] = None,
    ) -> None:
        kind = str(kind).lower()
        if kind not in ("ann", "snn"):
            raise ValueError(f"kind must be 'ann' or 'snn', got {kind!r}")
        self.kind = kind
        self.name = "ann_connectome" if kind == "ann" else "snn"
        self.action_limit = float(action_limit)
        self.task = task or ObstacleGoalTask()
        self.dt = float(dt)
        self.estimator = ReferenceAccelEstimator(dt=self.dt, plant_gain=1.0, pos_dim=3)
        self.sensor = sensor or SensorConfig()
        self._rng = np.random.default_rng(self.sensor.seed)

        edges = np.asarray(bundle["edges"], dtype=np.int64)
        polarity = np.asarray(bundle["polarity"], dtype=np.float64)
        signed = np.abs(np.asarray(bundle["w_mag"], dtype=np.float64)) * polarity[edges[1]]
        rec = SparseRecurrence(edges, signed, int(bundle["n_neurons"]))
        if kind == "ann":
            self.net = ConnectomeANN(
                bundle["w_in"], bundle["w_out"], rec,
                steps_per_frame=int(bundle.get("connectome_steps", 3)),
                limit=self.action_limit,
            )
        else:
            self.net = LosslessConnectomeSNN(
                bundle["w_in"], bundle["w_out"], rec,
                micro_steps=int(bundle.get("micro_steps", 10)),
                v_th=float(bundle.get("v_th", 1.0)),
                limit=self.action_limit,
            )
        self._last: Dict[str, float] = {}

    def reset(self) -> None:
        self.net.reset()
        self.estimator.reset()
        self._rng = np.random.default_rng(self.sensor.seed)
        self._last = {}

    def act(self, state: np.ndarray, ref) -> np.ndarray:
        scene = (getattr(ref, "meta", None) or {}).get("scene")
        if scene is None:
            raise ValueError("the connectome controller needs ref.meta['scene']")
        x = policy_input(
            state, ref, scene, self.estimator, self.task,
            sensor=self.sensor, rng=self._rng, pos_dim=3,
        )
        u = self.net.forward_input(x)
        if self.kind == "ann":
            self._last = dict(self.net.last_telemetry)
        else:
            self._last = self.net.last_telemetry(self.dt)
        return np.asarray(u, dtype=np.float64)

    @property
    def last_telemetry(self) -> Dict[str, float]:
        return dict(self._last)

    @property
    def last_spikes(self):
        """Per-neuron spike vector of the last frame (SNN), else ``None``."""
        return getattr(self.net, "last_spikes", None)
