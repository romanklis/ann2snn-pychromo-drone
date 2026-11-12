"""Per-neuron activity audit: find the SNN's white space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .scenarios import Scenario, evaluate_bundle

__all__ = [
    "FEATURE_NAMES",
    "ACTION_NAMES",
    "feature_name",
    "ActivityAudit",
    "audit_bundle",
    "write_activity",
]

FEATURE_NAMES: List[str] = (
    ["e_x", "e_y", "e_z", "edot_x", "edot_y", "edot_z", "uff_x", "uff_y", "uff_z"]
    + [f"range_{i:02d}" for i in range(32)]
    + ["min_range", "cos_bearing", "sin_bearing", "slope", "curvature"]
)
ACTION_NAMES = ["ax", "ay", "az"]


def feature_name(index: int) -> str:
    return FEATURE_NAMES[index] if 0 <= index < len(FEATURE_NAMES) else f"feature_{index}"


def _spikes_matrix(traces: Sequence[Dict[str, Any]], n_neurons: int):
    frames = []
    for trace in traces:
        spikes = trace.get("spikes")
        if not spikes:
            continue
        for row in spikes:
            if row is None:
                frames.append(np.zeros(n_neurons, dtype=np.uint8))
            else:
                arr = np.asarray(row, dtype=np.uint8).reshape(-1)
                if arr.size < n_neurons:
                    arr = np.pad(arr, (0, n_neurons - arr.size))
                frames.append(arr[:n_neurons])
    if not frames:
        return np.zeros((0, n_neurons), dtype=np.uint8)
    return np.stack(frames)


def _inputs_matrix(traces: Sequence[Dict[str, Any]], n_in: int):
    rows = []
    for trace in traces:
        inputs = trace.get("inputs")
        if not inputs:
            continue
        for row in inputs:
            if row is None:
                rows.append(np.zeros(n_in, dtype=np.float64))
            else:
                arr = np.asarray(row, dtype=np.float64).reshape(-1)
                if arr.size < n_in:
                    arr = np.pad(arr, (0, n_in - arr.size))
                rows.append(arr[:n_in])
    if not rows:
        return np.zeros((0, n_in), dtype=np.float64)
    return np.stack(rows)


@dataclass
class ActivityAudit:
    summary: Dict[str, Any]
    spikes: np.ndarray
    spike_counts: np.ndarray
    spike_frames: np.ndarray
    mean_rate: np.ndarray
    contribution: np.ndarray
    silent: np.ndarray
    rare: np.ndarray
    spare: np.ndarray
    critical: np.ndarray
    in_degree: np.ndarray
    out_degree: np.ndarray
    w_in_norm: np.ndarray
    w_out_norm: np.ndarray
    bias: np.ndarray
    traces: List[Dict[str, Any]]

    def to_summary(self) -> Dict[str, Any]:
        return self.summary


def audit_bundle(
    bundle: Dict[str, Any],
    scenario: Scenario,
    *,
    seeds: Sequence[int] = (2000,),
    steps: int = 200,
    max_episodes: int = 2,
    rare_frac: float = 0.01,
    top_k: int = 40,
    traces: Optional[Sequence[Dict[str, Any]]] = None,
) -> ActivityAudit:
    """Run a few episodes (or reuse captured traces) and quantify white space."""
    n_neurons = int(bundle["w_out"].shape[1])
    n_in = int(bundle["w_in"].shape[1])
    traces = [t for t in (traces or []) if t is not None][: max(1, int(max_episodes))]
    if not traces:
        traces = []
    for seed in ([] if traces else list(seeds)[: max(1, int(max_episodes))]):
        split = evaluate_bundle(
            scenario,
            bundle,
            [int(seed)],
            split="audit",
            steps=int(steps),
            capture=True,
        )
        for episode in split.episodes:
            if episode.trace is not None:
                traces.append(episode.trace)
    spikes = _spikes_matrix(traces, n_neurons)
    inputs = _inputs_matrix(traces, n_in)
    frames = int(spikes.shape[0])

    exact_counts = None
    exact_frames = None
    for trace in traces:
        if trace.get("spike_counts") is not None:
            arr = np.asarray(trace["spike_counts"], dtype=np.int64).reshape(-1)
            exact_counts = arr if exact_counts is None else exact_counts + arr
        if trace.get("spike_frames") is not None:
            arr = np.asarray(trace["spike_frames"], dtype=np.int64).reshape(-1)
            exact_frames = arr if exact_frames is None else exact_frames + arr
    spike_counts = (
        exact_counts
        if exact_counts is not None and exact_counts.size == n_neurons
        else spikes.sum(axis=0).astype(np.int64)
    )
    spike_frames = (
        exact_frames
        if exact_frames is not None and exact_frames.size == n_neurons
        else (spikes > 0).sum(axis=0).astype(np.int64)
    )
    micro_steps = int(bundle.get("micro_steps", 10))
    denom = max(1, frames * micro_steps)
    mean_rate = spike_counts.astype(np.float64) / float(denom)

    edges = np.asarray(bundle["edges"], dtype=np.int64)
    in_degree = np.bincount(edges[0], minlength=n_neurons)
    out_degree = np.bincount(edges[1], minlength=n_neurons)
    w_in_norm = np.linalg.norm(np.asarray(bundle["w_in"], dtype=np.float64), axis=1)
    w_out_norm = np.linalg.norm(np.asarray(bundle["w_out"], dtype=np.float64), axis=0)
    bias = (
        np.asarray(bundle["bias"], dtype=np.float64)
        if bundle.get("bias") is not None
        else np.zeros(n_neurons, dtype=np.float64)
    )
    contribution = w_out_norm * mean_rate

    silent = spike_counts == 0
    rare = (spike_frames <= max(0.0, rare_frac) * max(1, frames)) & ~silent
    spare = silent | rare
    live = ~spare
    if live.any():
        threshold = float(np.quantile(contribution[live], 0.95))
    else:
        threshold = 0.0
    critical = (contribution >= threshold) & (contribution > 0.0)

    # For each spare unit, the input features it already reads most strongly.
    top_features: Dict[int, List[Dict[str, Any]]] = {}
    spare_rank = np.argsort(-(silent.astype(float) * 2.0 + rare.astype(float) + w_in_norm / (w_in_norm.max() + 1e-9)))
    for unit in spare_rank[: max(1, int(top_k))]:
        row = np.abs(np.asarray(bundle["w_in"], dtype=np.float64)[int(unit)])
        idx = np.argsort(-row)[:6]
        top_features[int(unit)] = [
            {"index": int(i), "name": feature_name(int(i)), "weight": float(row[int(i)])}
            for i in idx
            if row[int(i)] > 0.0
        ]

    top_spare = []
    for unit in spare_rank[: max(1, int(top_k))]:
        u = int(unit)
        top_spare.append(
            {
                "unit": u,
                "silent": bool(silent[u]),
                "rare": bool(rare[u]),
                "spikes": int(spike_counts[u]),
                "frames": int(spike_frames[u]),
                "mean_rate_hz": float(mean_rate[u] * micro_steps / 0.02),
                "w_in_norm": float(w_in_norm[u]),
                "w_out_norm": float(w_out_norm[u]),
                "contribution": float(contribution[u]),
                "in_degree": int(in_degree[u]),
                "out_degree": int(out_degree[u]),
                "bias": float(bias[u]),
                "top_features": top_features.get(u, []),
            }
        )

    summary = {
        "scenario": scenario.name,
        "episodes": len(traces),
        "frames": frames,
        "n_neurons": n_neurons,
        "n_in": n_in,
        "n_edges": int(edges.shape[1]),
        "micro_steps": micro_steps,
        "silent_count": int(silent.sum()),
        "rare_count": int(rare.sum()),
        "spare_count": int(spare.sum()),
        "live_count": int(live.sum()),
        "critical_count": int(critical.sum()),
        "silent_frac": float(silent.mean()) if n_neurons else 0.0,
        "rare_frac": float(rare.mean()) if n_neurons else 0.0,
        "spare_frac": float(spare.mean()) if n_neurons else 0.0,
        "feature_names": list(FEATURE_NAMES),
        "action_names": list(ACTION_NAMES),
        "top_spare": top_spare,
        "critical_units": np.flatnonzero(critical).astype(int).tolist()[:200],
    }
    return ActivityAudit(
        summary=summary,
        spikes=spikes,
        spike_counts=spike_counts,
        spike_frames=spike_frames,
        mean_rate=mean_rate,
        contribution=contribution,
        silent=silent,
        rare=rare,
        spare=spare,
        critical=critical,
        in_degree=in_degree,
        out_degree=out_degree,
        w_in_norm=w_in_norm,
        w_out_norm=w_out_norm,
        bias=bias,
        traces=traces,
    )


def write_activity(audit: ActivityAudit, directory) -> Dict[str, str]:
    """Write ``activity.npz`` and ``audit.json`` to ``directory``."""
    from pathlib import Path
    import json

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    npz_path = path / "activity.npz"
    np.savez(
        npz_path,
        spikes=audit.spikes,
        spike_counts=audit.spike_counts,
        spike_frames=audit.spike_frames,
        mean_rate=audit.mean_rate,
        contribution=audit.contribution,
        silent=audit.silent.astype(np.uint8),
        rare=audit.rare.astype(np.uint8),
        spare=audit.spare.astype(np.uint8),
        critical=audit.critical.astype(np.uint8),
        in_degree=audit.in_degree,
        out_degree=audit.out_degree,
        w_in_norm=audit.w_in_norm,
        w_out_norm=audit.w_out_norm,
        bias=audit.bias,
    )
    json_path = path / "audit.json"
    json_path.write_text(json.dumps(audit.summary, indent=2))
    return {"activity": str(npz_path), "audit": str(json_path)}
