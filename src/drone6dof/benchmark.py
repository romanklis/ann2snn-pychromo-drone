"""Batch benchmark over controllers, producing a JSON-serializable report.

Numpy only (no Flask, no torch): the dashboard server is a thin layer over this,
and the tests exercise it directly.  Each controller runs its own
:class:`~drone6dof.sim.Simulation` on a fresh plant, on the same scene, so the
runs are directly comparable.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import (
    CONTROL_LIMIT,
    INIT_STATE,
    MAP_SOURCE_DEFAULT,
    PLANT_GAIN,
    SENSOR,
    STEPS,
    preset_scene,
)
from .dynamics import NumpyPlantBackend
from .params import DT, QuadParams
from .sim import Simulation
from .slam import truth_shell
from .weights import (
    DEFAULT_WEIGHTS_PATH,
    WeightsMissing,
    WeightsMismatch,
    default_fields,
    load_weights,
)

__all__ = [
    "CONTROLLERS",
    "controller_catalogue",
    "weights_info",
    "available_controllers",
    "run_controller",
    "run_benchmark",
]

#: (name, label) in display order
CONTROLLERS = [
    ("ds_guidance", "DS guidance (trainer)"),
    ("pid", "PID baseline"),
    ("ann", "Connectome ANN"),
    ("snn", "Connectome SNN (spiking)"),
    ("field_ann", "Structured field ANN + DS"),
    ("field_snn", "Structured field SNN + DS"),
]

_GUIDE = {
    "ds_guidance": "Analytic dynamical-system teacher; the distillation label source.",
    "pid": "Classical PD + feed-forward baseline; flies straight into the pillar.",
    "ann": "1000-neuron sparse recurrent connectome ANN, distilled from the teacher.",
    "snn": "Integrate-and-fire transfer of the connectome ANN (rate-coded).",
    "field_ann": "Small ANN predicting trajectory-anchored obstacle-field coefficients; DS modulation.",
    "field_snn": "Spiking transfer of the field ANN; the SNN never outputs control commands.",
}

#: capability flags shared by the catalogue and the report
#: ``sensor`` = has a LiDAR scan, ``spiking`` = emits spikes, ``display`` = shown in the dashboard
_META = {
    "ds_guidance": {"sensor": False, "spiking": False, "recurrent": False, "display": True},
    "pid": {"sensor": False, "spiking": False, "recurrent": False, "display": False},
    "ann": {"sensor": True, "spiking": False, "recurrent": True, "display": True},
    "snn": {"sensor": True, "spiking": True, "recurrent": True, "display": True},
    "field_ann": {"sensor": True, "spiking": False, "recurrent": True, "display": True},
    "field_snn": {"sensor": True, "spiking": True, "recurrent": True, "display": True},
}


def controller_catalogue() -> List[dict]:
    return [
        {
            "name": name,
            "label": label,
            "guide": _GUIDE[name],
            **_META[name],
        }
        for name, label in CONTROLLERS
    ]


def weights_info(weights_path=None) -> dict:
    """Load-ability of the connectome bundle (does not raise)."""
    path = str(weights_path or DEFAULT_WEIGHTS_PATH)
    try:
        bundle = load_weights(path, expect_fields=default_fields())
    except WeightsMissing as exc:
        return {"loaded": False, "path": path, "reason": str(exc)}
    except WeightsMismatch as exc:
        return {"loaded": False, "path": path, "reason": str(exc)}
    except Exception as exc:  # pragma: no cover - defensive
        return {"loaded": False, "path": path, "reason": f"{type(exc).__name__}: {exc}"}
    return {
        "loaded": True,
        "path": path,
        "fingerprint": bundle.get("fingerprint"),
        "n_in": int(bundle.get("n_in", 0)),
        "n_out": int(bundle.get("n_out", 0)),
        "n_neurons": int(bundle.get("n_neurons", 0)),
        "connectome_steps": int(bundle.get("connectome_steps", 3)),
        "micro_steps": int(bundle.get("micro_steps", 10)),
    }


def field_weights_info(path=None) -> dict:
    """Load-ability of the structured field bundle (does not raise)."""
    from .weights import (
        DEFAULT_FIELD_PATH,
        WeightsMissing,
        WeightsMismatch,
        load_field_weights,
    )

    path = str(path or DEFAULT_FIELD_PATH)
    try:
        bundle = load_field_weights(path)
    except (WeightsMissing, WeightsMismatch) as exc:
        return {"loaded": False, "path": path, "reason": str(exc)}
    except Exception as exc:  # pragma: no cover - defensive
        return {"loaded": False, "path": path, "reason": f"{type(exc).__name__}: {exc}"}
    return {
        "loaded": True,
        "path": path,
        "fingerprint": bundle.get("fingerprint"),
        "k": int(bundle.get("n_out", 0)),
        "n_neurons": int(bundle.get("n_neurons", 0)),
    }


def available_controllers(weights_path=None) -> Dict[str, dict]:
    info = weights_info(weights_path)
    field_info = field_weights_info()
    out = {}
    for name, _ in CONTROLLERS:
        if name in ("ann", "snn"):
            out[name] = {
                "available": bool(info["loaded"]),
                "reason": "" if info["loaded"] else f"weights not loaded: {info.get('reason', '')}",
            }
        elif name in ("field_ann", "field_snn"):
            out[name] = {
                "available": bool(field_info["loaded"]),
                "reason": "" if field_info["loaded"] else f"field weights not loaded: {field_info.get('reason', '')}",
            }
        else:
            out[name] = {"available": True, "reason": ""}
    return out


def _finite(value):
    """Recursively replace non-finite numbers with ``None`` (browser-safe JSON).

    ``np.nan`` / ``inf`` serialize to the literals ``NaN`` / ``Infinity``, which
    Python's ``json`` accepts but the browser's ``JSON.parse`` rejects — a missing
    telemetry sample used to make the whole dashboard payload unparsable.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, np.floating):
        return _finite(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    return value


def _telemetry_matrix(history: Dict[str, list]) -> Dict[str, list]:
    keys = set()
    for frame in history["telemetry"]:
        keys.update(frame.keys())
    out = {}
    for key in sorted(keys):
        out[key] = [_finite(frame.get(key)) for frame in history["telemetry"]]
    return out


def _surface_coverage(scene, map_, traj_xy) -> float:
    """Recall of the truth obstacle shell cells the sensor could observe."""
    shell = truth_shell(scene, map_)
    X, Y = np.meshgrid(map_.xs, map_.ys, indexing="ij")
    cells = np.stack([X.ravel(), Y.ravel()], axis=1)
    traj = np.asarray(traj_xy, dtype=np.float64)[::5]
    if len(traj) == 0:
        return 0.0
    d2 = ((cells[:, None, :] - traj[None, :, :]) ** 2).sum(axis=-1)
    visible = (d2.min(axis=1) <= SENSOR.r_max ** 2).reshape(X.shape)
    vis_shell = np.where((shell == 2) & visible, 2, 0)
    return float(map_.surface_recall(vis_shell))


def run_controller(
    name: str,
    scene,
    *,
    steps: int = STEPS,
    dt: float = DT,
    gain: float = PLANT_GAIN,
    control_limit: float = CONTROL_LIMIT,
    weights_path=None,
    seed: int = 0,
    map_source: str = MAP_SOURCE_DEFAULT,
) -> dict:
    from .cli import make_controller  # local import avoids an import cycle at module load

    backend = NumpyPlantBackend(QuadParams(), heading_target=scene.goal_np)
    controller = make_controller(name, scene, control_limit, weights_path)
    sim = Simulation(
        backend,
        controller,
        scene,
        dt=dt,
        steps=steps,
        gain=gain,
        control_limit=control_limit,
        initial_state=INIT_STATE,
        map_source=map_source,
    )
    sim.run()
    history = sim.history
    state = np.asarray(history["state"], dtype=np.float64)
    metrics = sim.metrics()
    if history["estimate"]:
        err = np.asarray(history["est_error"], dtype=np.float64)
        metrics = dict(metrics)
        metrics["pos_rmse_m"] = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))

    spikes = None
    if any(s is not None for s in history["spikes"]):
        spikes = np.zeros((len(history["spikes"]), 200), dtype=np.uint8)
        for i, frame in enumerate(history["spikes"]):
            if frame is not None:
                spikes[i, : len(frame)] = frame

    scan = None
    frames = history["scan"]
    if any(s is not None for s in frames):
        k = len(next(s for s in frames if s is not None))
        scan = np.full((len(frames), k), np.nan, dtype=np.float64)
        for i, frame in enumerate(frames):
            if frame is not None:
                scan[i] = frame
    angles = getattr(controller, "scan_angles", None)

    map_payload = None
    slam_payload = None
    if sim.slam_map is not None:
        frames = history.get("map_frames") or []
        shape = [len(frames[0]), len(frames[0][0])] if frames else [0, 0]
        map_payload = {
            "shape": shape,
            "bounds": [float(v) for v in sim.slam_map.cfg.bounds],
            "times": [float(t) for t in history.get("map_times", [])],
            "frames": frames,
        }
        m = metrics
        slam_payload = {
            "explored_frac": [float(v) for v in history.get("slam_explored", [])],
            "entropy_bits": [float(v) for v in history.get("slam_entropy", [])],
            "occupied_cells": [int(v) for v in history.get("slam_occupied", [])],
            "replans": int(sim.slam_map.replans),
            "path_found_step": m.get("slam_path_found_step", -1),
            "iou_vs_truth": m.get("slam_iou_vs_truth", 0.0),
            "surface_coverage": _surface_coverage(scene, sim.slam_map, state[:, :2]),
        }

    label = dict(CONTROLLERS).get(name, name)
    return {
        "controller": name,
        "label": label,
        "spiking": _META[name]["spiking"],
        "sensor": _META[name]["sensor"],
        "recurrent": _META[name]["recurrent"],
        "t": [float(k * dt) for k in range(len(state))],
        "trajectory": state[:, :3].tolist(),
        "state": state.tolist(),
        "attitude": np.asarray(history["rpy"], dtype=np.float64).tolist(),
        "command": np.asarray(history["command"], dtype=np.float64).tolist(),
        "goal_dist": [float(v) for v in history["goal_dist"]],
        "clearance": [float(v) for v in history["clearance"]],
        "success": [bool(v) for v in (scene.clearance_series(state) >= 0.0)],
        "telemetry": _telemetry_matrix(history),
        "spikes": None if spikes is None else spikes.tolist(),
        "scan": None if scan is None else np.round(scan, 3).tolist(),
        "scan_angles": None if (scan is None or angles is None) else [float(a) for a in angles],
        "estimate": None if not history["estimate"] else np.round(
            np.asarray(history["estimate"], dtype=np.float64), 3
        ).tolist(),
        "metrics": metrics,
        "map": map_payload,
        "slam": slam_payload,
    }


def run_benchmark(
    names: Optional[Sequence[str]] = None,
    *,
    steps: int = STEPS,
    dt: float = DT,
    gain: float = PLANT_GAIN,
    control_limit: float = CONTROL_LIMIT,
    weights_path=None,
    seed: int = 0,
    goal: Optional[Sequence[float]] = None,
    scene_name: Optional[str] = None,
    map_source: str = MAP_SOURCE_DEFAULT,
) -> dict:
    """Run each requested controller on one scene and return the full report.

    ``goal`` overrides the shipped goal (interactive command); ``scene_name``
    selects a named obstacle preset (``pillar``, ``boxes``, ``wall``, ``slalom``);
    ``map_source`` is ``"truth"`` (privileged scene) or ``"slam"`` (online map).
    Raises ``ValueError`` for a malformed goal or unknown scene.
    """
    requested = list(names) if names else [name for name, _ in CONTROLLERS]
    known = {name for name, _ in CONTROLLERS}
    unknown = [n for n in requested if n not in known]
    if unknown:
        raise KeyError(f"unknown controllers: {', '.join(unknown)}")

    scene = preset_scene(scene_name, goal=goal)
    available = available_controllers(weights_path)

    results: Dict[str, dict] = {}
    unavailable: Dict[str, str] = {}
    for name in requested:
        if not available[name]["available"]:
            unavailable[name] = available[name]["reason"]
            continue
        results[name] = run_controller(
            name, scene, steps=steps, dt=dt, gain=gain,
            control_limit=control_limit, weights_path=weights_path, seed=seed,
            map_source=map_source,
        )

    per_controller = {
        name: {
            "mean_goal_dist_m": res["metrics"]["mean_goal_dist_m"],
            "final_goal_dist_m": res["metrics"]["final_goal_dist_m"],
            "clearance_min_m": res["metrics"]["clearance_min_m"],
            "collisions": res["metrics"]["collisions"],
            "reached_goal": res["metrics"]["reached_goal"],
            "peak_g_force": res["metrics"]["peak_g_force"],
        }
        for name, res in results.items()
    }
    ranking = sorted(per_controller, key=lambda n: (
        per_controller[n]["collisions"],
        per_controller[n]["final_goal_dist_m"],
    ))

    delta = None
    if "ann" in per_controller and "snn" in per_controller:
        delta = {
            "mean_goal_dist_m": per_controller["snn"]["mean_goal_dist_m"]
            - per_controller["ann"]["mean_goal_dist_m"],
            "final_goal_dist_m": per_controller["snn"]["final_goal_dist_m"]
            - per_controller["ann"]["final_goal_dist_m"],
            "clearance_min_m": per_controller["snn"]["clearance_min_m"]
            - per_controller["ann"]["clearance_min_m"],
        }

    return _finite({
        "example": "quad6dof",
        "scene": scene.to_dict(),
        "scene_name": scene_name or "pillar",
        "goal": [float(v) for v in scene.goal],
        "steps": int(steps),
        "dt": float(dt),
        "seed": int(seed),
        "map_source": str(map_source),
        "controllers": [n for n in requested if n in results],
        "unavailable": unavailable,
        "results": results,
        "stats": {
            "ranking": ranking,
            "per_controller": per_controller,
            "snn_minus_ann": delta,
        },
        "weights": weights_info(weights_path),
    })
