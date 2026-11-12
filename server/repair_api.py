"""Read-only API for white-space repair artifacts.

The repair loop writes ``out/repair/<run_id>/``; this blueprint only serves those
files.  It never runs a simulation or imports torch, so it works in the dashboard
image.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
from flask import Blueprint, jsonify, request

from drone6dof.repair.artifacts import list_runs, load_cycle, load_json, load_run

repair_bp = Blueprint("repair", __name__)

_RUN_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def _root() -> Path:
    return Path(os.environ.get("DRONE6DOF_REPAIR_ROOT", "out/repair"))


def _safe(run_id: str) -> bool:
    return bool(_RUN_ID.match(str(run_id))) and ".." not in str(run_id)


@repair_bp.get("/api/repair/runs")
def repair_runs():
    return jsonify({"runs": list_runs(_root())})


@repair_bp.get("/api/repair/runs/<run_id>")
def repair_run(run_id: str):
    if not _safe(run_id):
        return jsonify({"error": "invalid run id"}), 400
    data = load_run(_root(), run_id)
    if data is None:
        return jsonify({"error": "run not found"}), 404
    return jsonify(data)


@repair_bp.get("/api/repair/runs/<run_id>/cycles/<int:cycle>")
def repair_cycle(run_id: str, cycle: int):
    if not _safe(run_id):
        return jsonify({"error": "invalid run id"}), 400
    data = load_cycle(_root(), run_id, cycle)
    if data is None:
        return jsonify({"error": "cycle not found"}), 404
    return jsonify(data)


@repair_bp.get("/api/repair/runs/<run_id>/activity")
def repair_activity(run_id: str):
    if not _safe(run_id):
        return jsonify({"error": "invalid run id"}), 400
    path = _root() / run_id / "baseline" / "activity.npz"
    if not path.exists():
        return jsonify({"error": "activity not found"}), 404
    with np.load(path, allow_pickle=False) as data:
        keys = (
            "spike_counts", "spike_frames", "mean_rate", "contribution",
            "silent", "rare", "spare", "critical", "w_in_norm", "w_out_norm", "bias",
        )
        payload = {key: data[key].tolist() for key in keys if key in data.files}
    return jsonify(payload)


@repair_bp.get("/api/repair/runs/<run_id>/traces")
def repair_traces(run_id: str):
    """Compact trajectories for the baseline and every cycle that has one."""
    if not _safe(run_id):
        return jsonify({"error": "invalid run id"}), 400
    data = load_run(_root(), run_id)
    if data is None:
        return jsonify({"error": "run not found"}), 404
    run_dir = _root() / run_id
    traces = {
        "baseline": load_json(run_dir / "baseline" / "trace.json"),
        "cycles": {},
    }
    for cycle in data.get("cycles", []):
        number = int(cycle.get("cycle", 0))
        trace = load_json(run_dir / "cycles" / f"{number:04d}.trace.json")
        if trace is not None:
            traces["cycles"][str(number)] = trace
    return jsonify(traces)


@repair_bp.get("/api/repair/runs/<run_id>/weights/<int:cycle>")
def repair_weights(run_id: str, cycle: int):
    if not _safe(run_id):
        return jsonify({"error": "invalid run id"}), 400
    data = load_cycle(_root(), run_id, cycle)
    if data is None:
        return jsonify({"error": "cycle not found"}), 404
    diff = data.get("diff")
    if diff is None:
        return jsonify({"error": "no weight diff for this cycle"}), 404
    return jsonify(diff)


@repair_bp.get("/api/repair/runs/<run_id>/events")
def repair_events(run_id: str):
    if not _safe(run_id):
        return jsonify({"error": "invalid run id"}), 400
    path = _root() / run_id / "events.jsonl"
    if not path.exists():
        return jsonify({"error": "run not found"}), 404
    try:
        since = max(0, int(request.args.get("since", 0)))
    except (TypeError, ValueError):
        return jsonify({"error": "'since' must be an integer"}), 400
    rows = []
    for i, line in enumerate(path.read_text().splitlines()):
        if i < since or not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return jsonify({"events": rows, "next": since + len(rows)})
