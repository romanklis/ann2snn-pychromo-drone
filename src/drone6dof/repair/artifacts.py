"""Run directories and cycle artifacts for the repair loop and inspector."""

from __future__ import annotations

import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .activity import ActivityAudit
from .types import CycleRecord, SplitResult

__all__ = [
    "create_run_dir",
    "snapshot_resume_round",
    "write_json",
    "write_jsonl",
    "append_event",
    "write_baseline",
    "write_cycle",
    "write_final",
    "write_report",
    "list_runs",
    "load_run",
    "load_cycle",
    "load_json",
    "compact_trace",
    "now_iso",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(value):
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path, data) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_finite(data), indent=2, sort_keys=True))
    return str(path)


def write_jsonl(path, rows: List[Dict[str, Any]]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(_finite(row), sort_keys=True) + "\n")
    return str(path)


def append_event(run_dir, event: Dict[str, Any]) -> None:
    path = Path(run_dir) / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(_finite(event), sort_keys=True) + "\n")


def create_run_dir(root, run_id: Optional[str] = None) -> Path:
    root = Path(root)
    if run_id is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_id = f"repair-{stamp}"
    run_id = re.sub(r"[^A-Za-z0-9._-]+", "-", str(run_id)).strip("-") or "repair"
    path = root / run_id
    path.mkdir(parents=True, exist_ok=True)
    for sub in ("baseline", "cycles", "final"):
        (path / sub).mkdir(exist_ok=True)
    return path


def snapshot_resume_round(
    run_dir,
    *,
    round_index: int,
    bundle_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Preserve the previous final/report/bundle before an in-place resume."""
    run_dir = Path(run_dir)
    round_dir = run_dir / "resumes" / f"round-{int(round_index):03d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    final = run_dir / "final"
    if final.exists():
        shutil.copytree(final, round_dir / "final", dirs_exist_ok=True)
    report = run_dir / "report.md"
    if report.exists():
        shutil.copy2(report, round_dir / "report.md")
    if bundle_path:
        bundle = Path(bundle_path)
        if bundle.exists():
            shutil.copy2(bundle, round_dir / "bundle.npz")
    write_json(
        round_dir / "resume.json",
        {
            "round": int(round_index),
            "created_at": now_iso(),
            "bundle": str(bundle_path) if bundle_path else None,
            **(metadata or {}),
        },
    )
    return round_dir


def _split_payload(split: Optional[SplitResult], *, include_traces: bool = False):
    if split is None:
        return None
    return split.to_dict(include_traces=include_traces)


def compact_trace(trace: Dict[str, Any], *, max_frames: int = 300, max_neurons: int = 200) -> Dict[str, Any]:
    """Downsample a frame trace for the browser."""
    if not trace:
        return {}
    n = len(trace.get("t", []))
    step = max(1, int(math.ceil(n / max(1, max_frames))))
    idx = list(range(0, n, step))
    out: Dict[str, Any] = {"stride": step, "frames": len(idx)}
    for key in ("t", "trajectory", "command", "goal_dist", "clearance"):
        values = trace.get(key)
        if values is None:
            out[key] = None
        else:
            out[key] = [values[i] for i in idx if i < len(values)]
    scan = trace.get("scan")
    if scan is not None:
        out["scan"] = [scan[i] for i in idx if i < len(scan)]
    spikes = trace.get("spikes")
    if spikes is not None:
        sampled = []
        for i in idx:
            if i >= len(spikes) or spikes[i] is None:
                sampled.append(None)
            else:
                row = spikes[i]
                sampled.append(row[:max_neurons])
        out["spikes"] = sampled
    return out


def write_baseline(
    run_dir,
    *,
    scenario,
    practice: SplitResult,
    regression: SplitResult,
    activity: ActivityAudit,
    probe: Optional[List[Dict[str, Any]]] = None,
    demo_features: Optional[np.ndarray] = None,
) -> None:
    base = Path(run_dir) / "baseline"
    write_json(base / "metrics.json", {
        "scenario": scenario.to_dict(),
        "practice": practice.metrics(),
        "regression": regression.metrics(),
    })
    write_json(base / "practice.json", practice.to_dict())
    write_json(base / "regression.json", regression.to_dict())
    trace = next((e.trace for e in practice.episodes if e.trace is not None), None)
    if trace is not None:
        write_json(base / "trace.json", compact_trace(trace))
    write_json(base / "audit.json", activity.summary)
    np.savez(
        base / "activity.npz",
        spike_counts=activity.spike_counts,
        spike_frames=activity.spike_frames,
        mean_rate=activity.mean_rate,
        contribution=activity.contribution,
        silent=activity.silent.astype(np.uint8),
        rare=activity.rare.astype(np.uint8),
        spare=activity.spare.astype(np.uint8),
        critical=activity.critical.astype(np.uint8),
        w_in_norm=activity.w_in_norm,
        w_out_norm=activity.w_out_norm,
        bias=activity.bias,
    )
    if probe is not None:
        write_json(base / "probe.json", probe)
    if demo_features is not None:
        write_json(base / "demo_features.json", {
            "shape": list(np.asarray(demo_features).shape),
            "mean": np.asarray(demo_features, dtype=np.float64).mean(axis=0).tolist(),
            "std": np.asarray(demo_features, dtype=np.float64).std(axis=0).tolist(),
        })


def write_probe_report(
    run_dir,
    *,
    summaries: List[Dict[str, Any]],
    criteria: Dict[str, Any],
    status: str = "no_shift",
    suggested_scenario: Optional[str] = None,
) -> Dict[str, Any]:
    """Write the auto-selection probe even when no scenario qualifies."""
    probe = {
        "status": status,
        "criteria": criteria,
        "suggested_scenario": suggested_scenario,
        "scenarios": summaries,
    }
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "probe.json", probe)
    lines = [
        "# Auto-selection probe",
        "",
        f"Status: `{status}`",
        "",
        "Criteria:",
        "",
        "```json",
        json.dumps(_finite(criteria), indent=2, sort_keys=True),
        "```",
        "",
        "| scenario | SNN near | SNN prog | SNN CF | teacher near | teacher prog | teacher CF | gap | ok | reason |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for item in summaries:
        snn = item.get("snn") or {}
        teacher = item.get("teacher") or {}
        reasons = "; ".join(item.get("reasons") or [])
        lines.append(
            f"| {item.get('scenario', '?')} "
            f"| {float(snn.get('near_success_rate', 0.0)):.2f} "
            f"| {float(snn.get('mean_final_progress', 0.0)):.2f} "
            f"| {float(snn.get('collision_free_rate', 0.0)):.2f} "
            f"| {float(teacher.get('near_success_rate', 0.0)):.2f} "
            f"| {float(teacher.get('mean_final_progress', 0.0)):.2f} "
            f"| {float(teacher.get('collision_free_rate', 0.0)):.2f} "
            f"| {float(item.get('gap', 0.0)):.2f} "
            f"| {bool(item.get('eligible', False))} "
            f"| {reasons} |"
        )
    if suggested_scenario:
        lines.extend(["", f"Suggested named scenario: `{suggested_scenario}`"])
    text = "\n".join(lines) + "\n"
    (run_dir / "probe.md").write_text(text)
    (run_dir / "report.md").write_text(text)
    return probe


def write_cycle(
    run_dir,
    record: CycleRecord,
    *,
    candidate: Optional[Dict[str, Any]] = None,
    diff: Optional[Dict[str, Any]] = None,
    practice: Optional[SplitResult] = None,
    regression: Optional[SplitResult] = None,
) -> None:
    cycles = Path(run_dir) / "cycles"
    stem = f"{int(record.cycle):04d}"
    write_json(cycles / f"{stem}.json", record.to_dict())
    if record.program is not None:
        write_json(cycles / f"{stem}.program.json", record.program.to_dict())
    if practice is not None:
        write_json(cycles / f"{stem}.practice.json", practice.to_dict())
        trace = next((e.trace for e in practice.episodes if e.trace is not None), None)
        if trace is not None:
            write_json(cycles / f"{stem}.trace.json", compact_trace(trace))
    if regression is not None:
        write_json(cycles / f"{stem}.regression.json", regression.to_dict())
    if diff is not None:
        write_json(cycles / f"{stem}.diff.json", diff)
    if candidate is not None:
        np.savez(
            cycles / f"{stem}.weights.npz",
            w_in=candidate["w_in"],
            w_out=candidate["w_out"],
            w_mag=candidate["w_mag"],
            edges=candidate["edges"],
            polarity=candidate["polarity"],
            bias=(
                np.asarray(candidate["bias"])
                if candidate.get("bias") is not None
                else np.zeros(int(candidate["w_out"].shape[1]))
            ),
        )
    append_event(run_dir, {
        "type": "cycle",
        "cycle": int(record.cycle),
        "accepted": bool(record.accepted),
        "score": float(record.score),
        "timestamp": record.finished_at or now_iso(),
    })


def write_final(
    run_dir,
    *,
    scenario,
    final_practice: SplitResult,
    final_regression: SplitResult,
    bundle_path: str,
    report_md: str,
    baseline_unseen: Optional[SplitResult] = None,
) -> None:
    final = Path(run_dir) / "final"
    write_json(final / "metrics.json", {
        "scenario": scenario.to_dict(),
        "practice": final_practice.metrics(),
        "regression": final_regression.metrics(),
        "baseline_unseen": (
            None if baseline_unseen is None else baseline_unseen.metrics()
        ),
        "bundle": bundle_path,
    })
    write_json(final / "practice.json", final_practice.to_dict())
    write_json(final / "regression.json", final_regression.to_dict())
    trace = next((e.trace for e in final_practice.episodes if e.trace is not None), None)
    if trace is not None:
        write_json(final / "trace.json", compact_trace(trace))
    (final / "report.md").write_text(report_md)
    append_event(run_dir, {"type": "final", "bundle": bundle_path, "timestamp": now_iso()})


def write_report(run_dir, text: str) -> str:
    path = Path(run_dir) / "report.md"
    path.write_text(text)
    return str(path)


def list_runs(root) -> List[Dict[str, Any]]:
    root = Path(root)
    if not root.exists():
        return []
    out = []
    for path in sorted(root.iterdir(), reverse=True):
        if not path.is_dir():
            continue
        manifest = path / "manifest.json"
        if not manifest.exists():
            continue
        try:
            data = json.loads(manifest.read_text())
        except Exception:
            continue
        final = path / "final" / "metrics.json"
        if final.exists():
            try:
                data["final"] = json.loads(final.read_text())
            except Exception:
                pass
        data["run_id"] = path.name
        out.append(data)
    return out


def load_json(path) -> Optional[Any]:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_run(root, run_id: str) -> Optional[Dict[str, Any]]:
    path = Path(root) / re.sub(r"[^A-Za-z0-9._-]+", "-", str(run_id))
    manifest = load_json(path / "manifest.json")
    if manifest is None:
        return None
    manifest["run_id"] = path.name
    manifest["cycles"] = []
    cycles_dir = path / "cycles"
    if cycles_dir.exists():
        for cycle_path in sorted(cycles_dir.glob("[0-9][0-9][0-9][0-9].json")):
            data = load_json(cycle_path)
            if data is not None:
                manifest["cycles"].append(data)
    manifest["status"] = manifest.get("status", "ok")
    manifest["final"] = load_json(path / "final" / "metrics.json")
    manifest["baseline"] = load_json(path / "baseline" / "metrics.json")
    manifest["baseline_audit"] = load_json(path / "baseline" / "audit.json")
    manifest["baseline_trace"] = load_json(path / "baseline" / "trace.json")
    manifest["probe"] = (
        manifest.get("probe")
        or load_json(path / "probe.json")
        or load_json(path / "baseline" / "probe.json")
    )
    return manifest


def load_cycle(root, run_id: str, cycle: int) -> Optional[Dict[str, Any]]:
    path = Path(root) / re.sub(r"[^A-Za-z0-9._-]+", "-", str(run_id))
    stem = f"{int(cycle):04d}"
    data = load_json(path / "cycles" / f"{stem}.json")
    if data is None:
        return None
    for suffix in ("program", "practice", "regression", "diff", "trace", "transcript"):
        extra = load_json(path / "cycles" / f"{stem}.{suffix}.json")
        if extra is not None:
            data[suffix] = extra
    return data
