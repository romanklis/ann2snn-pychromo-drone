"""CLI for the white-space repair harness.

    python tools/repair.py audit
    python tools/repair.py run --scenario auto --proposer template --cycles 4
    python tools/repair.py run --proposer llm --cycles 8
    python tools/repair.py runs
    python tools/repair.py report --run-id repair-...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from drone6dof.repair.activity import audit_bundle
from drone6dof.repair.artifacts import (
    create_run_dir,
    list_runs,
    load_run,
    now_iso,
    write_baseline,
    write_json,
    write_probe_report,
)
from drone6dof.repair.edits import load_demo_features
from drone6dof.repair.loop import NoShiftError, RepairLoop, RepairRunConfig
from drone6dof.repair.scenarios import (
    REGISTRY,
    auto_select_shift,
    evaluate_bundle,
    evaluate_teacher,
)
from drone6dof.repair.types import Budget
from drone6dof.weights import default_fields, load_weights

__all__ = ["main", "build_parser"]


def _bundle(args) -> Dict[str, Any]:
    return load_weights(args.bundle, expect_fields=default_fields())


def _probe_criteria(args) -> Dict[str, Any]:
    return {
        "probe_episodes": int(args.probe_episodes),
        "probe_steps": int(args.probe_steps),
        "min_teacher_near": float(args.probe_min_teacher_near),
        "min_teacher_progress": float(args.probe_min_teacher_progress),
        "min_teacher_collision_free": float(args.probe_min_teacher_collision_free),
        "max_snn_near": float(args.probe_max_snn_near),
        "max_snn_progress": float(args.probe_max_snn_progress),
        "min_gap": float(args.probe_min_gap),
        "strong_gap": float(args.probe_strong_gap),
    }


def _scenario(args, bundle):
    name = str(args.scenario or "auto").lower()
    if name != "auto":
        if name not in REGISTRY:
            raise SystemExit(f"unknown scenario {name!r}; choose from {sorted(REGISTRY)}")
        return REGISTRY[name], []
    criteria = _probe_criteria(args)
    scenario, summary = auto_select_shift(
        bundle,
        probe_seeds=[int(args.seed) + 50000 + i for i in range(int(args.probe_episodes))],
        probe_steps=int(args.probe_steps),
        max_snn_success=criteria["max_snn_near"],
        max_snn_progress=criteria["max_snn_progress"],
        min_teacher_success=criteria["min_teacher_near"],
        min_teacher_progress=criteria["min_teacher_progress"],
        min_teacher_collision_free=criteria["min_teacher_collision_free"],
        min_gap=criteria["min_gap"],
        strong_gap=criteria["strong_gap"],
    )
    if scenario is None:
        best = (
            sorted(
                summary,
                key=lambda item: (
                    -float(item.get("gap", 0.0)),
                    int(item.get("priority", 999)),
                ),
            )[0]
            if summary
            else None
        )
        suggested = str(best["scenario"]) if best else "goal_behind_wall"
        raise NoShiftError(
            f"auto-select found no qualifying shift; "
            f"try --scenario {suggested}",
            summaries=summary,
            criteria=criteria,
            suggested_scenario=suggested,
        )
    return scenario, summary


def _probe_table(summaries: List[Dict[str, Any]]) -> str:
    lines = [
        f"{'scenario':20s} {'SNN near':>8s} {'SNN prog':>8s} "
        f"{'T near':>7s} {'T prog':>7s} {'T cf':>6s} {'gap':>6s}  ok  reason"
    ]
    for item in summaries:
        snn = item.get("snn") or {}
        teacher = item.get("teacher") or {}
        reason = "; ".join(item.get("reasons") or [])
        lines.append(
            f"{str(item.get('scenario', '?'))[:20]:20s} "
            f"{float(snn.get('near_success_rate', 0.0)):8.2f} "
            f"{float(snn.get('mean_final_progress', 0.0)):8.2f} "
            f"{float(teacher.get('near_success_rate', 0.0)):7.2f} "
            f"{float(teacher.get('mean_final_progress', 0.0)):7.2f} "
            f"{float(teacher.get('collision_free_rate', 0.0)):6.2f} "
            f"{float(item.get('gap', 0.0)):6.2f} "
            f"{str(bool(item.get('eligible', False))):>4s}  {reason}"
        )
    return "\n".join(lines)


def _no_shift_text(exc: NoShiftError, *, run_dir=None) -> str:
    lines = [str(exc)]
    if exc.probe_path:
        lines.append(f"probe: {exc.probe_path}")
    if exc.report_path:
        lines.append(f"report: {exc.report_path}")
    if run_dir and not exc.probe_path:
        lines.append(f"run: {run_dir}")
    lines.extend(["", _probe_table(exc.summaries)])
    return "\n".join(lines)


def _budget(args) -> Budget:
    return Budget(
        max_cycles=int(args.cycles),
        max_turns=int(args.max_turns),
        max_tool_calls=int(args.max_tool_calls),
        max_practice_rollouts=int(args.max_practice_rollouts),
        max_units_per_cycle=int(args.max_units),
        max_new_edges_per_cycle=int(args.max_new_edges),
        max_param_change_per_cycle=int(args.max_param_change),
        quiet_eps=float(args.quiet_eps),
        accept_margin=float(args.accept_margin),
        accept_epsilon=float(args.accept_epsilon),
        regression_margin=float(args.regression_margin),
        strict_quiet=bool(args.strict_quiet),
        prune_units=not bool(args.no_prune),
        max_prunes=int(args.max_prunes),
        max_practice_per_call=int(args.max_practice_per_call),
        cycle_seconds=float(args.cycle_seconds),
    )


def cmd_audit(args) -> int:
    bundle = _bundle(args)
    run_dir = create_run_dir(args.out_root, args.run_id)
    try:
        scenario, probe = _scenario(args, bundle)
    except NoShiftError as exc:
        probe = write_probe_report(
            run_dir,
            summaries=exc.summaries,
            criteria=exc.criteria,
            status="no_shift",
            suggested_scenario=exc.suggested_scenario,
        )
        write_json(
            run_dir / "manifest.json",
            {
                "run_id": run_dir.name,
                "created_at": now_iso(),
                "status": "no_shift",
                "kind": "audit",
                "bundle": str(args.bundle),
                "bundle_fingerprint": bundle.get("fingerprint"),
                "proposer": "none",
                "model": None,
                "budget": _budget(args).to_dict(),
                "criteria": exc.criteria,
                "probe": probe,
                "suggested_scenario": exc.suggested_scenario,
            },
        )
        print(_no_shift_text(exc, run_dir=run_dir))
        return 1
    nominal = REGISTRY["nominal"]
    practice = evaluate_bundle(
        scenario,
        bundle,
        [int(args.seed) + i for i in range(int(args.practice_episodes))],
        split="practice",
        steps=300 if args.fast else int(args.practice_steps),
        capture=True,
    )
    regression = evaluate_bundle(
        nominal,
        bundle,
        [int(args.seed) + 20000 + i for i in range(int(args.regression_episodes))],
        split="regression",
        steps=300 if args.fast else int(args.regression_steps),
    )
    traces = [e.trace for e in practice.episodes if e.trace is not None]
    activity = audit_bundle(
        bundle,
        scenario,
        seeds=[int(args.seed), int(args.seed) + 1],
        steps=min(220, int(args.practice_steps)),
        max_episodes=2,
        traces=traces,
    )
    try:
        demos = load_demo_features()
    except Exception:
        demos = np.zeros((1, 46), dtype=np.float64)
    write_baseline(
        run_dir,
        scenario=scenario,
        practice=practice,
        regression=regression,
        activity=activity,
        probe=probe,
        demo_features=demos,
    )
    manifest = {
        "run_id": run_dir.name,
        "created_at": now_iso(),
        "status": "ok",
        "kind": "audit",
        "bundle": str(args.bundle),
        "bundle_fingerprint": bundle.get("fingerprint"),
        "scenario": scenario.to_dict(),
        "proposer": "none",
        "model": None,
        "budget": _budget(args).to_dict(),
        "probe": probe,
        "audit": activity.summary,
    }
    write_json(run_dir / "manifest.json", manifest)
    result = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "scenario": scenario.name,
        "practice": practice.metrics(),
        "regression": regression.metrics(),
        "activity": activity.summary,
        "probe": probe,
    }
    print(json.dumps(result, indent=2, default=str) if args.json else _audit_text(result))
    return 0


def _audit_text(result: Dict[str, Any]) -> str:
    a = result["activity"]
    p = result["practice"]
    return (
        f"scenario: {result['scenario']}\n"
        f"practice strict: {p['success_rate']:.3f}  "
        f"near-goal: {p.get('near_success_rate', 0.0):.3f}  "
        f"reached: {p['reached_rate']:.3f}  "
        f"collision-free: {p['collision_free_rate']:.3f}\n"
        f"silent: {a['silent_count']}/{a['n_neurons']} ({a['silent_frac']:.1%})  "
        f"rare: {a['rare_count']}  spare: {a['spare_count']}  "
        f"critical: {a['critical_count']}\n"
        f"run: {result['run_dir']}"
    )


def cmd_run(args) -> int:
    config = RepairRunConfig(
        bundle_path=str(args.bundle),
        out_root=str(args.out_root),
        run_id=args.run_id,
        resume_run=args.resume_run,
        scenario=args.scenario,
        proposer=args.proposer,
        model=args.model,
        llm_provider=args.llm_provider,
        budget=_budget(args),
        practice_episodes=int(args.practice_episodes),
        practice_steps=300 if args.fast else int(args.practice_steps),
        unseen_episodes=int(args.unseen_episodes),
        unseen_steps=300 if args.fast else int(args.unseen_steps),
        regression_episodes=int(args.regression_episodes),
        regression_steps=300 if args.fast else int(args.regression_steps),
        probe_episodes=int(args.probe_episodes),
        probe_steps=int(args.probe_steps),
        probe_min_teacher_near=float(args.probe_min_teacher_near),
        probe_min_teacher_progress=float(args.probe_min_teacher_progress),
        probe_min_teacher_collision_free=float(
            args.probe_min_teacher_collision_free
        ),
        probe_max_snn_near=float(args.probe_max_snn_near),
        probe_max_snn_progress=float(args.probe_max_snn_progress),
        probe_min_gap=float(args.probe_min_gap),
        probe_strong_gap=float(args.probe_strong_gap),
        practice_seed_base=int(args.seed),
        patience=int(args.patience),
        allow_live=bool(args.allow_live),
        require_quiet=bool(args.strict_quiet),
        repair_bundle_dir=str(args.repair_bundle_dir),
    )
    try:
        result = RepairLoop(config).run()
    except NoShiftError as exc:
        print(_no_shift_text(exc))
        raise SystemExit(1) from exc
    except Exception as exc:
        raise SystemExit(f"repair run failed: {exc}") from exc
    print(json.dumps(result, indent=2, default=str) if args.json else _run_text(result))
    return 0


def _run_text(result: Dict[str, Any]) -> str:
    b = result["baseline_practice"]
    f = result["final_practice"]
    r = result["final_regression"]
    return (
        f"run: {result['run_id']}  scenario: {result['scenario']}\n"
        f"practice near-goal: {b.get('near_success_rate', 0.0):.3f} -> "
        f"{f.get('near_success_rate', 0.0):.3f}\n"
        f"practice progress: {b.get('mean_final_progress', 0.0):.3f} -> "
        f"{f.get('mean_final_progress', 0.0):.3f}\n"
        f"nominal regression: {r.get('near_success_rate', 0.0):.3f}\n"
        f"cycles: {result['cycles']}  accepted: {result['accepted_cycles']}\n"
        f"bundle: {result['bundle']}\n"
        f"dir: {result['run_dir']}"
    )


def cmd_runs(args) -> int:
    runs = list_runs(args.out_root)
    if args.json:
        print(json.dumps(runs, indent=2, default=str))
    else:
        for run in runs:
            final = (run.get("final") or {}).get("practice", {})
            near = final.get("near_success_rate", final.get("success_rate"))
            progress = final.get("mean_final_progress")
            print(
                f"{run.get('run_id', '?'):40s} "
                f"scenario={str((run.get('scenario') or {}).get('name', '?')):20s} "
                f"final_near={near} final_progress={progress}"
            )
    return 0


def cmd_report(args) -> int:
    root = Path(args.out_root) / str(args.run_id)
    if args.json:
        data = load_run(args.out_root, args.run_id)
        if data is None:
            raise SystemExit(f"no run {args.run_id!r} under {args.out_root}")
        print(json.dumps(data, indent=2, default=str))
        return 0
    report = root / "final" / "report.md"
    if not report.exists():
        raise SystemExit(f"no report at {report}")
    print(report.read_text())
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="White-space repair for the connectome SNN")
    ap.add_argument("--bundle", default="weights/quad6dof_connectome.npz")
    ap.add_argument("--out-root", default="out/repair")
    ap.add_argument("--run-id", default=None)

    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--json", action="store_true")
        p.add_argument("--scenario", default="auto",
                       help="auto or one of " + ", ".join(sorted(REGISTRY)))
        p.add_argument("--seed", type=int, default=1000)
        p.add_argument("--practice-episodes", type=int, default=12)
        p.add_argument("--practice-steps", type=int, default=300)
        p.add_argument("--unseen-episodes", type=int, default=20)
        p.add_argument("--unseen-steps", type=int, default=500)
        p.add_argument("--regression-episodes", type=int, default=6)
        p.add_argument("--regression-steps", type=int, default=300)
        p.add_argument("--probe-episodes", type=int, default=3)
        p.add_argument("--probe-steps", type=int, default=500)
        p.add_argument("--probe-min-teacher-near", type=float, default=0.33)
        p.add_argument("--probe-min-teacher-progress", type=float, default=0.60)
        p.add_argument("--probe-min-teacher-collision-free", type=float, default=0.60)
        p.add_argument("--probe-max-snn-near", type=float, default=0.20)
        p.add_argument("--probe-max-snn-progress", type=float, default=0.20)
        p.add_argument("--probe-min-gap", type=float, default=0.35)
        p.add_argument("--probe-strong-gap", type=float, default=0.75)
        p.add_argument("--cycles", type=int, default=12)
        p.add_argument("--patience", type=int, default=3)
        p.add_argument("--max-turns", type=int, default=30)
        p.add_argument("--max-tool-calls", type=int, default=120)
        p.add_argument("--max-practice-rollouts", type=int, default=18)
        p.add_argument("--max-units", type=int, default=24)
        p.add_argument("--max-new-edges", type=int, default=256)
        p.add_argument("--max-param-change", type=int, default=8192)
        p.add_argument("--quiet-eps", type=float, default=1e-3)
        p.add_argument("--accept-margin", type=float, default=0.02,
                       help="deprecated; acceptance is monotonic")
        p.add_argument("--accept-epsilon", type=float, default=1e-6)
        p.add_argument("--regression-margin", type=float, default=0.02)
        p.add_argument("--strict-quiet", action="store_true",
                       help="make quietness a hard validation failure")
        p.add_argument("--no-prune", action="store_true",
                       help="disable unit-level pruning on rejected programs")
        p.add_argument("--max-prunes", type=int, default=4)
        p.add_argument("--max-practice-per-call", type=int, default=6)
        p.add_argument("--fast", action="store_true",
                       help="300-step smoke mode; not for acceptance claims")
        p.add_argument("--cycle-seconds", type=float, default=60.0)
        p.add_argument("--allow-live", action="store_true")
        p.add_argument("--allow-active", action="store_true",
                       help="legacy alias for the default soft quietness")

    audit = sub.add_parser("audit", help="baseline probe + white-space audit")
    common(audit)
    audit.set_defaults(func=cmd_audit)

    run = sub.add_parser("run", help="run the repair loop")
    common(run)
    run.add_argument("--proposer", default="template",
                     choices=["template", "random", "llm", "gradient"])
    run.add_argument("--model", default=None)
    run.add_argument(
        "--resume-run",
        default=None,
        help="append new cycles to an existing completed run from its final bundle",
    )
    run.add_argument(
        "--llm-provider",
        default=None,
        choices=["openai", "deepseek", "openai-compatible"],
        help="LLM transport preset (default: REPAIR_LLM_PROVIDER or openai)",
    )
    run.add_argument("--repair-bundle-dir", default="weights/repair")
    run.set_defaults(func=cmd_run)

    runs = sub.add_parser("runs", help="list repair runs")
    runs.add_argument("--json", action="store_true")
    runs.set_defaults(func=cmd_runs)

    report = sub.add_parser("report", help="print a run report")
    report.add_argument("--run-id", required=True)
    report.add_argument("--json", action="store_true")
    report.set_defaults(func=cmd_report)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
