"""The white-space repair loop: audit, propose, validate, evaluate, record."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..weights import default_fields, load_weights
from .activity import audit_bundle, write_activity
from .artifacts import (
    append_event,
    create_run_dir,
    load_json,
    load_run,
    now_iso,
    snapshot_resume_round,
    write_baseline,
    write_cycle,
    write_final,
    write_json,
    write_probe_report,
    write_report,
)
from .edits import (
    apply_program,
    clone_bundle,
    diff_bundles,
    load_demo_features,
    program_subsets,
    program_units,
    save_repaired_bundle,
    validate_candidate,
)
from .proposers.base import ProposalContext
from .proposers.llm import LLMConfig, LLMProposer
from .proposers.random import RandomProposer
from .proposers.template import TemplateProposer
from .scenarios import (
    REGISTRY,
    Scenario,
    auto_select_shift,
    evaluate_bundle,
    evaluate_teacher,
)
from .types import Budget, CycleRecord, Diagnosis, EditProgram, SplitResult, ValidationReport

__all__ = [
    "NoShiftError",
    "ResumeError",
    "RepairRunConfig",
    "RepairLoop",
    "run_repair",
    "make_proposer",
]


@dataclass
class RepairRunConfig:
    bundle_path: str = "weights/quad6dof_connectome.npz"
    out_root: str = "out/repair"
    run_id: Optional[str] = None
    resume_run: Optional[str] = None
    scenario: str = "auto"
    proposer: str = "template"
    model: Optional[str] = None
    llm_provider: Optional[str] = None
    budget: Budget = field(default_factory=Budget)
    practice_episodes: int = 12
    practice_steps: int = 500
    unseen_episodes: int = 20
    unseen_steps: int = 500
    regression_episodes: int = 6
    regression_steps: int = 500
    probe_episodes: int = 3
    probe_steps: int = 500
    probe_min_teacher_near: float = 0.33
    probe_min_teacher_progress: float = 0.60
    probe_min_teacher_collision_free: float = 0.60
    probe_max_snn_near: float = 0.20
    probe_max_snn_progress: float = 0.20
    probe_min_gap: float = 0.35
    probe_strong_gap: float = 0.75
    practice_seed_base: int = 1000
    unseen_seed_base: int = 10000
    regression_seed_base: int = 20000
    patience: int = 3
    allow_live: bool = False
    require_quiet: bool = False
    repair_bundle_dir: str = "weights/repair"


def make_proposer(config: RepairRunConfig, *, client=None):
    name = str(config.proposer).lower()
    if name == "template":
        return TemplateProposer()
    if name == "random":
        return RandomProposer(seed=0)
    if name == "llm":
        return LLMProposer(
            config=LLMConfig.from_env(
                provider=config.llm_provider, model=config.model
            ),
            client=client,
        )
    if name == "gradient":
        from .proposers.gradient import GradientProposer

        return GradientProposer()
    raise ValueError(f"unknown proposer {config.proposer!r}")


def _seeds(base: int, n: int) -> List[int]:
    return [int(base) + i for i in range(max(0, int(n)))]


def _score(split: SplitResult) -> float:
    """Primary monotonic objective: mean final progress toward the goal."""
    return float(split.mean_final_progress)


def _accept_candidate(
    practice: SplitResult,
    regression: SplitResult,
    baseline_practice: SplitResult,
    baseline_regression: SplitResult,
    best_score: float,
    epsilon: float,
    guard_margin: float = 0.02,
) -> bool:
    """Accept any absolute progress gain that does not damage safety/nominal."""
    return (
        float(practice.mean_final_progress) > float(best_score) + float(epsilon)
        and practice.collision_free_rate
        >= baseline_practice.collision_free_rate - float(guard_margin)
        and regression.success_rate
        >= baseline_regression.success_rate - float(guard_margin)
        and regression.near_success_rate
        >= baseline_regression.near_success_rate - float(guard_margin)
        and regression.collision_free_rate
        >= baseline_regression.collision_free_rate - float(guard_margin)
        and regression.mean_final_progress
        >= baseline_regression.mean_final_progress - float(guard_margin)
    )


class NoShiftError(RuntimeError):
    """Auto-selection found no qualifying shift; the probe was persisted."""

    def __init__(
        self,
        message: str,
        *,
        summaries: List[Dict[str, Any]],
        criteria: Dict[str, Any],
        suggested_scenario: Optional[str] = None,
        run_dir: Optional[str] = None,
        probe_path: Optional[str] = None,
        report_path: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.summaries = summaries
        self.criteria = criteria
        self.suggested_scenario = suggested_scenario
        self.run_dir = run_dir
        self.probe_path = probe_path
        self.report_path = report_path


class ResumeError(RuntimeError):
    """A requested run cannot be resumed safely."""


class RepairLoop:
    def __init__(
        self,
        config: RepairRunConfig,
        *,
        bundle: Optional[Dict[str, Any]] = None,
        scenario: Optional[Scenario] = None,
        proposer=None,
        client=None,
    ) -> None:
        self.config = config
        self.proposer = proposer or make_proposer(config, client=client)
        self.root = Path(config.out_root)
        self.activity = None
        self.probe_summary: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self.manifest: Dict[str, Any] = {}
        self.resume_info: Optional[Dict[str, Any]] = None
        self.start_cycle = 1

        if config.resume_run:
            if bundle is not None:
                raise ResumeError("--bundle cannot be combined with --resume-run")
            loaded = load_run(self.root, config.resume_run)
            if loaded is None:
                raise ResumeError(
                    f"run {config.resume_run!r} not found under {self.root}"
                )
            manifest = dict(loaded)
            if manifest.get("status") != "ok":
                raise ResumeError(
                    f"run {config.resume_run!r} has status "
                    f"{manifest.get('status')!r}; only completed runs can resume"
                )
            final_metrics = manifest.get("final") or {}
            bundle_path = manifest.get("bundle_out") or final_metrics.get("bundle")
            base_path = manifest.get("bundle")
            if not bundle_path or not Path(str(bundle_path)).exists():
                raise ResumeError(
                    f"run {config.resume_run!r} has no final repaired bundle"
                )
            if not base_path or not Path(str(base_path)).exists():
                raise ResumeError(
                    f"run {config.resume_run!r} has no original base bundle"
                )
            scenario_name = (manifest.get("scenario") or {}).get("name")
            if scenario_name not in REGISTRY:
                raise ResumeError(
                    f"run scenario {scenario_name!r} is not in the current registry"
                )
            if scenario is not None and scenario.name != scenario_name:
                raise ResumeError(
                    f"resume scenario is {scenario_name!r}, not {scenario.name!r}"
                )
            self.base_bundle = load_weights(
                base_path, expect_fields=default_fields()
            )
            self.bundle = load_weights(
                bundle_path, expect_fields=default_fields()
            )
            self.scenario = REGISTRY[scenario_name]
            self.config.bundle_path = str(base_path)
            self.config.run_id = str(manifest.get("run_id") or config.resume_run)
            self.run_dir = create_run_dir(self.root, self.config.run_id)
            self.history = list(manifest.get("cycles") or [])
            self.start_cycle = (
                max(int(item.get("cycle", 0)) for item in self.history) + 1
                if self.history
                else 1
            )
            splits = manifest.get("splits") or {}
            for attr in (
                "practice_episodes",
                "practice_steps",
                "unseen_episodes",
                "unseen_steps",
                "regression_episodes",
                "regression_steps",
                "practice_seed_base",
                "unseen_seed_base",
                "regression_seed_base",
            ):
                if attr in splits:
                    setattr(self.config, attr, int(splits[attr]))
            round_index = int(manifest.get("resume_count", 0)) + 1
            self.resume_info = {
                "manifest": manifest,
                "bundle_path": str(bundle_path),
                "base_path": str(base_path),
                "round": round_index,
                "start_cycle": self.start_cycle,
                "previous_final": final_metrics,
                "requested_scenario": config.scenario,
            }
            self.probe_summary = list(manifest.get("probe") or [])
        else:
            self.bundle = bundle if bundle is not None else load_weights(
                config.bundle_path, expect_fields=default_fields()
            )
            self.base_bundle = self.bundle
            self.scenario = scenario
            self.run_dir = create_run_dir(self.root, config.run_id)

        self.demo_features = self._load_demos()

    def _load_demos(self) -> np.ndarray:
        try:
            return load_demo_features()
        except Exception:
            nominal = REGISTRY["nominal"]
            split = evaluate_teacher(
                nominal, [self.config.practice_seed_base], split="demos", steps=200, capture=True
            )
            trace = next((e.trace for e in split.episodes if e.trace is not None), None)
            rows = []
            if trace and trace.get("inputs"):
                rows = [row for row in trace["inputs"] if row is not None]
            if not rows:
                return np.zeros((1, 46), dtype=np.float64)
            return np.asarray(rows, dtype=np.float64)

    def _model_name(self) -> Optional[str]:
        cfg = getattr(self.proposer, "config", None)
        return getattr(cfg, "model", None) or self.config.model

    def _provider_name(self) -> Optional[str]:
        cfg = getattr(self.proposer, "config", None)
        return getattr(cfg, "provider", None) or self.config.llm_provider

    def _resume_setup(self, nominal: Scenario):
        info = self.resume_info or {}
        run_dir = self.run_dir
        round_dir = snapshot_resume_round(
            run_dir,
            round_index=int(info["round"]),
            bundle_path=info.get("bundle_path"),
            metadata={
                "resumed_from": info.get("bundle_path"),
                "start_cycle": int(info["start_cycle"]),
                "scenario": self.scenario.name,
            },
        )
        info["round_dir"] = str(round_dir)

        def _saved_split(name: str, scenario, fallback_seeds, steps, split):
            payload = load_json(run_dir / "baseline" / f"{name}.json")
            if payload:
                return SplitResult.from_dict(payload)
            return evaluate_bundle(
                scenario,
                self.base_bundle,
                fallback_seeds,
                split=split,
                steps=steps,
            )

        practice_seeds = _seeds(
            self.config.practice_seed_base, self.config.practice_episodes
        )
        regression_seeds = _seeds(
            self.config.regression_seed_base, self.config.regression_episodes
        )
        baseline_practice = _saved_split(
            "practice", self.scenario, practice_seeds,
            self.config.practice_steps, "practice"
        )
        baseline_regression = _saved_split(
            "regression", nominal, regression_seeds,
            self.config.regression_steps, "regression"
        )
        resume_practice = evaluate_bundle(
            self.scenario,
            self.bundle,
            practice_seeds,
            split="practice",
            steps=self.config.practice_steps,
            capture=True,
        )
        resume_regression = evaluate_bundle(
            nominal,
            self.bundle,
            regression_seeds,
            split="regression",
            steps=self.config.regression_steps,
        )
        traces = [e.trace for e in resume_practice.episodes if e.trace is not None]
        self.activity = audit_bundle(
            self.bundle,
            self.scenario,
            seeds=practice_seeds[:2],
            steps=min(220, int(self.config.practice_steps)),
            max_episodes=2,
            traces=traces,
        )
        write_activity(self.activity, round_dir)
        write_json(
            round_dir / "resumed_baseline.json",
            {
                "practice": resume_practice.metrics(),
                "regression": resume_regression.metrics(),
                "bundle": info.get("bundle_path"),
                "start_cycle": int(info["start_cycle"]),
            },
        )
        target_features = self._target_features(resume_practice)

        accepted_cycles = [
            int(item.get("cycle", 0))
            for item in self.history
            if item.get("accepted")
        ]
        best_cycle = max(accepted_cycles) if accepted_cycles else 0
        best_score = _score(resume_practice)

        manifest = {
            key: value
            for key, value in (info.get("manifest") or {}).items()
            if key not in {"cycles", "final", "baseline", "baseline_audit", "baseline_trace"}
        }
        manifest["status"] = "ok"
        manifest["resume_count"] = int(info["round"])
        manifest["resumed_at"] = now_iso()
        manifest["bundle_in"] = info.get("bundle_path")
        manifest["start_cycle"] = int(info["start_cycle"])
        manifest["budget_current"] = self.config.budget.to_dict()
        manifest["proposer_current"] = str(self.config.proposer)
        manifest["model_current"] = self._model_name()
        manifest["provider_current"] = self._provider_name()
        manifest["splits"] = {
            "practice_episodes": self.config.practice_episodes,
            "practice_steps": self.config.practice_steps,
            "unseen_episodes": self.config.unseen_episodes,
            "unseen_steps": self.config.unseen_steps,
            "regression_episodes": self.config.regression_episodes,
            "regression_steps": self.config.regression_steps,
            "probe_episodes": self.config.probe_episodes,
            "practice_seed_base": self.config.practice_seed_base,
            "unseen_seed_base": self.config.unseen_seed_base,
            "regression_seed_base": self.config.regression_seed_base,
        }
        resumes = list(manifest.get("resumes") or [])
        resumes.append(
            {
                "round": int(info["round"]),
                "resumed_at": manifest["resumed_at"],
                "from_bundle": info.get("bundle_path"),
                "start_cycle": int(info["start_cycle"]),
                "practice": resume_practice.metrics(),
                "regression": resume_regression.metrics(),
                "round_dir": str(round_dir),
            }
        )
        manifest["resumes"] = resumes
        write_json(run_dir / "manifest.json", manifest)
        append_event(
            run_dir,
            {
                "type": "resume",
                "round": int(info["round"]),
                "start_cycle": int(info["start_cycle"]),
                "bundle": info.get("bundle_path"),
                "timestamp": now_iso(),
            },
        )
        self.manifest = manifest
        return (
            baseline_practice,
            baseline_regression,
            clone_bundle(self.bundle),
            resume_practice,
            resume_regression,
            best_score,
            best_cycle,
            target_features,
            practice_seeds,
            regression_seeds,
        )

    def _target_features(self, split: SplitResult) -> np.ndarray:
        trace = next((e.trace for e in split.episodes if e.trace is not None), None)
        if trace and trace.get("inputs"):
            rows = [row for row in trace["inputs"] if row is not None]
            if rows:
                return np.asarray(rows, dtype=np.float64)
        return np.zeros((0, int(self.bundle["w_in"].shape[1])), dtype=np.float64)

    def _resolve_scenario(self) -> Tuple[Scenario, List[Dict[str, Any]]]:
        if self.scenario is not None:
            return self.scenario, []
        name = str(self.config.scenario or "auto").lower()
        if name != "auto":
            if name not in REGISTRY:
                raise ValueError(f"unknown scenario {name!r}; choose from {sorted(REGISTRY)}")
            return REGISTRY[name], []
        criteria = {
            "probe_episodes": int(self.config.probe_episodes),
            "probe_steps": int(self.config.probe_steps),
            "min_teacher_near": float(self.config.probe_min_teacher_near),
            "min_teacher_progress": float(self.config.probe_min_teacher_progress),
            "min_teacher_collision_free": float(
                self.config.probe_min_teacher_collision_free
            ),
            "max_snn_near": float(self.config.probe_max_snn_near),
            "max_snn_progress": float(self.config.probe_max_snn_progress),
            "min_gap": float(self.config.probe_min_gap),
            "strong_gap": float(self.config.probe_strong_gap),
        }
        scenario, summary = auto_select_shift(
            self.bundle,
            probe_seeds=_seeds(
                self.config.practice_seed_base + 50000, self.config.probe_episodes
            ),
            probe_steps=int(self.config.probe_steps),
            max_snn_success=float(self.config.probe_max_snn_near),
            max_snn_progress=float(self.config.probe_max_snn_progress),
            min_teacher_success=float(self.config.probe_min_teacher_near),
            min_teacher_progress=float(self.config.probe_min_teacher_progress),
            min_teacher_collision_free=float(
                self.config.probe_min_teacher_collision_free
            ),
            min_gap=float(self.config.probe_min_gap),
            strong_gap=float(self.config.probe_strong_gap),
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
            probe = write_probe_report(
                self.run_dir,
                summaries=summary,
                criteria=criteria,
                status="no_shift",
                suggested_scenario=suggested,
            )
            manifest = {
                "run_id": self.run_dir.name,
                "created_at": now_iso(),
                "status": "no_shift",
                "bundle": str(self.config.bundle_path),
                "bundle_fingerprint": self.bundle.get("fingerprint"),
                "proposer": str(self.config.proposer),
                "model": self._model_name(),
                "provider": self._provider_name(),
                "budget": self.config.budget.to_dict(),
                "criteria": criteria,
                "probe": probe,
                "suggested_scenario": suggested,
            }
            write_json(self.run_dir / "manifest.json", manifest)
            append_event(
                self.run_dir,
                {
                    "type": "no_shift",
                    "suggested_scenario": suggested,
                    "timestamp": now_iso(),
                },
            )
            probe_path = self.run_dir / "probe.json"
            report_path = self.run_dir / "probe.md"
            raise NoShiftError(
                f"auto-select found no qualifying shift; probe report: "
                f"{probe_path}; try --scenario {suggested}",
                summaries=summary,
                criteria=criteria,
                suggested_scenario=suggested,
                run_dir=str(self.run_dir),
                probe_path=str(probe_path),
                report_path=str(report_path),
            )
        return scenario, summary

    def _diagnose(self, practice: SplitResult) -> Diagnosis:
        failed = [e for e in practice.episodes if e.failure != "none"][:5]
        examples = []
        for episode in failed:
            item = episode.summary()
            trace = episode.trace or {}
            goal = trace.get("goal_dist") or []
            scan = trace.get("scan") or []
            command = trace.get("command") or []
            item["tail_goal_dist"] = goal[-8:] if goal else []
            item["min_range"] = (
                float(np.min(np.asarray(scan, dtype=np.float64)[:, 41]))
                if scan and len(scan) and len(scan[0]) > 41
                else None
            )
            item["mean_abs_command"] = (
                float(np.mean(np.abs(np.asarray(command, dtype=np.float64))))
                if command else None
            )
            examples.append(item)
        return Diagnosis(
            scenario=practice.scenario,
            split_metrics=practice.metrics(),
            failure_counts=practice.failure_counts(),
            activity=self.activity.summary if self.activity is not None else {},
            examples=examples,
            history=self.history[-8:],
            notes=(
                "Repair the dominant failure without changing the nominal distribution."
            ),
        )

    def _record_rejection(
        self,
        cycle: int,
        program: Optional[EditProgram],
        validation: ValidationReport,
        *,
        note: str,
        started: str,
    ) -> CycleRecord:
        record = CycleRecord(
            cycle=cycle,
            scenario=self.scenario.name,
            proposer=getattr(self.proposer, "name", self.config.proposer),
            model=self._model_name(),
            program=program,
            validation=validation,
            practice=None,
            regression=None,
            accepted=False,
            score=float("nan"),
            budget=self.config.budget.to_dict(),
            started_at=started,
            finished_at=now_iso(),
            note=note,
        )
        write_cycle(self.run_dir, record)
        self.history.append(record.to_dict())
        return record

    def run(self) -> Dict[str, Any]:
        started = time.time()
        nominal = REGISTRY["nominal"]
        if self.resume_info:
            (
                baseline_practice,
                baseline_regression,
                best,
                best_practice,
                best_regression,
                best_score,
                best_cycle,
                target_features,
                practice_seeds,
                regression_seeds,
            ) = self._resume_setup(nominal)
            start_cycle = self.start_cycle
        else:
            self.scenario, self.probe_summary = self._resolve_scenario()
            self.manifest = {
                "run_id": self.run_dir.name,
                "created_at": now_iso(),
                "status": "ok",
                "bundle": str(self.config.bundle_path),
                "bundle_fingerprint": self.bundle.get("fingerprint"),
                "scenario": self.scenario.to_dict(),
                "proposer": str(self.config.proposer),
                "model": self._model_name(),
                "provider": self._provider_name(),
                "budget": self.config.budget.to_dict(),
                "splits": {
                    "practice_episodes": self.config.practice_episodes,
                    "practice_steps": self.config.practice_steps,
                    "unseen_episodes": self.config.unseen_episodes,
                    "unseen_steps": self.config.unseen_steps,
                    "regression_episodes": self.config.regression_episodes,
                    "regression_steps": self.config.regression_steps,
                    "probe_episodes": self.config.probe_episodes,
                    "practice_seed_base": self.config.practice_seed_base,
                    "unseen_seed_base": self.config.unseen_seed_base,
                    "regression_seed_base": self.config.regression_seed_base,
                },
                "probe": self.probe_summary,
            }
            write_json(self.run_dir / "manifest.json", self.manifest)
            append_event(
                self.run_dir,
                {
                    "type": "start",
                    "scenario": self.scenario.name,
                    "timestamp": now_iso(),
                },
            )

            practice_seeds = _seeds(
                self.config.practice_seed_base, self.config.practice_episodes
            )
            regression_seeds = _seeds(
                self.config.regression_seed_base, self.config.regression_episodes
            )
            baseline_practice = evaluate_bundle(
                self.scenario,
                self.bundle,
                practice_seeds,
                split="practice",
                steps=self.config.practice_steps,
                capture=True,
            )
            baseline_regression = evaluate_bundle(
                nominal,
                self.bundle,
                regression_seeds,
                split="regression",
                steps=self.config.regression_steps,
            )
            traces = [e.trace for e in baseline_practice.episodes if e.trace is not None]
            self.activity = audit_bundle(
                self.bundle,
                self.scenario,
                seeds=practice_seeds[:2],
                steps=min(220, int(self.config.practice_steps)),
                max_episodes=2,
                traces=traces,
            )
            write_baseline(
                self.run_dir,
                scenario=self.scenario,
                practice=baseline_practice,
                regression=baseline_regression,
                activity=self.activity,
                probe=self.probe_summary,
                demo_features=self.demo_features,
            )
            target_features = self._target_features(baseline_practice)

            best = clone_bundle(self.bundle)
            best_practice = baseline_practice
            best_regression = baseline_regression
            best_score = _score(baseline_practice)
            best_cycle = 0
            start_cycle = 1

        patience = 0
        for cycle in range(
            start_cycle, start_cycle + int(self.config.budget.max_cycles)
        ):
            cycle_started = time.time()
            started_at = now_iso()
            diagnosis = self._diagnose(best_practice)
            practice_callback = lambda limit: evaluate_bundle(
                self.scenario,
                best,
                practice_seeds[: max(1, int(limit))],
                split="llm-practice",
                steps=self.config.practice_steps,
            )
            candidate_callback = lambda candidate, limit: evaluate_bundle(
                self.scenario,
                candidate,
                practice_seeds[: max(1, int(limit))],
                split="llm-practice",
                steps=self.config.practice_steps,
            )
            context = ProposalContext(
                cycle=cycle,
                scenario=self.scenario,
                budget=self.config.budget,
                activity=self.activity,
                diagnosis=diagnosis,
                demo_features=self.demo_features,
                target_features=target_features,
                history=self.history,
                best_metrics=best_practice.metrics(),
                base_bundle=best,
                practice_callback=practice_callback,
                candidate_callback=candidate_callback,
                allow_live=self.config.allow_live,
                model=self._model_name(),
            )
            try:
                program = self.proposer.propose(context)
            except Exception as exc:
                transcript = getattr(self.proposer, "transcript", None)
                if transcript:
                    write_json(
                        self.run_dir / "cycles" / f"{cycle:04d}.transcript.json",
                        transcript,
                    )
                validation = ValidationReport(
                    ok=False, issues=[f"proposer failed: {type(exc).__name__}: {exc}"]
                )
                self._record_rejection(cycle, None, validation, note="proposer error", started=started_at)
                patience += 1
                if patience >= self.config.patience:
                    break
                continue
            transcript = getattr(self.proposer, "transcript", None)
            if transcript:
                write_json(self.run_dir / "cycles" / f"{cycle:04d}.transcript.json", transcript)
            if not program.ops:
                validation = ValidationReport(ok=False, issues=["empty program"])
                self._record_rejection(cycle, program, validation, note="empty program", started=started_at)
                patience += 1
                if patience >= self.config.patience:
                    break
                continue
            try:
                candidate, report = apply_program(
                    best,
                    program,
                    budget=self.config.budget,
                    spare_units=np.flatnonzero(self.activity.spare).astype(int).tolist(),
                    allow_live=self.config.allow_live,
                )
                report = validate_candidate(
                    best,
                    candidate,
                    program,
                    report,
                    demo_features=self.demo_features,
                    target_features=target_features,
                    budget=self.config.budget,
                    require_quiet=self.config.require_quiet,
                )
            except Exception as exc:
                validation = ValidationReport(ok=False, issues=[f"edit failed: {type(exc).__name__}: {exc}"])
                self._record_rejection(cycle, program, validation, note="edit error", started=started_at)
                patience += 1
                if patience >= self.config.patience:
                    break
                continue
            if not report.ok:
                self._record_rejection(cycle, program, report, note="validation failed", started=started_at)
                patience += 1
                if patience >= self.config.patience:
                    break
                continue

            practice = evaluate_bundle(
                self.scenario,
                candidate,
                practice_seeds,
                split="practice",
                steps=self.config.practice_steps,
                capture=True,
            )
            regression = evaluate_bundle(
                nominal,
                candidate,
                regression_seeds,
                split="regression",
                steps=self.config.regression_steps,
            )
            score = _score(practice)
            epsilon = float(self.config.budget.accept_epsilon)
            accepted = _accept_candidate(
                practice,
                regression,
                baseline_practice,
                baseline_regression,
                best_score,
                epsilon,
                float(self.config.budget.regression_margin),
            )
            prune_note = ""
            all_units = program_units(program)
            if (
                not accepted
                and self.config.budget.prune_units
                and len(all_units) > 1
            ):
                # The LLM often appends one untested unit to a validated pair.
                # Screen small unit subsets by quietness, then evaluate the
                # quietest one or two in closed loop.  Singles come first, which
                # is how a safe one-unit repair is recovered from a noisy pair.
                screened = []
                for subset in program_subsets(
                    program,
                    max_units=3,
                    max_candidates=max(4, int(self.config.budget.max_prunes) * 2),
                ):
                    try:
                        pcandidate, preport = apply_program(
                            best,
                            subset,
                            budget=self.config.budget,
                            spare_units=np.flatnonzero(self.activity.spare).astype(int).tolist(),
                            allow_live=self.config.allow_live,
                        )
                        preport = validate_candidate(
                            best,
                            pcandidate,
                            subset,
                            preport,
                            demo_features=self.demo_features,
                            target_features=target_features,
                            budget=self.config.budget,
                            require_quiet=self.config.require_quiet,
                        )
                    except Exception:
                        continue
                    if not preport.ok:
                        continue
                    delta = float(
                        (preport.quietness or {}).get("max_abs_delta", float("inf"))
                    )
                    screened.append((delta, subset, pcandidate, preport))
                screened.sort(key=lambda item: item[0])
                for _, pprogram, pcandidate, preport in screened[:2]:
                    ppractice = evaluate_bundle(
                        self.scenario,
                        pcandidate,
                        practice_seeds,
                        split="practice",
                        steps=self.config.practice_steps,
                        capture=True,
                    )
                    preg = evaluate_bundle(
                        nominal,
                        pcandidate,
                        regression_seeds,
                        split="regression",
                        steps=self.config.regression_steps,
                    )
                    pscore = _score(ppractice)
                    paccepted = _accept_candidate(
                        ppractice,
                        preg,
                        baseline_practice,
                        baseline_regression,
                        best_score,
                        epsilon,
                        float(self.config.budget.regression_margin),
                    )
                    if paccepted:
                        candidate, program, report = pcandidate, pprogram, preport
                        practice, regression, score, accepted = (
                            ppractice,
                            preg,
                            pscore,
                            True,
                        )
                        kept = set(program_units(pprogram))
                        removed = sorted(set(all_units) - kept)
                        report.pruned_units = [int(u) for u in removed]
                        prune_note = (
                            f"accepted subset units {sorted(kept)} "
                            f"(pruned {removed})"
                        )
                        break
            diff = diff_bundles(best, candidate)
            record = CycleRecord(
                cycle=cycle,
                scenario=self.scenario.name,
                proposer=getattr(self.proposer, "name", self.config.proposer),
                model=self._model_name(),
                program=program,
                validation=report,
                practice=practice.to_dict(),
                regression=regression.to_dict(),
                accepted=bool(accepted),
                score=float(score),
                budget={
                    **self.config.budget.to_dict(),
                    "rollouts": len(practice.episodes) + len(regression.episodes),
                    "wall_s": float(time.time() - cycle_started),
                    "tool_calls": getattr(getattr(self.proposer, "last_toolbox", None), "calls", 0),
                },
                started_at=started_at,
                finished_at=now_iso(),
                note=(
                    prune_note
                    if prune_note
                    else ("accepted" if accepted else "not accepted")
                ),
            )
            write_cycle(
                self.run_dir,
                record,
                candidate=candidate,
                diff=diff,
                practice=practice,
                regression=regression,
            )
            self.history.append(record.to_dict())
            if accepted:
                best = candidate
                best_practice = practice
                best_regression = regression
                best_score = score
                best_cycle = cycle
                patience = 0
                # Refresh the white-space map so the next cycle does not treat the
                # units we just programmed as unused.
                traces = [e.trace for e in practice.episodes if e.trace is not None]
                self.activity = audit_bundle(
                    best,
                    self.scenario,
                    seeds=practice_seeds[:1],
                    steps=min(220, int(self.config.practice_steps)),
                    max_episodes=1,
                    traces=traces,
                )
            else:
                patience += 1
            if patience >= self.config.patience:
                break

        unseen_seeds = _seeds(
            self.config.unseen_seed_base, self.config.unseen_episodes
        )
        final_practice = evaluate_bundle(
            self.scenario,
            best,
            unseen_seeds,
            split="unseen",
            steps=self.config.unseen_steps,
            capture=True,
        )
        baseline_unseen = evaluate_bundle(
            self.scenario,
            self.base_bundle,
            unseen_seeds,
            split="unseen",
            steps=self.config.unseen_steps,
        )
        final_regression = evaluate_bundle(
            nominal,
            best,
            _seeds(self.config.regression_seed_base + 50000, self.config.regression_episodes),
            split="unseen",
            steps=self.config.regression_steps,
        )
        repair_dir = Path(self.config.repair_bundle_dir)
        repair_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = repair_dir / f"{self.run_dir.name}.npz"
        save_repaired_bundle(
            bundle_path,
            best,
            repair={
                "run_id": self.run_dir.name,
                "base_fingerprint": self.base_bundle.get("fingerprint"),
                "scenario": self.scenario.name,
                "best_cycle": int(best_cycle),
                "cycles": len(self.history),
                "accepted_cycles": int(
                    sum(1 for h in self.history if h.get("accepted"))
                ),
                "resume_count": (
                    int(self.resume_info["round"]) if self.resume_info else 0
                ),
                "resumed_from": (
                    self.resume_info.get("bundle_path")
                    if self.resume_info
                    else None
                ),
                "start_cycle": int(start_cycle),
            },
        )
        report = self._report_text(
            baseline_practice,
            baseline_regression,
            baseline_unseen,
            final_practice,
            final_regression,
            str(bundle_path),
        )
        write_final(
            self.run_dir,
            scenario=self.scenario,
            baseline_unseen=baseline_unseen,
            final_practice=final_practice,
            final_regression=final_regression,
            bundle_path=str(bundle_path),
            report_md=report,
        )
        write_report(self.run_dir, report)
        self.manifest["finished_at"] = now_iso()
        self.manifest["final"] = final_practice.metrics()
        self.manifest["bundle_out"] = str(bundle_path)
        write_json(self.run_dir / "manifest.json", self.manifest)
        append_event(
            self.run_dir,
            {
                "type": "done",
                "success": final_practice.success_rate,
                "bundle": str(bundle_path),
                "timestamp": now_iso(),
            },
        )
        return {
            "run_id": self.run_dir.name,
            "run_dir": str(self.run_dir),
            "scenario": self.scenario.name,
            "baseline_practice": baseline_practice.metrics(),
            "baseline_unseen": baseline_unseen.metrics(),
            "baseline_regression": baseline_regression.metrics(),
            "final_practice": final_practice.metrics(),
            "final_regression": final_regression.metrics(),
            "bundle": str(bundle_path),
            "cycles": len(self.history),
            "accepted_cycles": int(sum(1 for h in self.history if h.get("accepted"))),
            "resume_count": int(self.resume_info["round"]) if self.resume_info else 0,
            "start_cycle": int(start_cycle),
            "wall_s": float(time.time() - started),
        }

    def _report_text(
        self,
        baseline_practice: SplitResult,
        baseline_regression: SplitResult,
        baseline_unseen: SplitResult,
        final_practice: SplitResult,
        final_regression: SplitResult,
        bundle_path: str,
    ) -> str:
        lines = [
            f"# White-space repair report: {self.run_dir.name}",
            "",
            f"- Scenario: `{self.scenario.name}` — {self.scenario.description}",
            f"- Proposer: `{self.config.proposer}`"
            + (f" ({self._model_name()})" if self._model_name() else ""),
            f"- Base bundle: `{self.config.bundle_path}`",
            f"- Repaired bundle: `{bundle_path}`",
            f"- Cycles: {len(self.history)}; accepted: "
            f"{sum(1 for h in self.history if h.get('accepted'))}",
            *(
                [
                    f"- Resumed round: {int(self.resume_info['round'])} from "
                    f"`{self.resume_info.get('bundle_path')}`; new cycles start at "
                    f"{int(self.resume_info['start_cycle'])}"
                ]
                if self.resume_info
                else []
            ),
            "",
            "## Evaluation",
            "",
            "| split | strict | near-goal | reached | collision-free | progress | mean final dist |",
            "|---|---|---|---|---|---|---|",
            f"| baseline practice | {baseline_practice.success_rate:.3f} | "
            f"{baseline_practice.near_success_rate:.3f} | "
            f"{baseline_practice.reached_rate:.3f} | "
            f"{baseline_practice.collision_free_rate:.3f} | "
            f"{baseline_practice.mean_final_progress:.3f} | "
            f"{baseline_practice.mean_final_goal_dist_m:.3f} |",
            f"| baseline unseen | {baseline_unseen.success_rate:.3f} | "
            f"{baseline_unseen.near_success_rate:.3f} | "
            f"{baseline_unseen.reached_rate:.3f} | "
            f"{baseline_unseen.collision_free_rate:.3f} | "
            f"{baseline_unseen.mean_final_progress:.3f} | "
            f"{baseline_unseen.mean_final_goal_dist_m:.3f} |",
            f"| final unseen | {final_practice.success_rate:.3f} | "
            f"{final_practice.near_success_rate:.3f} | "
            f"{final_practice.reached_rate:.3f} | "
            f"{final_practice.collision_free_rate:.3f} | "
            f"{final_practice.mean_final_progress:.3f} | "
            f"{final_practice.mean_final_goal_dist_m:.3f} |",
            "",
            "## Nominal regression",
            "",
            f"- baseline progress: {baseline_regression.mean_final_progress:.3f}",
            f"- final progress: {final_regression.mean_final_progress:.3f}",
            f"- baseline collision-free: {baseline_regression.collision_free_rate:.3f}",
            f"- final collision-free: {final_regression.collision_free_rate:.3f}",
            "",
            "## Cycles",
            "",
            "| cycle | accepted | strict | near | progress | edit |",
            "|---|---|---|---|---|---|",
        ]
        for item in self.history:
            program = item.get("program") or {}
            rationale = str(program.get("rationale", "")).replace("|", "/")
            metrics = (item.get("practice") or {}).get("metrics", {})
            success = metrics.get("success_rate")
            near = metrics.get("near_success_rate")
            progress = metrics.get("mean_final_progress")
            def _num(value):
                return "—" if value is None else f"{float(value):.3f}"
            lines.append(
                f"| {item.get('cycle')} | {item.get('accepted')} | "
                f"{_num(success)} | {_num(near)} | {_num(progress)} | "
                f"{rationale[:140]} |"
            )
        lines.extend(
            [
                "",
                "## Failure taxonomy (final practice)",
                "",
                "```json",
                __import__("json").dumps(final_practice.failure_counts(), indent=2),
                "```",
                "",
                "Gradient-free by construction: no optimizer, no new demonstrations, "
                "and every edit is expressed through the white-space DSL.",
                "",
            ]
        )
        return "\n".join(lines)


def run_repair(config: Optional[RepairRunConfig] = None, **kwargs) -> Dict[str, Any]:
    config = config or RepairRunConfig(**kwargs)
    return RepairLoop(config).run()
