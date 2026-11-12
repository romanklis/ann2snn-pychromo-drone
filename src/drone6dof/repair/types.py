"""Plain data types shared by the repair loop, tools and inspector."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass
class Budget:
    """Budgets for one repair run and one cycle.

    Defaults are deliberately small; the CLI can raise them after a throughput
    probe.  ``max_practice_rollouts`` is the per-cycle simulator budget handed to
    the proposer.
    """

    max_cycles: int = 12
    max_turns: int = 30
    max_tool_calls: int = 120
    max_practice_rollouts: int = 18
    max_units_per_cycle: int = 24
    max_new_edges_per_cycle: int = 256
    max_param_change_per_cycle: int = 8192
    quiet_eps: float = 1e-3
    min_rate_hz: float = 0.5
    #: Deprecated: acceptance is monotonic on mean final progress.
    accept_margin: float = 0.02
    accept_epsilon: float = 1e-6
    regression_margin: float = 0.02
    strict_quiet: bool = False
    prune_units: bool = True
    max_prunes: int = 4
    max_practice_per_call: int = 6
    cycle_seconds: float = 60.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EditOp:
    """One white-space edit operation.

    Only the fields relevant to ``kind`` are used; the rest stay ``None``.  The
    DSL is intentionally small so every proposal can be validated before it is
    applied.
    """

    kind: str
    unit: Optional[int] = None
    values: Optional[List[float]] = None
    value: Optional[float] = None
    src: Optional[int] = None
    dst: Optional[int] = None
    magnitude: Optional[float] = None
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EditOp":
        if not isinstance(data, dict):
            raise ValueError("edit op must be an object")
        data = dict(data)
        if "kind" not in data:
            for alias in ("type", "op", "action"):
                if alias in data:
                    data["kind"] = data.pop(alias)
                    break
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown edit-op fields: {sorted(unknown)}")
        values = data.get("values")
        if values is not None:
            try:
                values = [float(v) for v in values]
            except (TypeError, ValueError) as exc:
                raise ValueError("edit-op 'values' must be a list of numbers") from exc
        return cls(
            kind=str(data.get("kind", "")),
            unit=None if data.get("unit") is None else int(data["unit"]),
            values=values,
            value=None if data.get("value") is None else float(data["value"]),
            src=None if data.get("src") is None else int(data["src"]),
            dst=None if data.get("dst") is None else int(data["dst"]),
            magnitude=(
                None if data.get("magnitude") is None else float(data["magnitude"])
            ),
            rationale=str(data.get("rationale", "")),
        )


@dataclass
class EditProgram:
    program_id: str
    ops: List[EditOp] = field(default_factory=list)
    rationale: str = ""
    proposer: str = "template"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "program_id": self.program_id,
            "ops": [op.to_dict() for op in self.ops],
            "rationale": self.rationale,
            "proposer": self.proposer,
            "metadata": _jsonable(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EditProgram":
        if not isinstance(data, dict):
            raise ValueError("program must be an object")
        ops = data.get("ops", [])
        if not isinstance(ops, list):
            raise ValueError("program 'ops' must be a list")
        return cls(
            program_id=str(data.get("program_id", "program")),
            ops=[EditOp.from_dict(op) for op in ops],
            rationale=str(data.get("rationale", "")),
            proposer=str(data.get("proposer", "external")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ValidationReport:
    ok: bool = True
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    quietness: Dict[str, float] = field(default_factory=dict)
    liveness: Dict[str, float] = field(default_factory=dict)
    changed_params: int = 0
    edited_units: List[int] = field(default_factory=list)
    new_edges: int = 0
    auto_isolated_units: List[int] = field(default_factory=list)
    pruned_units: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeResult:
    episode_id: int
    seed: int
    scenario: str
    success: bool
    reached_goal: bool
    collision_free: bool
    final_goal_dist_m: float
    closest_goal_dist_m: float
    clearance_min_m: float
    collisions: int
    steps: int
    time_to_goal_s: Optional[float]
    progress: float
    final_progress: float = 0.0
    command_saturation_frac: float = 0.0
    mean_power_w: float = 0.0
    near_success: bool = False
    failure: str = "none"
    trace: Optional[Dict[str, Any]] = None

    def summary(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("trace", None)
        return _jsonable(data)

    def to_dict(self, *, include_trace: bool = False) -> Dict[str, Any]:
        data = self.summary()
        if include_trace and self.trace is not None:
            data["trace"] = _jsonable(self.trace)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeResult":
        allowed = {f.name for f in fields(cls)}
        values = {
            key: value
            for key, value in dict(data).items()
            if key in allowed and key != "trace"
        }
        values["trace"] = None
        return cls(**values)


@dataclass
class SplitResult:
    scenario: str
    split: str
    steps: int
    episodes: List[EpisodeResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.episodes)

    @property
    def success_rate(self) -> float:
        return (
            sum(1 for e in self.episodes if e.success) / self.n if self.n else 0.0
        )

    @property
    def near_success_rate(self) -> float:
        return (
            sum(1 for e in self.episodes if e.near_success) / self.n
            if self.n
            else 0.0
        )

    @property
    def collision_free_rate(self) -> float:
        return (
            sum(1 for e in self.episodes if e.collision_free) / self.n
            if self.n
            else 0.0
        )

    @property
    def reached_rate(self) -> float:
        return (
            sum(1 for e in self.episodes if e.reached_goal) / self.n
            if self.n
            else 0.0
        )

    @property
    def mean_final_goal_dist_m(self) -> float:
        return (
            sum(e.final_goal_dist_m for e in self.episodes) / self.n
            if self.n
            else float("nan")
        )

    @property
    def mean_clearance_min_m(self) -> float:
        return (
            sum(e.clearance_min_m for e in self.episodes) / self.n
            if self.n
            else float("nan")
        )

    @property
    def mean_final_progress(self) -> float:
        return (
            sum(e.final_progress for e in self.episodes) / self.n
            if self.n
            else float("nan")
        )

    @property
    def mean_power_w(self) -> float:
        return (
            sum(e.mean_power_w for e in self.episodes) / self.n
            if self.n
            else float("nan")
        )

    def failure_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for episode in self.episodes:
            counts[episode.failure] = counts.get(episode.failure, 0) + 1
        return dict(sorted(counts.items()))

    def metrics(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "split": self.split,
            "steps": self.steps,
            "episodes": self.n,
            "success_rate": self.success_rate,
            "near_success_rate": self.near_success_rate,
            "reached_rate": self.reached_rate,
            "collision_free_rate": self.collision_free_rate,
            "mean_final_goal_dist_m": self.mean_final_goal_dist_m,
            "mean_clearance_min_m": self.mean_clearance_min_m,
            "mean_final_progress": self.mean_final_progress,
            "mean_power_w": self.mean_power_w,
            "failure_counts": self.failure_counts(),
        }

    def to_dict(self, *, include_traces: bool = False) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "split": self.split,
            "steps": self.steps,
            "metrics": self.metrics(),
            "episodes": [
                e.to_dict(include_trace=include_traces) for e in self.episodes
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SplitResult":
        episodes = [
            EpisodeResult.from_dict(item)
            for item in (data.get("episodes") or [])
            if isinstance(item, dict)
        ]
        return cls(
            scenario=str(data.get("scenario", "unknown")),
            split=str(data.get("split", "unknown")),
            steps=int(data.get("steps", 0)),
            episodes=episodes,
        )


@dataclass
class Diagnosis:
    scenario: str
    split_metrics: Dict[str, Any]
    failure_counts: Dict[str, int]
    activity: Dict[str, Any]
    examples: List[Dict[str, Any]] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass
class CycleRecord:
    cycle: int
    scenario: str
    proposer: str
    model: Optional[str]
    program: Optional[EditProgram]
    validation: ValidationReport
    practice: Optional[Dict[str, Any]]
    regression: Optional[Dict[str, Any]]
    accepted: bool
    score: float
    budget: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(
            {
                "cycle": self.cycle,
                "scenario": self.scenario,
                "proposer": self.proposer,
                "model": self.model,
                "program": None if self.program is None else self.program.to_dict(),
                "validation": self.validation.to_dict(),
                "practice": self.practice,
                "regression": self.regression,
                "accepted": self.accepted,
                "score": self.score,
                "budget": self.budget,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "note": self.note,
            }
        )
