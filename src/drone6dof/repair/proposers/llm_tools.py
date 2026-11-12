"""Budgeted tools exposed to the LLM proposer.

The LLM never receives the simulator.  It can inspect the network, probe spare
units on synthetic feature vectors, and request practice rollouts through the
harness; every call is counted against the run budget.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..activity import ACTION_NAMES, FEATURE_NAMES
from ..edits import apply_program, clone_bundle, validate_candidate
from ..types import Budget, EditOp, EditProgram, SplitResult
from ...connectome import RateCodedConnectomeSNN, SparseRecurrence

__all__ = ["TOOL_SCHEMA", "RepairToolbox"]


TOOL_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_features",
            "description": "List the 46 policy input features and the 3 action dimensions.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_activity",
            "description": "Return the white-space audit summary (silent/rare/spare units).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_spare_units",
            "description": "List the top spare neurons with their input/output norms.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_weights",
            "description": "Read w_in row, w_out column or bias for one unit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "unit": {"type": "integer"},
                    "what": {"type": "string", "enum": ["w_in", "w_out", "bias"]},
                },
                "required": ["unit", "what"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "probe_unit",
            "description": (
                "Measure a unit's spike rate while sweeping one feature, optionally "
                "with a candidate w_in row and bias."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "unit": {"type": "integer"},
                    "feature_index": {"type": "integer"},
                    "values": {"type": "array", "items": {"type": "number"}},
                    "weights": {"type": "array", "items": {"type": "number"}},
                    "bias": {"type": "number"},
                },
                "required": ["unit", "feature_index", "values"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_program",
            "description": (
                "Apply a candidate program and return hard issues, warnings, "
                "quietness and liveness without spending practice rollouts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ops": {"type": "array", "items": {"type": "object"}},
                    "rationale": {"type": "string"},
                },
                "required": ["ops"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_practice",
            "description": "Run a bounded batch of practice episodes and return success metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 24},
                    "ops": {"type": "array", "items": {"type": "object"}},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_program",
            "description": "Submit the final edit program as a list of DSL operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ops": {"type": "array", "items": {"type": "object"}},
                    "rationale": {"type": "string"},
                },
                "required": ["ops", "rationale"],
            },
        },
    },
]


def _rec(bundle: Dict[str, Any]) -> SparseRecurrence:
    edges = np.asarray(bundle["edges"], dtype=np.int64)
    polarity = np.asarray(bundle["polarity"], dtype=np.float64)
    signed = np.abs(np.asarray(bundle["w_mag"], dtype=np.float64)) * polarity[edges[1]]
    return SparseRecurrence(edges, signed, int(bundle["w_out"].shape[1]))


class RepairToolbox:
    def __init__(
        self,
        base_bundle: Dict[str, Any],
        activity,
        *,
        budget: Budget,
        demo_features: Optional[np.ndarray] = None,
        target_features: Optional[np.ndarray] = None,
        practice_callback: Optional[Callable[[int], SplitResult]] = None,
        candidate_callback: Optional[Callable[[Dict[str, Any], int], SplitResult]] = None,
        spare_units: Optional[List[int]] = None,
        allow_live: bool = False,
    ) -> None:
        self.base_bundle = base_bundle
        self.activity = activity
        self.budget = budget
        self.demo_features = demo_features
        self.target_features = target_features
        self.practice_callback = practice_callback
        self.candidate_callback = candidate_callback
        self.spare_units = spare_units
        self.allow_live = bool(allow_live)
        self.calls = 0
        self.practice_runs = 0
        self.submitted: Optional[EditProgram] = None
        self.last_validated: Optional[EditProgram] = None
        self.last_validated_quietness = float("inf")

    def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self.calls += 1
        if self.calls > int(self.budget.max_tool_calls):
            return {"ok": False, "error": "tool-call budget exhausted"}
        args = arguments or {}
        try:
            if name == "list_features":
                return {"ok": True, "features": FEATURE_NAMES, "actions": ACTION_NAMES}
            if name == "get_activity":
                return {"ok": True, "activity": self.activity.summary}
            if name == "list_spare_units":
                limit = int(args.get("limit", 20))
                return {
                    "ok": True,
                    "units": self.activity.summary.get("top_spare", [])[:limit],
                }
            if name == "read_weights":
                unit = int(args["unit"])
                what = str(args.get("what", "w_in"))
                if what == "w_in":
                    values = np.asarray(self.base_bundle["w_in"])[unit]
                elif what == "w_out":
                    values = np.asarray(self.base_bundle["w_out"])[:, unit]
                elif what == "bias":
                    bias = self.base_bundle.get("bias")
                    values = np.zeros(int(self.base_bundle["w_out"].shape[1])) if bias is None else np.asarray(bias)
                    values = values[unit:unit + 1]
                else:
                    return {"ok": False, "error": f"unknown weights {what!r}"}
                return {"ok": True, "unit": unit, "what": what, "values": np.asarray(values).tolist()}
            if name == "probe_unit":
                return self._probe_unit(args)
            if name == "validate_program":
                ops = args.get("ops")
                if not isinstance(ops, list):
                    return {"ok": False, "error": "'ops' must be a list"}
                program = EditProgram(
                    program_id="llm-validate",
                    ops=[EditOp.from_dict(op) for op in ops],
                    rationale=str(args.get("rationale", "")),
                    proposer="llm",
                )
                candidate, report = apply_program(
                    self.base_bundle,
                    program,
                    budget=self.budget,
                    spare_units=self.spare_units,
                    allow_live=self.allow_live,
                )
                report = validate_candidate(
                    self.base_bundle,
                    candidate,
                    program,
                    report,
                    demo_features=self.demo_features,
                    target_features=self.target_features,
                    budget=self.budget,
                    require_quiet=False,
                )
                quiet = (report.quietness or {}).get("max_abs_delta", 0.0)
                if report.ok and float(quiet) <= self.last_validated_quietness:
                    self.last_validated = program
                    self.last_validated_quietness = float(quiet)
                return {
                    "ok": bool(report.ok),
                    "issues": report.issues,
                    "warnings": report.warnings,
                    "quietness": report.quietness,
                    "quiet_ok": float(quiet) <= float(self.budget.quiet_eps),
                    "liveness": report.liveness,
                    "auto_isolated_units": report.auto_isolated_units,
                    "changed_params": report.changed_params,
                }
            if name == "run_practice":
                limit = int(args.get("limit", 4))
                if self.practice_runs >= int(self.budget.max_practice_rollouts):
                    return {"ok": False, "error": "practice-rollout budget exhausted"}
                remaining = max(1, int(self.budget.max_practice_rollouts) - self.practice_runs)
                limit = min(
                    limit,
                    remaining,
                    max(1, int(self.budget.max_practice_per_call)),
                )
                ops = args.get("ops")
                if ops is not None:
                    if not isinstance(ops, list):
                        return {"ok": False, "error": "'ops' must be a list"}
                    program = EditProgram(
                        program_id="llm-practice",
                        ops=[EditOp.from_dict(op) for op in ops],
                        rationale=str(args.get("rationale", "")),
                        proposer="llm",
                    )
                    candidate, report = apply_program(
                        self.base_bundle,
                        program,
                        budget=self.budget,
                        spare_units=self.spare_units,
                        allow_live=self.allow_live,
                    )
                    if not report.ok:
                        return {"ok": False, "errors": report.issues}
                    if self.candidate_callback is None:
                        return {"ok": False, "error": "candidate rollouts are unavailable"}
                    result = self.candidate_callback(candidate, limit)
                else:
                    if self.practice_callback is None:
                        return {"ok": False, "error": "practice rollouts are unavailable"}
                    result = self.practice_callback(limit)
                self.practice_runs += limit
                return {"ok": True, "metrics": result.metrics()}
            if name == "submit_program":
                ops = args.get("ops", [])
                if not isinstance(ops, list):
                    return {"ok": False, "error": "'ops' must be a list"}
                program = EditProgram(
                    program_id="llm-submitted",
                    ops=[EditOp.from_dict(op) for op in ops],
                    rationale=str(args.get("rationale", "")),
                    proposer="llm",
                )
                self.submitted = program
                return {"ok": True, "program": program.to_dict()}
            return {"ok": False, "error": f"unknown tool {name!r}"}
        except Exception as exc:  # pragma: no cover - defensive
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _probe_unit(self, args: Dict[str, Any]) -> Dict[str, Any]:
        unit = int(args["unit"])
        feature_index = int(args["feature_index"])
        values = [float(v) for v in args.get("values", [])]
        if not values:
            return {"ok": False, "error": "values must not be empty"}
        n_in = int(self.base_bundle["w_in"].shape[1])
        if not (0 <= feature_index < n_in):
            return {"ok": False, "error": f"feature_index outside [0, {n_in})"}
        candidate = clone_bundle(self.base_bundle)
        weights = args.get("weights")
        if weights is not None:
            row = np.asarray(weights, dtype=np.float64).reshape(-1)
            if row.size != n_in:
                return {"ok": False, "error": f"weights must have {n_in} entries"}
            candidate["w_in"][unit, :] = row
        if args.get("bias") is not None:
            n = int(candidate["w_out"].shape[1])
            if candidate.get("bias") is None:
                candidate["bias"] = np.zeros(n, dtype=np.float64)
            candidate["bias"][unit] = float(args["bias"])
        rows = np.zeros((len(values), n_in), dtype=np.float64)
        rows[:, feature_index] = np.asarray(values, dtype=np.float64)
        net = RateCodedConnectomeSNN(
            candidate["w_in"], candidate["w_out"], _rec(candidate),
            micro_steps=int(candidate.get("micro_steps", 10)),
            v_th=float(candidate.get("v_th", 1.0)),
            limit=12.0,
            readout_gain=float(candidate.get("readout_gain", 1.0)),
            bias=candidate.get("bias"),
        )
        rates = []
        for row in rows:
            net.reset()
            net.forward_input(row)
            rates.append(
                float(net.spike_frames[unit]) / max(1, net.frames) / 0.02
            )
        return {
            "ok": True,
            "unit": unit,
            "feature_index": feature_index,
            "feature_name": FEATURE_NAMES[feature_index] if feature_index < len(FEATURE_NAMES) else f"feature_{feature_index}",
            "values": values,
            "rate_hz": rates,
            "frames": int(net.frames),
        }
