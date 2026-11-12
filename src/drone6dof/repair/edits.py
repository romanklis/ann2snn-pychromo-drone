"""Structured white-space edit DSL: apply, validate, diff and save."""

from __future__ import annotations

import copy
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..connectome import ConnectomeANN, RateCodedConnectomeSNN, SparseRecurrence
from ..weights import DEFAULT_REF_IO_PATH, save_weights
from .types import Budget, EditOp, EditProgram, ValidationReport

__all__ = [
    "EditError",
    "clone_bundle",
    "load_demo_features",
    "forward_actions",
    "apply_program",
    "validate_candidate",
    "diff_bundles",
    "save_repaired_bundle",
    "program_units",
    "program_subsets",
    "prune_program_units",
]


class EditError(ValueError):
    """An edit is malformed or violates a guard rail."""


def clone_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, value in bundle.items():
        if isinstance(value, np.ndarray):
            out[key] = np.array(value, dtype=value.dtype, copy=True)
        elif isinstance(value, dict):
            out[key] = copy.deepcopy(value)
        elif isinstance(value, list):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def _recurrence(bundle: Dict[str, Any]) -> SparseRecurrence:
    edges = np.asarray(bundle["edges"], dtype=np.int64)
    polarity = np.asarray(bundle["polarity"], dtype=np.float64)
    signed = np.abs(np.asarray(bundle["w_mag"], dtype=np.float64)) * polarity[edges[1]]
    return SparseRecurrence(edges, signed, int(bundle["w_out"].shape[1]))


def forward_actions(
    bundle: Dict[str, Any],
    features: np.ndarray,
    *,
    kind: str = "snn",
) -> np.ndarray:
    """Run the net on a feature matrix without a simulator.

    The recurrent state is reset before the batch and carried across rows, which
    matches deployment.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("features must be a 2-D array")
    rec = _recurrence(bundle)
    if str(kind).lower() == "ann":
        net = ConnectomeANN(
            bundle["w_in"],
            bundle["w_out"],
            rec,
            steps_per_frame=int(bundle.get("connectome_steps", 3)),
            limit=12.0,
        )
    else:
        net = RateCodedConnectomeSNN(
            bundle["w_in"],
            bundle["w_out"],
            rec,
            micro_steps=int(bundle.get("micro_steps", 10)),
            v_th=float(bundle.get("v_th", 1.0)),
            limit=12.0,
            readout_gain=float(bundle.get("readout_gain", 1.0)),
            bias=bundle.get("bias"),
        )
    net.reset()
    return np.asarray([net.forward_input(row) for row in x], dtype=np.float64)


def load_demo_features(path=None) -> np.ndarray:
    """Load the fixed in-distribution demonstration inputs."""
    path = Path(path or DEFAULT_REF_IO_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"no demo feature file at {path}; run `make train` or pass --demos"
        )
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["inputs"], dtype=np.float64)


def program_units(program: EditProgram) -> List[int]:
    units: Set[int] = set()
    for op in program.ops:
        for value in (op.unit, op.src, op.dst):
            if value is not None:
                units.add(int(value))
    return sorted(units)


def program_subsets(
    program: EditProgram,
    *,
    max_units: int = 3,
    max_candidates: int = 8,
) -> List[EditProgram]:
    """Generate small unit subsets for pruning/screening.

    Singles come first, so a safe one-unit repair is preferred over a larger
    program with one contaminating unit.
    """
    units = program_units(program)
    if len(units) < 2:
        return []
    subsets: List[EditProgram] = []
    for size in range(1, min(int(max_units), len(units)) + 1):
        for combo in combinations(units, size):
            keep = set(combo)
            ops = [
                op
                for op in program.ops
                if all(
                    value is None or int(value) in keep
                    for value in (op.unit, op.src, op.dst)
                )
            ]
            if not ops or len(ops) == len(program.ops):
                continue
            metadata = dict(program.metadata)
            metadata["pruned_from"] = program.program_id
            metadata["subset_units"] = sorted(keep)
            subsets.append(
                EditProgram(
                    program_id=f"{program.program_id}-subset-{'-'.join(map(str, combo))}",
                    ops=ops,
                    rationale=program.rationale,
                    proposer=program.proposer,
                    metadata=metadata,
                )
            )
            if len(subsets) >= int(max_candidates):
                return subsets
    return subsets


def prune_program_units(program: EditProgram, units: Iterable[int]) -> EditProgram:
    """Return the program with every op touching ``units`` removed."""
    drop = {int(u) for u in units}
    ops = []
    for op in program.ops:
        touched = {
            value
            for value in (op.unit, op.src, op.dst)
            if value is not None
        }
        if touched & drop:
            continue
        ops.append(op)
    metadata = dict(program.metadata)
    metadata["pruned_from"] = program.program_id
    metadata["pruned_units"] = sorted(drop)
    return EditProgram(
        program_id=f"{program.program_id}-pruned",
        ops=ops,
        rationale=program.rationale,
        proposer=program.proposer,
        metadata=metadata,
    )


def _changed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        return np.ones_like(b, dtype=bool)
    return ~np.isclose(a, b, rtol=0.0, atol=1e-12)


def _changed_count(a: np.ndarray, b: np.ndarray) -> int:
    """Count changed entries, treating appended edges as new entries only."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape == b.shape:
        return int(np.sum(_changed(a, b)))
    if a.ndim == 2 and b.ndim == 2 and a.shape[0] == b.shape[0]:
        common = min(a.shape[1], b.shape[1])
        count = int(np.sum(_changed(a[:, :common], b[:, :common])))
        count += abs(int(a.shape[1]) - int(b.shape[1])) * int(a.shape[0])
        return count
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    common = min(flat_a.size, flat_b.size)
    count = int(np.sum(_changed(flat_a[:common], flat_b[:common])))
    count += abs(int(flat_a.size) - int(flat_b.size))
    return count


def apply_program(
    bundle: Dict[str, Any],
    program: EditProgram,
    *,
    budget: Optional[Budget] = None,
    spare_units: Optional[Iterable[int]] = None,
    allow_live: bool = False,
    max_abs_w_in: float = 10.0,
    max_abs_w_out: float = 10.0,
    max_abs_bias: float = 10.0,
) -> Tuple[Dict[str, Any], ValidationReport]:
    """Apply a program to a copy of ``bundle`` and return the candidate + report."""
    budget = budget or Budget()
    candidate = clone_bundle(bundle)
    n_neurons = int(candidate["w_out"].shape[1])
    n_in = int(candidate["w_in"].shape[1])
    n_out = int(candidate["w_out"].shape[0])
    spare = None if spare_units is None else {int(u) for u in spare_units}
    issues: List[str] = []
    warnings: List[str] = []
    edited_units: Set[int] = set()
    new_edges = 0

    if not isinstance(program, EditProgram):
        raise EditError("program must be an EditProgram")
    if not program.ops:
        warnings.append("program has no operations")

    # A repurposed unit must not leak its old recurrent program.  Isolate every
    # unit whose input/readout/bias is rewritten, then apply explicit edge ops
    # afterwards so a proposer can still add a deliberate relay.
    repurpose_kinds = {
        "set_input_row",
        "add_input_row",
        "set_output_col",
        "add_output_col",
        "set_bias",
        "add_bias",
        "silence_output",
    }
    auto_isolate = {
        int(op.unit)
        for op in program.ops
        if op.kind in repurpose_kinds and op.unit is not None
    }
    if auto_isolate:
        edges = np.asarray(candidate["edges"], dtype=np.int32)
        mask = np.isin(edges[0], list(auto_isolate)) | np.isin(
            edges[1], list(auto_isolate)
        )
        candidate["w_mag"][mask] = 0.0

    def require_unit(op: EditOp) -> int:
        if op.unit is None:
            raise EditError(f"{op.kind} requires 'unit'")
        unit = int(op.unit)
        if not (0 <= unit < n_neurons):
            raise EditError(f"unit {unit} outside [0, {n_neurons})")
        edited_units.add(unit)
        if spare is not None and unit not in spare and not allow_live:
            raise EditError(
                f"unit {unit} is not in the spare set; pass allow_live to override"
            )
        return unit

    for op in program.ops:
        kind = str(op.kind)
        if kind in ("set_input_row", "add_input_row"):
            unit = require_unit(op)
            if op.values is None:
                raise EditError(f"{kind} requires 'values'")
            values = np.asarray(op.values, dtype=np.float64).reshape(-1)
            if values.size != n_in:
                raise EditError(
                    f"{kind} for unit {unit} needs {n_in} values, got {values.size}"
                )
            if not np.all(np.isfinite(values)):
                raise EditError(f"{kind} contains non-finite values")
            if float(np.max(np.abs(values), initial=0.0)) > float(max_abs_w_in):
                raise EditError(f"{kind} exceeds max |w_in| = {max_abs_w_in}")
            if kind == "set_input_row":
                candidate["w_in"][unit, :] = values
            else:
                candidate["w_in"][unit, :] += values
        elif kind in ("set_output_col", "add_output_col"):
            unit = require_unit(op)
            if op.values is None:
                raise EditError(f"{kind} requires 'values'")
            values = np.asarray(op.values, dtype=np.float64).reshape(-1)
            if values.size != n_out:
                raise EditError(
                    f"{kind} for unit {unit} needs {n_out} values, got {values.size}"
                )
            if not np.all(np.isfinite(values)):
                raise EditError(f"{kind} contains non-finite values")
            if float(np.max(np.abs(values), initial=0.0)) > float(max_abs_w_out):
                raise EditError(f"{kind} exceeds max |w_out| = {max_abs_w_out}")
            if kind == "set_output_col":
                candidate["w_out"][:, unit] = values
            else:
                candidate["w_out"][:, unit] += values
        elif kind in ("set_bias", "add_bias"):
            unit = require_unit(op)
            if op.value is None:
                raise EditError(f"{kind} requires 'value'")
            if not np.isfinite(op.value):
                raise EditError(f"{kind} contains a non-finite value")
            if abs(float(op.value)) > float(max_abs_bias):
                raise EditError(f"{kind} exceeds max |bias| = {max_abs_bias}")
            if candidate.get("bias") is None:
                candidate["bias"] = np.zeros(n_neurons, dtype=np.float64)
            if kind == "set_bias":
                candidate["bias"][unit] = float(op.value)
            else:
                candidate["bias"][unit] += float(op.value)
        elif kind == "silence_output":
            unit = require_unit(op)
            candidate["w_out"][:, unit] = 0.0
        elif kind == "isolate_unit":
            unit = require_unit(op)
            # Zero all recurrent synapses touching the spare unit.  A silent unit
            # normally contributes nothing, so this is free; once it becomes a
            # program unit, it must communicate only through its dense w_out column
            # and any edges the program explicitly re-adds.
            edges = np.asarray(candidate["edges"], dtype=np.int32)
            mask = (edges[0] == unit) | (edges[1] == unit)
            candidate["w_mag"][mask] = 0.0
        elif kind == "add_edge":
            if op.src is None or op.dst is None or op.magnitude is None:
                raise EditError("add_edge requires 'src', 'dst' and 'magnitude'")
            src, dst = int(op.src), int(op.dst)
            for name, value in (("src", src), ("dst", dst)):
                if not (0 <= value < n_neurons):
                    raise EditError(f"edge {name} {value} outside [0, {n_neurons})")
            if spare is not None and dst not in spare and not allow_live:
                raise EditError(
                    f"edge destination {dst} is not spare; pass allow_live to override"
                )
            mag = abs(float(op.magnitude))
            if not np.isfinite(mag) or mag <= 0.0:
                raise EditError("edge magnitude must be finite and positive")
            edges = np.asarray(candidate["edges"], dtype=np.int32)
            candidate["edges"] = np.concatenate(
                [edges, np.array([[dst], [src]], dtype=np.int32)], axis=1
            )
            candidate["w_mag"] = np.concatenate(
                [np.asarray(candidate["w_mag"], dtype=np.float64), np.array([mag])]
            )
            edited_units.add(dst)
            new_edges += 1
        elif kind == "set_edge":
            if op.src is None or op.dst is None or op.magnitude is None:
                raise EditError("set_edge requires 'src', 'dst' and 'magnitude'")
            src, dst = int(op.src), int(op.dst)
            edges = np.asarray(candidate["edges"], dtype=np.int32)
            matches = np.flatnonzero((edges[0] == dst) & (edges[1] == src))
            if len(matches) == 0:
                raise EditError(f"no existing edge {src}->{dst} to set")
            mag = abs(float(op.magnitude))
            if not np.isfinite(mag):
                raise EditError("edge magnitude must be finite")
            candidate["w_mag"][int(matches[0])] = mag
            edited_units.add(dst)
        else:
            raise EditError(f"unknown edit kind {kind!r}")

    if len(edited_units) > int(budget.max_units_per_cycle):
        issues.append(
            f"{len(edited_units)} units edited; budget is {budget.max_units_per_cycle}"
        )
    if new_edges > int(budget.max_new_edges_per_cycle):
        issues.append(
            f"{new_edges} new edges; budget is {budget.max_new_edges_per_cycle}"
        )

    changed = 0
    for key in ("w_in", "w_out", "w_mag", "edges", "bias"):
        base_val = bundle.get(key)
        cand_val = candidate.get(key)
        if base_val is None and cand_val is None:
            continue
        if base_val is None:
            changed += int(np.asarray(cand_val).size)
        elif cand_val is None:
            changed += int(np.asarray(base_val).size)
        else:
            changed += _changed_count(np.asarray(base_val), np.asarray(cand_val))
    if changed > int(budget.max_param_change_per_cycle):
        issues.append(
            f"{changed} parameters changed; budget is {budget.max_param_change_per_cycle}"
        )

    if np.any(np.asarray(candidate["w_mag"], dtype=np.float64) < 0.0):
        issues.append("w_mag must be non-negative")
    if not np.array_equal(
        np.asarray(candidate["polarity"]), np.asarray(bundle["polarity"])
    ):
        issues.append("polarity is immutable (Dale's law)")

    report = ValidationReport(
        ok=not issues,
        issues=issues,
        warnings=warnings,
        changed_params=int(changed),
        edited_units=sorted(edited_units),
        new_edges=int(new_edges),
        auto_isolated_units=sorted(auto_isolate),
    )
    return candidate, report


def validate_candidate(
    base_bundle: Dict[str, Any],
    candidate: Dict[str, Any],
    program: EditProgram,
    report: ValidationReport,
    *,
    demo_features: Optional[np.ndarray] = None,
    target_features: Optional[np.ndarray] = None,
    budget: Optional[Budget] = None,
    kind: str = "snn",
    require_quiet: bool = False,
) -> ValidationReport:
    """Add quietness, liveness and shape checks to an apply report."""
    budget = budget or Budget()
    issues = list(report.issues)
    warnings = list(report.warnings)

    if candidate["w_in"].shape != base_bundle["w_in"].shape:
        issues.append("w_in shape changed")
    if candidate["w_out"].shape != base_bundle["w_out"].shape:
        issues.append("w_out shape changed")

    quietness: Dict[str, float] = {}
    if demo_features is not None and len(demo_features):
        try:
            base_actions = forward_actions(base_bundle, demo_features, kind=kind)
            cand_actions = forward_actions(candidate, demo_features, kind=kind)
            delta = np.abs(cand_actions - base_actions)
            quietness = {
                "max_abs_delta": float(np.max(delta)),
                "mean_abs_delta": float(np.mean(delta)),
                "frames": int(len(demo_features)),
            }
            if quietness["max_abs_delta"] > float(budget.quiet_eps):
                message = (
                    "quietness warning: max action delta "
                    f"{quietness['max_abs_delta']:.4g} > {budget.quiet_eps}"
                )
                if require_quiet:
                    issues.append(message.replace("warning", "failed"))
                else:
                    warnings.append(message)
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"quietness probe failed: {exc}")

    liveness: Dict[str, float] = {}
    units = [int(u) for u in report.edited_units]
    if target_features is not None and len(target_features) and units:
        try:
            rec = _recurrence(candidate)
            net = RateCodedConnectomeSNN(
                candidate["w_in"],
                candidate["w_out"],
                rec,
                micro_steps=int(candidate.get("micro_steps", 10)),
                v_th=float(candidate.get("v_th", 1.0)),
                limit=12.0,
                readout_gain=float(candidate.get("readout_gain", 1.0)),
                bias=candidate.get("bias"),
            )
            net.reset()
            for row in np.asarray(target_features, dtype=np.float64):
                net.forward_input(row)
            if net.frames:
                rates = (
                    net.spike_frames[units].astype(np.float64)
                    / max(1, net.frames)
                    / 0.02
                )
                liveness = {
                    "min_rate_hz": float(np.min(rates)) if len(rates) else 0.0,
                    "max_rate_hz": float(np.max(rates)) if len(rates) else 0.0,
                    "mean_rate_hz": float(np.mean(rates)) if len(rates) else 0.0,
                    "frames": int(net.frames),
                }
                if liveness["max_rate_hz"] < float(budget.min_rate_hz):
                    warnings.append(
                        "edited units are silent on the target traces; the program "
                        "may be inert"
                    )
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"liveness probe failed: {exc}")

    return ValidationReport(
        ok=not issues,
        issues=issues,
        warnings=warnings,
        quietness=quietness,
        liveness=liveness,
        changed_params=int(report.changed_params),
        edited_units=list(report.edited_units),
        new_edges=int(report.new_edges),
        auto_isolated_units=list(report.auto_isolated_units),
        pruned_units=list(report.pruned_units),
    )


def diff_bundles(base: Dict[str, Any], candidate: Dict[str, Any], *, top_k: int = 200) -> Dict[str, Any]:
    """Sparse, browser-friendly description of what changed."""
    n = int(candidate["w_out"].shape[1])
    w_in_delta = np.asarray(candidate["w_in"]) - np.asarray(base["w_in"])
    w_out_delta = np.asarray(candidate["w_out"]) - np.asarray(base["w_out"])
    base_bias = base.get("bias")
    cand_bias = candidate.get("bias")
    if base_bias is None and cand_bias is None:
        bias_delta = np.zeros(n, dtype=np.float64)
    elif base_bias is None:
        bias_delta = np.asarray(cand_bias, dtype=np.float64)
    elif cand_bias is None:
        bias_delta = -np.asarray(base_bias, dtype=np.float64)
    else:
        bias_delta = np.asarray(cand_bias) - np.asarray(base_bias)

    changed_units = np.flatnonzero(
        np.any(np.abs(w_in_delta) > 1e-12, axis=1)
        | np.any(np.abs(w_out_delta) > 1e-12, axis=0)
        | (np.abs(bias_delta) > 1e-12)
    )
    order = changed_units[np.argsort(-(
        np.linalg.norm(w_in_delta[changed_units], axis=1)
        + np.linalg.norm(w_out_delta[:, changed_units], axis=0)
        + np.abs(bias_delta[changed_units])
    ))][: max(1, int(top_k))] if len(changed_units) else np.zeros(0, dtype=int)

    n_old = int(np.asarray(base["edges"]).shape[1])
    edges = np.asarray(candidate["edges"], dtype=np.int64)
    added_edges = [
        {"src": int(edges[1, i]), "dst": int(edges[0, i]), "magnitude": float(candidate["w_mag"][i])}
        for i in range(n_old, edges.shape[1])
    ]
    return {
        "n_neurons": n,
        "changed_units": [int(u) for u in changed_units],
        "top_units": [
            {
                "unit": int(u),
                "w_in_norm": float(np.linalg.norm(w_in_delta[int(u)])),
                "w_out_norm": float(np.linalg.norm(w_out_delta[:, int(u)])),
                "bias": float(bias_delta[int(u)]),
            }
            for u in order
        ],
        "added_edges": added_edges[: max(1, int(top_k))],
        "n_added_edges": int(edges.shape[1] - n_old),
        "changed_params": int(np.sum(np.abs(w_in_delta) > 1e-12)
                              + np.sum(np.abs(w_out_delta) > 1e-12)
                              + np.sum(np.abs(bias_delta) > 1e-12)),
    }


def save_repaired_bundle(
    path,
    bundle: Dict[str, Any],
    *,
    repair: Optional[Dict[str, Any]] = None,
) -> str:
    """Save a repaired bundle without changing the base fingerprint."""
    fields = bundle.get("fields")
    if not isinstance(fields, dict):
        raise ValueError("bundle has no fields; cannot preserve the fingerprint")
    scalars = {
        "seed": int(bundle.get("seed", 42)),
        "connectome_steps": int(bundle.get("connectome_steps", 3)),
        "micro_steps": int(bundle.get("micro_steps", 10)),
        "v_th": float(bundle.get("v_th", 1.0)),
    }
    if bundle.get("readout_gain") is not None:
        scalars["readout_gain"] = float(bundle["readout_gain"])
    return save_weights(
        path,
        w_in=bundle["w_in"],
        w_out=bundle["w_out"],
        w_mag=bundle["w_mag"],
        edges=bundle["edges"],
        polarity=bundle["polarity"],
        fields=fields,
        bias=bundle.get("bias"),
        repair=repair,
        **scalars,
    )
