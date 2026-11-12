"""Deterministic white-space template: a context-gated reflex.

The template is a fallback for the LLM.  It finds an input feature whose target
distribution lies outside the in-distribution demo range, writes a threshold gate
into one spare neuron, and uses one more spare neuron as the readout of a small
fixed action nudge.  It is deliberately conservative: the gate is silent on the
demos, so a valid program is quiet even when the nudge is not the right one.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..activity import FEATURE_NAMES
from ..types import EditOp, EditProgram
from .base import ProposalContext, Proposer

__all__ = ["TemplateProposer"]


def _feature_separation(demo: np.ndarray, target: np.ndarray) -> Optional[Dict[str, Any]]:
    """Find the feature that separates target from demo most strongly."""
    if demo.size == 0 or target.size == 0:
        return None
    best: Optional[Dict[str, Any]] = None
    for j in range(demo.shape[1]):
        d = np.asarray(demo[:, j], dtype=np.float64)
        t = np.asarray(target[:, j], dtype=np.float64)
        dmax, dmin = float(np.max(d)), float(np.min(d))
        tmax, tmin = float(np.max(t)), float(np.min(t))
        scale = float(np.std(d) + 0.05)
        if tmax > dmax:
            score = (tmax - dmax) / scale
            direction = 1
            threshold = dmax
        elif tmin < dmin:
            score = (dmin - tmin) / scale
            direction = -1
            threshold = dmin
        else:
            continue
        # Prefer interpretable sensing/goal features over feed-forward zeros.
        if j in (0, 1, 2):
            score *= 1.5
        if j in (6, 7, 8, 44):
            score *= 1.4
        if j in (41, 42, 43):
            score *= 1.25
        if best is None or score > best["score"]:
            best = {
                "index": j,
                "name": FEATURE_NAMES[j] if j < len(FEATURE_NAMES) else f"feature_{j}",
                "direction": direction,
                "threshold": threshold,
                "score": score,
                "target_max": tmax,
                "target_min": tmin,
                "demo_max": dmax,
                "demo_min": dmin,
            }
    return best


def _action_hint(failure: str, target_rows: np.ndarray) -> List[float]:
    """A small fixed action nudge inferred from the failure taxonomy."""
    if target_rows.size == 0:
        return [0.0, 0.0, 0.3]
    mean = np.mean(np.asarray(target_rows, dtype=np.float64), axis=0)
    if failure in ("collision", "stall_short", "orbit", "sensor_blind"):
        cos_b = float(mean[42]) if mean.size > 43 else 0.0
        sin_b = float(mean[43]) if mean.size > 43 else 0.0
        return [-0.6 * cos_b, -0.6 * sin_b, 0.25]
    if failure in ("stall_at_goal", "wrong_direction", "timeout"):
        err = np.asarray(mean[:3], dtype=np.float64)
        n = float(np.linalg.norm(err))
        if n > 1e-6:
            err = err / n
        return [float(-0.5 * err[0]), float(-0.5 * err[1]), float(-0.5 * err[2])]
    if failure == "saturation":
        return [0.0, 0.0, 0.25]
    return [0.0, 0.0, 0.3]


class TemplateProposer(Proposer):
    name = "template"

    def propose(self, context: ProposalContext) -> EditProgram:
        summary = context.activity.summary
        spare = [
            int(item["unit"])
            for item in summary.get("top_spare", [])
            if int(item["unit"]) not in set(summary.get("critical_units", []))
        ]
        if len(spare) < 2:
            return EditProgram(
                program_id=f"cycle-{context.cycle:04d}-template",
                ops=[],
                rationale="not enough spare units for a gated reflex",
                proposer=self.name,
            )
        demo = np.asarray(context.demo_features, dtype=np.float64)
        target = np.asarray(context.target_features, dtype=np.float64)
        sep = _feature_separation(demo, target)
        if sep is None or sep["score"] <= 0.25:
            return EditProgram(
                program_id=f"cycle-{context.cycle:04d}-template",
                ops=[],
                rationale="no context feature separates the target from the demos",
                proposer=self.name,
            )

        gate, action_unit = spare[0], spare[1]
        j = int(sep["index"])
        direction = int(sep["direction"])
        threshold = float(sep["threshold"])
        margin = max(0.12, min(0.20, 0.01 * float(sep["score"])))
        weights = np.zeros(int(context.activity.summary["n_in"]), dtype=np.float64)
        weights[j] = float(direction)
        bias = -threshold - margin if direction > 0 else threshold - margin
        hint = _action_hint(context.diagnosis.failure_counts and max(
            context.diagnosis.failure_counts, key=context.diagnosis.failure_counts.get
        ) or "timeout", target)
        n_out = 3

        ops = [
            EditOp(
                kind="silence_output",
                unit=gate,
                rationale="gate must not contribute its old readout",
            ),
            EditOp(
                kind="isolate_unit",
                unit=gate,
                rationale="isolate recurrent synapses so only the new gate fires",
            ),
            EditOp(
                kind="isolate_unit",
                unit=action_unit,
                rationale="isolate recurrent synapses so only the gate drives it",
            ),
            EditOp(
                kind="set_input_row",
                unit=gate,
                values=[float(v) for v in weights],
                rationale=(
                    f"gate reads {sep['name']} in the "
                    f"{'high' if direction > 0 else 'low'} direction"
                ),
            ),
            EditOp(
                kind="set_bias",
                unit=gate,
                value=float(bias),
                rationale=f"threshold {threshold:.3f} with margin {margin:.3f}",
            ),
            EditOp(
                kind="set_input_row",
                unit=action_unit,
                values=[0.0] * int(context.activity.summary["n_in"]),
                rationale="action unit is driven only by the gate",
            ),
            EditOp(
                kind="set_bias",
                unit=action_unit,
                value=-0.2,
                rationale="suppress the action unit unless the gate fires",
            ),
            EditOp(
                kind="add_edge",
                src=int(gate),
                dst=int(action_unit),
                magnitude=1.0,
                rationale="gate -> action relay",
            ),
            EditOp(
                kind="set_output_col",
                unit=action_unit,
                values=[float(5.0 * v) for v in hint[:n_out]],
                rationale="fixed action nudge while the gate is on",
            ),
        ]
        return EditProgram(
            program_id=f"cycle-{context.cycle:04d}-template",
            ops=ops,
            rationale=(
                f"context gate on {sep['name']} with a fixed "
                f"{hint} action nudge"
            ),
            proposer=self.name,
            metadata={"feature": sep, "action": hint},
        )
