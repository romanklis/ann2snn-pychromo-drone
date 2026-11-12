"""Bounded random search baseline for the edit DSL."""

from __future__ import annotations

import numpy as np

from ..types import EditOp, EditProgram
from .base import ProposalContext, Proposer

__all__ = ["RandomProposer"]


class RandomProposer(Proposer):
    name = "random"

    def __init__(self, seed: int = 0, n_ops: int = 4, scale: float = 0.2) -> None:
        self.seed = int(seed)
        self.n_ops = int(n_ops)
        self.scale = float(scale)

    def propose(self, context: ProposalContext) -> EditProgram:
        rng = np.random.default_rng(self.seed + int(context.cycle))
        spare = [
            int(item["unit"])
            for item in context.activity.summary.get("top_spare", [])
        ]
        if not spare:
            return EditProgram(
                program_id=f"cycle-{context.cycle:04d}-random",
                ops=[],
                rationale="no spare units",
                proposer=self.name,
            )
        n_in = int(context.activity.summary["n_in"])
        n_out = int(context.activity.summary.get("n_out", 3))
        ops = []
        for _ in range(min(self.n_ops, len(spare))):
            unit = int(spare[int(rng.integers(len(spare)))])
            kind = str(rng.choice(["input", "output", "bias"]))
            if kind == "input":
                values = rng.normal(0.0, self.scale, size=n_in).tolist()
                ops.append(EditOp(kind="add_input_row", unit=unit, values=values,
                                  rationale="random input row"))
            elif kind == "output":
                values = rng.normal(0.0, self.scale, size=n_out).tolist()
                ops.append(EditOp(kind="add_output_col", unit=unit, values=values,
                                  rationale="random output column"))
            else:
                value = float(rng.normal(0.0, self.scale))
                ops.append(EditOp(kind="add_bias", unit=unit, value=value,
                                  rationale="random bias"))
        return EditProgram(
            program_id=f"cycle-{context.cycle:04d}-random",
            ops=ops,
            rationale="bounded random search baseline",
            proposer=self.name,
        )
