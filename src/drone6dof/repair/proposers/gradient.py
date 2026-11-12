"""Optional gradient proposer (ablation only; not part of white-space repair).

The first implementation intentionally keeps the default loop gradient-free.  A
future BPTT/surrogate-gradient proposer can implement this interface in the
training image, write the same artifact records, and must pass the same
validation and unseen-evaluation gates.
"""

from __future__ import annotations

from ..types import EditProgram
from .base import ProposalContext, Proposer

__all__ = ["GradientProposer"]


class GradientProposer(Proposer):
    name = "gradient"

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def propose(self, context: ProposalContext) -> EditProgram:
        raise NotImplementedError(
            "the gradient proposer is an optional ablation and is not implemented "
            "in this build; use --proposer template, random or llm"
        )
