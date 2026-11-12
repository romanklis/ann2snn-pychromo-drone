"""Edit proposers for the white-space repair loop."""

from .base import ProposalContext, Proposer
from .template import TemplateProposer
from .random import RandomProposer

__all__ = [
    "ProposalContext",
    "Proposer",
    "TemplateProposer",
    "RandomProposer",
]
