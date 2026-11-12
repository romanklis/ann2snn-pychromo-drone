"""Proposer protocol and the context passed to every proposer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..activity import ActivityAudit
from ..scenarios import Scenario
from ..types import Budget, Diagnosis, EditProgram, SplitResult


@dataclass
class ProposalContext:
    cycle: int
    scenario: Scenario
    budget: Budget
    activity: ActivityAudit
    diagnosis: Diagnosis
    demo_features: np.ndarray
    target_features: np.ndarray
    history: List[Dict[str, Any]] = field(default_factory=list)
    best_metrics: Dict[str, Any] = field(default_factory=dict)
    base_bundle: Optional[Dict[str, Any]] = None
    practice_callback: Optional[Callable[[int], SplitResult]] = None
    candidate_callback: Optional[Callable[[Dict[str, Any], int], SplitResult]] = None
    allow_live: bool = False
    model: Optional[str] = None


class Proposer:
    """Base class; subclasses return an :class:`EditProgram`."""

    name = "proposer"

    def propose(self, context: ProposalContext) -> EditProgram:  # pragma: no cover
        raise NotImplementedError
