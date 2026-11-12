"""White-space repair for the distilled connectome SNN.

The package is numpy-only so the repair loop and its inspector can run in the
dashboard image.  See ``docs/repair.md`` for the design and the CLI.
"""

from .types import (
    Budget,
    CycleRecord,
    Diagnosis,
    EditOp,
    EditProgram,
    EpisodeResult,
    SplitResult,
    ValidationReport,
)

__all__ = [
    "Budget",
    "CycleRecord",
    "Diagnosis",
    "EditOp",
    "EditProgram",
    "EpisodeResult",
    "SplitResult",
    "ValidationReport",
]
