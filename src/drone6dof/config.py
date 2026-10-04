"""The ``quad6dof`` example configuration (ported values from ANN2SNN).

Constants mirror the ``QUAD6DOF`` :class:`ExampleSpec` on the ``drone-example``
branch (MIT); see ``NOTICE``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional, Sequence, Tuple

from .params import DT
from .scene import Scene, SceneSpec

__all__ = [
    "EXAMPLE_NAME",
    "DT",
    "STEPS",
    "CONTROL_LIMIT",
    "PLANT_GAIN",
    "INIT_STATE",
    "GOAL_TOLERANCE",
    "QUAD_SCENE",
    "DS_TEACHER_KWARGS",
    "GOAL_BOUNDS",
    "validate_goal",
    "build_scene",
]

EXAMPLE_NAME = "quad6dof"

#: 10 s at 50 Hz
STEPS: int = 500

#: acceleration *demand* limit [m/s^2] — the policy/teacher command space, not the
#: instantaneous thrust envelope (measured ~7.8 m/s² at full rotor efficiency).
CONTROL_LIMIT: float = 12.0
PLANT_GAIN: float = 1.0

INIT_STATE: Tuple[float, ...] = (-2.2, 0.0, 0.5, 0.0, 0.0, 0.0)

GOAL_TOLERANCE: float = 0.30

#: Pillar on the start->goal line; radii are the prototype's, radial gain re-tuned.
QUAD_SCENE = SceneSpec(
    goal=(0.0, 0.0, 2.5),
    obstacle=(-1.0, 0.0, 0.8),
    core_radius=0.35,
    influence_radius=1.05,
    dead_radius=0.62,
)

#: Teacher kwargs from the example (cap 1.4 so the nominal DS still clears the
#: pillar; radial gain 0.6).
DS_TEACHER_KWARGS = {"speed_cap": 1.4, "ds_radial_gain": 0.6}

#: Allowed interactive goal range (metres): keeps a commanded goal inside the
#: flight box the demo is validated for.
GOAL_BOUNDS: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
    (-3.0, 3.0),
    (-3.0, 3.0),
    (0.2, 3.5),
)


def validate_goal(goal: Optional[Sequence[float]]) -> Optional[Tuple[float, float, float]]:
    """Validate an interactive goal, returning it as floats.

    ``None`` means "use the shipped goal".  Raises ``ValueError`` for a malformed
    or out-of-bounds goal so the API can answer 400 instead of flying somewhere
    nonsensical.
    """
    if goal is None:
        return None
    try:
        values = [float(v) for v in goal]
    except (TypeError, ValueError) as exc:
        raise ValueError("goal must be three numbers [x, y, z]") from exc
    if len(values) != 3:
        raise ValueError("goal must be three numbers [x, y, z]")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("goal components must be finite")
    for value, (low, high), axis in zip(values, GOAL_BOUNDS, "xyz"):
        if not (low <= value <= high):
            raise ValueError(f"goal {axis}={value:.3f} outside [{low}, {high}]")
    return (values[0], values[1], values[2])


def build_scene(seed: int = 0, goal: Optional[Sequence[float]] = None) -> Scene:
    """Instantiate the example scene, optionally at an interactive ``goal``."""
    scene = QUAD_SCENE.instantiate(seed)
    goal = validate_goal(goal)
    if goal is not None:
        scene = dataclasses.replace(scene, goal=goal)
    return scene
