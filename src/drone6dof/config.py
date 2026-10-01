"""The ``quad6dof`` example configuration (ported values from ANN2SNN).

Constants mirror the ``QUAD6DOF`` :class:`ExampleSpec` on the ``drone-example``
branch (MIT); see ``NOTICE``.
"""

from __future__ import annotations

from typing import Tuple

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


def build_scene(seed: int = 0) -> Scene:
    """Instantiate the example scene (deterministic)."""
    return QUAD_SCENE.instantiate(seed)
