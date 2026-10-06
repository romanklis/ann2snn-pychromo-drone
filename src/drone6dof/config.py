"""The ``quad6dof`` example configuration (ported values from ANN2SNN).

Constants mirror the ``QUAD6DOF`` :class:`ExampleSpec` on the ``drone-example``
branch (MIT); see ``NOTICE``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional, Sequence, Tuple

from .geometry import BoxObstacle
from .params import DT
from .planner import PlannerConfig
from .scene import Scene, SceneSpec
from .sensor import SensorConfig
from .sensors import SensorConfig as SensorSuiteConfig

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
    "SENSOR",
    "SENSOR_SUITE",
    "ESTIMATOR",
    "PLANNER",
    "GUIDANCE",
    "SLAM",
    "MAP_SOURCE_DEFAULT",
    "OBSTACLE_LAYOUT",
    "PRESET_SCENES",
    "validate_goal",
    "preset_scene",
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
#: Margins are widened a little so estimate-noise excursions still clear it.
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

#: LiDAR model for the sensor-conditioned policy (see :mod:`drone6dof.sensor`).
SENSOR = SensorConfig()

#: Onboard sensor suite for the UKF (see :mod:`drone6dof.sensors`).
SENSOR_SUITE = SensorSuiteConfig()

#: UKF process-model parameters.
ESTIMATOR = {"rotor_tau": 0.05, "rotor_omega_max": 1400.0}

#: Global A* planner (see :mod:`drone6dof.planner`) + path-tracking guidance.
PLANNER = PlannerConfig()
GUIDANCE = {"speed": 1.4, "k_path": 1.0, "lookahead": 0.5, "brake": 0.8,
            "snap": 0.30, "hold": 0.80}

#: SLAM-lite occupancy mapper (LiDAR + UKF pose; see :mod:`drone6dof.slam`).
SLAM = {
    "res": 0.1,
    "bounds": [-3.5, 3.5, -3.5, 3.5],
    "log_odds_free": -0.4,
    "log_odds_occ": 0.85,
    "log_odds_clamp": 4.0,
    "occ_thresh": 0.6,
    "free_thresh": 0.4,
    "inflate": 0.30,
    "replan_period": 10,
    "viz_stride": 10,
    "viz_shape": 40,
}
#: default map source: "truth" keeps the shipped behaviour; "slam" uses the mapper
MAP_SOURCE_DEFAULT = "truth"

#: Obstacle distribution the policy is trained on (randomised box layouts).
OBSTACLE_LAYOUT = {
    "max_obstacles": 3,
    "n_obstacles_choices": [1, 2, 3],
    "box_half_min": 0.15,
    "box_half_max": 0.5,
    "cylinder_radius_min": 0.2,
    "cylinder_radius_max": 0.4,
    "region_x": 2.5,
    "region_y": 2.5,
    "min_clearance": 0.4,
    "use_cylinders": True,
}

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


def _box(cx, cy, hx, hy, angle=0.0):
    return BoxObstacle(
        center=(float(cx), float(cy)), half=(float(hx), float(hy)), angle=float(angle),
        dead_radius=0.55, influence_radius=1.4,
    )


#: Named benchmark scenes (boxes/cylinders; 2.5D vertical cross-sections).
PRESET_SCENES = {
    "pillar": lambda: (),
    "boxes": lambda: (
        _box(-1.0, 0.0, 0.35, 0.35, 0.4),
        _box(0.8, 1.2, 0.45, 0.25, -0.6),
    ),
    "wall": lambda: (_box(-0.5, 0.0, 0.15, 1.0, 0.0),),
    "slalom": lambda: (
        _box(-1.2, 0.6, 0.3, 0.3, 0.3),
        _box(0.0, -0.8, 0.3, 0.3, -0.3),
        _box(1.2, 0.6, 0.3, 0.3, 0.0),
    ),
}


def preset_scene(name: Optional[str] = None, goal: Optional[Sequence[float]] = None) -> Scene:
    """Scene for a named preset (``pillar`` = the shipped single cylinder)."""
    if not name or name == "pillar":
        return build_scene(goal=goal)
    if name not in PRESET_SCENES:
        raise ValueError(f"unknown scene {name!r}; choose from {sorted(PRESET_SCENES)}")
    resolved_goal = validate_goal(goal) or (0.0, 0.0, 2.5)
    return Scene(goal=tuple(resolved_goal), obstacles=PRESET_SCENES[name](), seed=0)
