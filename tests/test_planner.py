"""Global A* planner and path reference: routes exist, are clear, are deterministic."""

from __future__ import annotations

import numpy as np
import pytest

from drone6dof.config import INIT_STATE, build_scene, preset_scene
from drone6dof.planner import PlannerConfig, path_clearance, plan_path, plan_path_grid
from drone6dof.reference import path_reference

_SCENES = ("pillar", "boxes", "wall", "slalom")


def _scene(name):
    return build_scene() if name == "pillar" else preset_scene(name)


@pytest.mark.parametrize("name", _SCENES)
def test_plan_exists_and_is_clear(name):
    scene = _scene(name)
    start = INIT_STATE[:2]
    goal = scene.goal_np[:2]
    path = plan_path(scene, start, goal)
    assert path is not None, name
    assert path[0] == pytest.approx((float(start[0]), float(start[1])))
    assert path[-1] == pytest.approx((float(goal[0]), float(goal[1])))
    assert path_clearance(scene, path) > 0.0


@pytest.mark.parametrize("name", _SCENES)
def test_plan_is_deterministic(name):
    scene = _scene(name)
    a = plan_path(scene, INIT_STATE[:2], scene.goal_np[:2])
    b = plan_path(scene, INIT_STATE[:2], scene.goal_np[:2])
    assert a == b


def test_goal_inside_inflation_is_handled():
    # wall/slalom place the goal within `inflate` of a surface; the planner must
    # still return a route (reduced inflation near the goal).
    for name in ("wall", "slalom"):
        scene = _scene(name)
        assert plan_path(scene, INIT_STATE[:2], scene.goal_np[:2]) is not None


def test_path_reference_shapes_and_terminal_stop():
    scene = _scene("slalom")
    ref = path_reference(scene, INIT_STATE[:3], scene.goal_np, steps=400, dt=0.02, speed=1.4)
    assert ref.pos.shape == (400, 3) and ref.vel.shape == (400, 3) and ref.acc.shape == (400, 3)
    assert np.all(np.isfinite(ref.pos)) and np.all(np.isfinite(ref.vel))
    assert ref.meta["kind"] == "path"
    assert np.linalg.norm(ref.pos[-1] - scene.goal_np) < 0.2
    assert np.linalg.norm(ref.vel[-1]) < 1e-6
    assert np.max(np.linalg.norm(ref.vel, axis=1)) <= 1.4 + 1e-6


def test_path_reference_falls_back_to_straight_line():
    scene = _scene("pillar")
    # an unreachable goal (outside the planning window) still yields a reference
    ref = path_reference(scene, INIT_STATE[:3], np.array([50.0, 50.0, 1.0]), steps=50)
    assert ref.pos.shape == (50, 3)


def test_plan_path_grid_routes_around_a_blocked_wall():
    xs = np.arange(-2.0, 2.0 + 1e-9, 0.1)
    ys = np.arange(-2.0, 2.0 + 1e-9, 0.1)
    blocked = np.zeros((len(xs), len(ys)), dtype=bool)
    # a wall at x~0 with a gap for |y| >= 0.8
    for i, x in enumerate(xs):
        if abs(x - 0.0) <= 0.05:
            for j, y in enumerate(ys):
                if abs(y) < 0.8:
                    blocked[i, j] = True
    way = plan_path_grid(blocked, xs, ys, (-1.5, 0.0), (1.5, 0.0))
    assert way is not None
    assert way[0] == pytest.approx((-1.5, 0.0))
    assert way[-1] == pytest.approx((1.5, 0.0))
    assert any(abs(p[1]) >= 0.8 for p in way)          # detours through the gap


def test_plan_path_grid_out_of_bounds_is_none():
    xs = np.arange(-1.0, 1.0 + 1e-9, 0.1)
    ys = np.arange(-1.0, 1.0 + 1e-9, 0.1)
    blocked = np.zeros((len(xs), len(ys)), dtype=bool)
    assert plan_path_grid(blocked, xs, ys, (-1.0, 0.0), (50.0, 50.0)) is None
