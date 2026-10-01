"""Scene and reference helpers."""

from __future__ import annotations

import numpy as np
import pytest

from drone6dof.config import build_scene
from drone6dof.reference import goal_reference
from drone6dof.scene import SceneSpec
from drone6dof.task import ObstacleGoalTask


def test_scene_clearance_signs():
    scene = build_scene()
    obs = scene.obstacle_np
    assert scene.clearance(scene.goal_np) > 0.0
    at_center = np.array([obs[0], obs[1], 1.0])
    assert scene.clearance(at_center) == pytest.approx(-scene.core_radius)
    on_surface = np.array([obs[0] + scene.core_radius, obs[1], 1.0])
    assert scene.clearance(on_surface) == pytest.approx(0.0, abs=1e-9)


def test_scene_without_obstacle_is_infinite():
    scene = SceneSpec(obstacle=None).instantiate()
    assert scene.clearance([0.0, 0.0, 0.0]) == float("inf")
    series = scene.clearance_series(np.zeros((4, 3)))
    assert np.all(np.isinf(series))


def test_goal_reference_shape_and_meta():
    scene = build_scene()
    ref = goal_reference(steps=10, goal=scene.goal, dt=0.02, meta={"scene": scene})
    assert len(ref) == 10
    assert ref.pos.shape == (10, 3)
    assert np.allclose(ref.pos, scene.goal_np)
    assert np.allclose(ref.vel, 0.0) and np.allclose(ref.acc, 0.0)
    assert ref.at(0).meta["scene"] is scene


def test_task_telemetry_and_success_mask():
    scene = build_scene()
    task = ObstacleGoalTask()
    state = np.array([-2.2, 0.0, 0.5, 0.0, 0.0, 0.0])
    tele = task.frame_telemetry(state, scene)
    assert set(tele) == {"goal_dist", "clearance", "lyapunov_v", "lyapunov_vdot"}

    traj = np.array(
        [
            [0.0, 0.0, 2.5, 0, 0, 0],       # at the goal
            [-1.0, 0.2, 0.8, 0, 0, 0],      # inside the core
            [1.0, 1.0, 1.0, 0, 0, 0],       # clear
        ]
    )
    mask = task.success_mask(traj, scene)
    assert mask.tolist() == [True, False, True]
