"""Controller output tests (no PyChrono required)."""

from __future__ import annotations

import numpy as np
import pytest

from drone6dof.config import DS_TEACHER_KWARGS, CONTROL_LIMIT, build_scene
from drone6dof.control import ClassicalPDController, DSGuidanceController
from drone6dof.reference import goal_reference


def _ref(scene):
    return goal_reference(steps=1, goal=scene.goal, meta={"scene": scene})


def test_pid_matches_the_analytic_formula():
    scene = build_scene()
    ref = _ref(scene)
    ctrl = ClassicalPDController(omega_n=3.5, zeta=0.85, plant_gain=1.0,
                                 action_limit=CONTROL_LIMIT)
    state = np.array([-1.5, 0.4, 1.2, 0.5, -0.3, 0.1])
    rp = ref.at(0)
    e = state[:3] - rp.pos
    edot = state[3:6] - rp.vel
    expected = -(3.5 ** 2 * e + 2 * 0.85 * 3.5 * edot)
    assert np.allclose(ctrl.raw_act(state, rp), expected)
    assert np.allclose(ctrl.act(state, rp), np.clip(expected, -CONTROL_LIMIT, CONTROL_LIMIT))


def test_pid_saturates_to_the_action_limit():
    scene = build_scene()
    ref = _ref(scene)
    ctrl = ClassicalPDController(action_limit=2.0, plant_gain=1.0)
    state = np.array([50.0, 50.0, 50.0, 0.0, 0.0, 0.0])
    u = ctrl.act(state, ref.at(0))
    assert np.all(np.abs(u) <= 2.0 + 1e-12)


def test_ds_command_is_finite_and_bounded():
    scene = build_scene()
    ref = _ref(scene)
    ctrl = DSGuidanceController(action_limit=CONTROL_LIMIT, scene=scene, **DS_TEACHER_KWARGS)
    state = np.array([-2.2, 0.0, 0.5, 0.0, 0.0, 0.0])
    u = ctrl.act(state, ref.at(0))
    assert u.shape == (3,)
    assert np.isfinite(u).all()
    assert np.all(np.abs(u) <= CONTROL_LIMIT + 1e-12)
    assert ctrl.last_telemetry["ds_goal_dist"] == pytest.approx(
        float(np.linalg.norm(state[:3] - scene.goal_np))
    )


def test_ds_modulates_near_the_obstacle():
    """Inside the influence radius the DS field must deflect away from the pillar."""
    scene = build_scene()
    ref = _ref(scene)
    ctrl = DSGuidanceController(action_limit=CONTROL_LIMIT, scene=scene, **DS_TEACHER_KWARGS)
    obs = scene.obstacle_np
    # just in front of the pillar, on the start->goal line
    state = np.array([obs[0] - 0.5, 0.0, 0.8, 1.0, 0.0, 0.0])
    ctrl.act(state, ref.at(0))
    tele = ctrl.last_telemetry
    assert tele["ds_active"] == 1.0
    assert tele["ds_deflection_deg"] > 1.0


def test_ds_far_from_obstacle_is_not_deflected():
    scene = build_scene()
    ref = _ref(scene)
    ctrl = DSGuidanceController(action_limit=CONTROL_LIMIT, scene=scene, **DS_TEACHER_KWARGS)
    # beyond the influence radius of the pillar but with the same goal bearing
    state = np.array([-2.2, 2.4, 0.5, 0.0, 0.0, 0.0])
    ctrl.act(state, ref.at(0))
    assert ctrl.last_telemetry["ds_active"] == 0.0
    # the arccos in the deflection metric has an epsilon denominator, so "no
    # deflection" is a hair above zero rather than exactly zero (same as upstream)
    assert ctrl.last_telemetry["ds_deflection_deg"] < 0.1
