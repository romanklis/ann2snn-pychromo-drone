"""Controller saturation and plant numerical robustness."""

from __future__ import annotations

import numpy as np
import pytest

from drone6dof.config import build_scene
from drone6dof.control import ClassicalPDController, DSGuidanceController
from drone6dof.params import QuadParams
from drone6dof.plant import Quad6DoF
from drone6dof.reference import RefPoint


def _ref(pos):
    return RefPoint(pos=np.asarray(pos, dtype=float), vel=np.zeros(3), acc=np.zeros(3))


def test_pd_output_is_saturated():
    ctrl = ClassicalPDController(action_limit=1.0)
    u = ctrl.act(np.zeros(6), _ref([100.0, 0.0, 0.0]))
    assert np.all(np.abs(u) <= 1.0 + 1e-9)
    assert np.all(np.isfinite(u))


def test_ds_output_is_saturated():
    scene = build_scene()
    ctrl = DSGuidanceController(action_limit=1.0, scene=scene)
    u = ctrl.act(np.array([50.0, 0.0, 0.5, 0.0, 0.0, 0.0]), _ref(scene.goal_np))
    assert np.all(np.abs(u) <= 1.0 + 1e-9)
    assert np.all(np.isfinite(u))


def test_plant_is_finite_and_orthonormal_under_extreme_command():
    plant = Quad6DoF(QuadParams())
    state = plant(np.zeros(6), np.array([100.0, -100.0, 100.0]),
                  dt=0.02, limit=12.0, gain=1.0)
    assert np.all(np.isfinite(state))
    assert np.all(np.isfinite(plant.specific_force_body))
    assert np.linalg.norm(plant.R.T @ plant.R - np.eye(3)) < 1e-6
    assert np.linalg.det(plant.R) == pytest.approx(1.0, abs=1e-9)
    assert np.all(plant.omega_m >= 0.0)
