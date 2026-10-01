"""PyChrono visualization smoke test (skipped without PyChrono or a display)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from drone6dof.viz import chrono_available

pytestmark = pytest.mark.skipif(
    not chrono_available() or not os.environ.get("DISPLAY"),
    reason="PyChrono and a display (Xvfb) are required",
)


def test_build_sync_and_render_one_frame():
    from drone6dof.config import CONTROL_LIMIT, INIT_STATE, build_scene
    from drone6dof.control import DSGuidanceController
    from drone6dof.dynamics import NumpyPlantBackend
    from drone6dof.sim import Simulation
    from drone6dof.viz import ChronoViz

    scene = build_scene()
    backend = NumpyPlantBackend(heading_target=scene.goal_np)
    controller = DSGuidanceController(action_limit=CONTROL_LIMIT, scene=scene)
    sim = Simulation(backend, controller, scene, steps=30, initial_state=INIT_STATE)

    viz = ChronoViz(camera_mode="fixed")
    viz.build(scene, INIT_STATE, steps=30)
    try:
        for _ in range(20):
            sim.step()
            viz.sync(sim.position, sim.rotation, backend.telemetry, sim.dt, sim.k)
        viz.vis.BeginScene()
        viz.vis.Render()
        viz.vis.EndScene()

        # the rendered body pose must match the mapped plant position
        expected = viz.to_vis(sim.position)
        actual = viz.drone.GetPos()
        assert np.allclose([actual.x, actual.y, actual.z], expected, atol=1e-9)
    finally:
        viz.close()
