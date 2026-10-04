"""PyChrono visualization smoke test (skipped without PyChrono or a display).

The render runs in a **subprocess**: Irrlicht/software-GL can segfault, which in
this environment is flaky (it renders correctly on the real display) and would
otherwise take down the whole pytest process.  Retrying in a fresh process is
robust; a persistently broken renderer still fails the test.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from drone6dof.viz import chrono_available

pytestmark = pytest.mark.skipif(
    not chrono_available() or not os.environ.get("DISPLAY"),
    reason="PyChrono and a display (Xvfb) are required",
)

_SCRIPT = textwrap.dedent(
    """
    import numpy as np
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
    for _ in range(20):
        sim.step()
        viz.sync(sim.position, sim.rotation, backend.telemetry, sim.dt, sim.k)
    viz.vis.BeginScene()
    viz.vis.Render()
    viz.vis.EndScene()

    expected = viz.to_vis(sim.position)
    actual = viz.drone.GetPos()
    assert np.allclose([actual.x, actual.y, actual.z], expected, atol=1e-9)
    viz.close()
    print("VIZ_OK")
    """
)


def test_build_sync_and_render_one_frame():
    last = None
    for _ in range(3):
        last = subprocess.run(
            [sys.executable, "-c", _SCRIPT], capture_output=True, text=True
        )
        if last.returncode == 0 and "VIZ_OK" in last.stdout:
            return
    pytest.fail(
        f"chrono render subprocess failed (rc={last.returncode})\n"
        f"stdout tail: {last.stdout[-400:]}\n"
        f"stderr tail: {last.stderr[-400:]}"
    )
