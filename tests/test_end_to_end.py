"""End-to-end closed-loop checks: DS guidance avoids, PID collides."""

from __future__ import annotations

from pathlib import Path

import pytest

from drone6dof.cli import make_controller
from drone6dof.config import CONTROL_LIMIT, INIT_STATE, build_scene
from drone6dof.dynamics import NumpyPlantBackend
from drone6dof.sim import Simulation
from drone6dof.weights import DEFAULT_WEIGHTS_PATH

_HAS_WEIGHTS = Path(DEFAULT_WEIGHTS_PATH).exists()


def _run(name: str, steps: int = 500) -> tuple[dict, Simulation]:
    scene = build_scene()
    backend = NumpyPlantBackend(heading_target=scene.goal_np)
    controller = make_controller(name, scene, CONTROL_LIMIT)
    sim = Simulation(
        backend, controller, scene, steps=steps, initial_state=INIT_STATE
    )
    sim.run()
    return sim.metrics(), sim


def test_ds_guidance_clears_the_pillar_and_reaches_the_goal():
    metrics, sim = _run("ds_guidance")
    assert metrics["collisions"] == 0
    assert metrics["clearance_min_m"] > 0.05
    assert metrics["reached_goal"] is True
    assert metrics["peak_g_force"] > 1.0


def test_pid_collides_with_the_pillar():
    metrics, _ = _run("pid")
    assert metrics["collisions"] > 0
    assert metrics["clearance_min_m"] < 0.0


@pytest.mark.skipif(not _HAS_WEIGHTS, reason="connectome weights bundle not present")
@pytest.mark.parametrize("name", ["ann", "snn"])
def test_learned_brains_clear_the_pillar_and_reach_the_goal(name):
    metrics, _ = _run(name)
    assert metrics["collisions"] == 0, metrics
    assert metrics["clearance_min_m"] > 0.0, metrics
    assert metrics["reached_goal"] is True, metrics


def test_ds_reaches_a_moved_goal():
    """The teacher generalises to an interactive goal (straight path clears the pillar)."""
    scene = build_scene(goal=(2.0, 2.0, 1.5))
    backend = NumpyPlantBackend(heading_target=scene.goal_np)
    controller = make_controller("ds_guidance", scene, CONTROL_LIMIT)
    # far goals need a longer horizon than the shipped 10 s to settle (the DS
    # leaves a small steady orbit); 16 s is the dashboard's custom-goal horizon
    sim = Simulation(backend, controller, scene, steps=800, initial_state=INIT_STATE)
    sim.run()
    metrics = sim.metrics()
    assert metrics["reached_goal"] is True, metrics
    assert metrics["collisions"] == 0, metrics
    assert metrics["closest_goal_dist_m"] < 0.30, metrics


def test_csv_export_has_one_row_per_frame(tmp_path):
    _, sim = _run("ds_guidance", steps=40)
    path = tmp_path / "telemetry.csv"
    sim.to_csv(str(path))
    lines = path.read_text().strip().splitlines()
    assert len(lines) == len(sim.history["state"]) + 1     # header + recorded frames
    header = lines[0].split(",")
    for col in ("t", "px", "pz", "roll", "goal_dist", "clearance", "g_force", "rpm1"):
        assert col in header, col
