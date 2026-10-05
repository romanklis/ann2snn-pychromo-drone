"""End-to-end closed-loop checks: DS guidance avoids, PID collides."""

from __future__ import annotations

from pathlib import Path

import pytest

from drone6dof.benchmark import weights_info
from drone6dof.cli import make_controller
from drone6dof.config import CONTROL_LIMIT, INIT_STATE, build_scene
from drone6dof.dynamics import NumpyPlantBackend
from drone6dof.sim import Simulation
from drone6dof.weights import DEFAULT_WEIGHTS_PATH

_HAS_WEIGHTS = bool(weights_info().get("loaded"))


def _run(name: str, steps: int = 500) -> tuple[dict, Simulation]:
    scene = build_scene()
    backend = NumpyPlantBackend(heading_target=scene.goal_np)
    controller = make_controller(name, scene, CONTROL_LIMIT)
    sim = Simulation(
        backend, controller, scene, steps=steps, initial_state=INIT_STATE
    )
    sim.run()
    return sim.metrics(), sim


def test_ds_guidance_clears_the_pillar_and_approaches_the_goal():
    """Estimate-only control: avoid and approach (the estimate noise widens the
    final tolerance, so the strict 0.30 m reach is not asserted here)."""
    metrics, sim = _run("ds_guidance")
    assert metrics["collisions"] == 0
    assert metrics["clearance_min_m"] > 0.05
    assert metrics["closest_goal_dist_m"] < 0.6, metrics
    assert metrics["peak_g_force"] > 1.0


def test_pid_collides_with_the_pillar():
    metrics, _ = _run("pid")
    assert metrics["collisions"] > 0
    assert metrics["clearance_min_m"] < 0.0


@pytest.mark.skipif(not _HAS_WEIGHTS, reason="connectome weights bundle not present")
@pytest.mark.parametrize("name", ["ann", "snn"])
def test_learned_brains_avoid_and_approach(name):
    """Sensor-only arms clear the pillar and approach the goal.

    The policy sees a noisy LiDAR scan, not the oracle geometry, so it is weaker
    than the teacher and may not settle inside the 0.30 m tolerance in 10 s; the
    meaningful checks are avoidance and a close approach.
    """
    metrics, _ = _run(name)
    assert metrics["collisions"] == 0, metrics
    assert metrics["clearance_min_m"] > 0.0, metrics
    # avoidance is strict; the approach budget is loose because estimate-only
    # control is conservative (the SNN especially)
    budget = {"ann": 0.6, "snn": 2.0}[name]
    assert metrics["closest_goal_dist_m"] < budget, metrics


@pytest.mark.skipif(not _HAS_WEIGHTS, reason="connectome weights bundle not present")
@pytest.mark.parametrize("name", ["ann", "snn"])
def test_learned_brains_avoid_box_obstacles(name):
    from drone6dof.config import preset_scene

    scene = preset_scene("boxes", goal=(0.0, 0.0, 2.5))
    backend = NumpyPlantBackend(heading_target=scene.goal_np)
    controller = make_controller(name, scene, CONTROL_LIMIT)
    sim = Simulation(backend, controller, scene, steps=500, initial_state=INIT_STATE)
    sim.run()
    metrics = sim.metrics()
    assert metrics["collisions"] == 0, metrics
    assert metrics["clearance_min_m"] > 0.0, metrics


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


def test_ds_avoids_a_box_obstacle():
    """The teacher generalises from the cylinder to a box (closest-point normal)."""
    from drone6dof.geometry import BoxObstacle
    from drone6dof.scene import Scene

    scene = Scene(
        goal=(2.0, 2.0, 1.5),
        obstacles=(BoxObstacle(center=(-1.0, 0.0), half=(0.3, 0.5), angle=0.3),),
    )
    backend = NumpyPlantBackend(heading_target=scene.goal_np)
    controller = make_controller("ds_guidance", scene, CONTROL_LIMIT)
    sim = Simulation(backend, controller, scene, steps=800, initial_state=INIT_STATE)
    sim.run()
    metrics = sim.metrics()
    assert metrics["collisions"] == 0, metrics
    assert metrics["clearance_min_m"] > 0.0, metrics
    assert metrics["final_goal_dist_m"] < 0.6, metrics


def test_csv_export_has_one_row_per_frame(tmp_path):
    _, sim = _run("ds_guidance", steps=40)
    path = tmp_path / "telemetry.csv"
    sim.to_csv(str(path))
    lines = path.read_text().strip().splitlines()
    assert len(lines) == len(sim.history["state"]) + 1     # header + recorded frames
    header = lines[0].split(",")
    for col in ("t", "px", "pz", "roll", "goal_dist", "clearance", "g_force", "rpm1"):
        assert col in header, col
