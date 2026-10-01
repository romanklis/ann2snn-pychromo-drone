"""End-to-end closed-loop checks: DS guidance avoids, PID collides."""

from __future__ import annotations

from drone6dof.cli import make_controller
from drone6dof.config import CONTROL_LIMIT, INIT_STATE, build_scene
from drone6dof.dynamics import NumpyPlantBackend
from drone6dof.sim import Simulation


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


def test_csv_export_has_one_row_per_frame(tmp_path):
    _, sim = _run("ds_guidance", steps=40)
    path = tmp_path / "telemetry.csv"
    sim.to_csv(str(path))
    lines = path.read_text().strip().splitlines()
    assert len(lines) == len(sim.history["state"]) + 1     # header + recorded frames
    header = lines[0].split(",")
    for col in ("t", "px", "pz", "roll", "goal_dist", "clearance", "g_force", "rpm1"):
        assert col in header, col
