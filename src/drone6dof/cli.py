"""Command-line entry point for the PyChrono 6-DoF drone demo."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .config import (
    CONTROL_LIMIT,
    DS_TEACHER_KWARGS,
    INIT_STATE,
    PLANT_GAIN,
    STEPS,
    build_scene,
)
from .control import ClassicalPDController, DSGuidanceController
from .dynamics import NumpyPlantBackend
from .params import DT, QuadParams
from .sim import Simulation
from .task import ObstacleGoalTask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drone6dof",
        description="PyChrono visualization of the ANN2SNN 6-DoF quadcopter example.",
    )
    parser.add_argument(
        "--controller",
        choices=["ds_guidance", "pid"],
        default="ds_guidance",
        help="controller driving the drone (default: ds_guidance)",
    )
    parser.add_argument(
        "--vis",
        choices=["irrlicht", "none"],
        default="irrlicht",
        help="render with Chrono/Irrlicht, or run headless (default: irrlicht)",
    )
    parser.add_argument("--seconds", type=float, default=STEPS * DT, help="episode length [s]")
    parser.add_argument("--dt", type=float, default=DT, help="outer timestep [s]")
    parser.add_argument("--gain", type=float, default=PLANT_GAIN, help="rotor efficiency")
    parser.add_argument("--damping", type=float, default=0.0, help="extra linear drag [1/s]")
    parser.add_argument("--inner-hz", type=float, default=QuadParams().inner_hz,
                        help="plant inner substep rate [Hz]")
    parser.add_argument("--seed", type=int, default=0, help="scene seed (deterministic)")
    parser.add_argument(
        "--camera",
        choices=["fixed", "orbit", "follow"],
        default="orbit",
        help="camera mode for the Irrlicht view (default: orbit)",
    )
    parser.add_argument("--out", default=None, help="write per-frame telemetry CSV here")
    parser.add_argument("--traj", default=None, help="write trajectory NPZ here")
    parser.add_argument("--record-dir", default=None,
                        help="write PNG snapshots every --record-every frames (Irrlicht)")
    parser.add_argument("--record-every", type=int, default=5, help="PNG snapshot stride")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="stop after N frames (debugging)")
    parser.add_argument("--json", action="store_true", help="print metrics as JSON")
    return parser


def make_controller(name: str, scene, control_limit: float):
    if name == "ds_guidance":
        return DSGuidanceController(
            action_limit=control_limit,
            scene=scene,
            task=ObstacleGoalTask(),
            **DS_TEACHER_KWARGS,
        )
    if name == "pid":
        return ClassicalPDController(
            plant_gain=PLANT_GAIN, action_limit=control_limit, pos_dim=3
        )
    raise ValueError(f"unknown controller {name!r}")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    dt = float(args.dt)
    steps = max(1, int(round(float(args.seconds) / dt)))
    scene = build_scene(seed=args.seed)
    params = QuadParams(inner_hz=float(args.inner_hz))
    backend = NumpyPlantBackend(params, heading_target=scene.goal_np)
    controller = make_controller(args.controller, scene, CONTROL_LIMIT)
    sim = Simulation(
        backend,
        controller,
        scene,
        dt=dt,
        steps=steps,
        gain=float(args.gain),
        damping=float(args.damping),
        control_limit=CONTROL_LIMIT,
        initial_state=INIT_STATE,
    )

    if args.vis == "none":
        from . import viz as _viz  # noqa: F401  (kept out of the import path otherwise)

        sim.run()
        metrics = sim.metrics()
        if args.out:
            sim.to_csv(args.out)
        if args.traj:
            sim.to_npz(args.traj)
        _report(metrics, args)
        return 0

    # Irrlicht
    from .viz import ChronoViz

    viz = ChronoViz(
        params,
        camera_mode=args.camera,
    )
    viz.build(scene, INIT_STATE, steps=steps)
    # A recording is a finite headless job: stop at the end of the episode so it
    # does not render forever waiting for a window close.
    max_frames = args.max_frames
    if args.record_dir and max_frames is None:
        max_frames = steps
    try:
        viz.render_loop(
            sim,
            realtime=True,
            max_frames=max_frames,
            record_dir=args.record_dir,
            record_every=args.record_every,
        )
    finally:
        viz.close()
    metrics = sim.metrics()
    if args.out:
        sim.to_csv(args.out)
    if args.traj:
        sim.to_npz(args.traj)
    _report(metrics, args)
    return 0


def _report(metrics: dict, args) -> None:
    if args.json:
        print(json.dumps(metrics, indent=2))
        return
    print("--- drone6dof ---")
    for key, value in metrics.items():
        print(f"{key:>20}: {value}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
