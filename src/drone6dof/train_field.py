"""Train the structured field network: LiDAR -> compact obstacle-field coefficients.

The network never learns the control law.  Given the privileged teacher potential
``U*`` (analytic, from the true obstacle geometry), the target is the vector of
``U*`` sampled at the trajectory-anchored basis centres.  A small connectome ANN
is distilled to that target and transferred to an integrate-and-fire SNN.

    python -m drone6dof.train_field --epochs 200
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .config import (
    CONTROL_LIMIT,
    DS_TEACHER_KWARGS,
    INIT_STATE,
    PLANT_GAIN,
    SENSOR,
    SENSOR_SUITE,
    STEPS,
    build_scene,
)
from .connectome import ConnectomeController, ConnectomeTopology
from .control import DSGuidanceController
from .dynamics import NumpyPlantBackend
from .estimator import ErrorStateUKF
from .field import FieldConfig, teacher_coeffs
from .sensors import SensorSuite
from .task import ObstacleGoalTask
from .train import (
    BOUNDS_HIGH,
    _input,
    _sample_goal,
    _scene_pool,
    _closed_loop_metrics,
)
from .weights import (
    DEFAULT_FIELD_PATH,
    default_field_fields,
    make_field_fields,
    save_field_weights,
)

__all__ = ["main", "build_field_dataset", "train_field"]

#: the field net is deliberately small (fewer parameters than the end-to-end net)
FIELD_NET = {"n_neurons": 128, "k": 8, "seed": 42}

_DEFAULT_FIELD = FieldConfig()
_FIELD_REF_PATH = Path(DEFAULT_FIELD_PATH).with_name("quad6dof_field_ref.npz")


def _rollout_frames(layout_scene, n_frames: int, state0) -> Tuple[list, list]:
    """Run teacher+plant+UKF and return per-frame (input, field target)."""
    task = ObstacleGoalTask(goal_tolerance=0.30)
    from .reference import goal_reference

    teacher = DSGuidanceController(
        action_limit=CONTROL_LIMIT, scene=layout_scene, task=task, **DS_TEACHER_KWARGS
    )
    goal = layout_scene.goal_np
    ref = goal_reference(steps=n_frames, goal=goal, dt=0.02, meta={"scene": layout_scene})
    backend = NumpyPlantBackend(heading_target=goal)
    estimator = ErrorStateUKF(params=backend.plant.p, sensor=SENSOR_SUITE, dt=0.02)
    sensors = SensorSuite(SENSOR_SUITE, lidar=SENSOR, obstacles=layout_scene.obstacles, dt=0.02)
    rng = np.random.default_rng(0)
    p0 = backend.reset(state0)
    est = estimator.reset(p0[:3], p0[3:6])
    sensors.reset()

    xs, ys = [], []
    for k in range(n_frames):
        rp = ref.at(k)
        u = teacher.act(est, rp)
        xs.append(_input(est, rp, layout_scene, SENSOR, rng))
        ys.append(teacher_coeffs(layout_scene, est, _DEFAULT_FIELD))
        backend.step(u, dt=0.02, limit=CONTROL_LIMIT, gain=PLANT_GAIN)
        truth = {
            "p": backend.position, "v": np.asarray(backend.state, dtype=np.float64)[3:6],
            "R": backend.rotation, "omega": backend.omega, "omega_m": backend.omega_m,
            "mass": backend.plant.p.m, "C_T": backend.plant.p.C_T,
        }
        est = estimator.step(sensors.measure(k + 1, truth), backend.rotor_target)
    return xs, ys


def build_field_dataset(
    scene,
    *,
    n_random: int = 3000,
    n_rollout: int = 300,
    n_scenes: int = 6,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """(LiDAR features, teacher field coefficients) samples across scenes."""
    rng = np.random.default_rng(int(seed))
    pool = _scene_pool(scene, rng, max(1, int(n_scenes)))

    xs, ys = [], []
    hi = np.asarray(BOUNDS_HIGH, dtype=np.float64)
    for i, layout_scene in enumerate(pool):
        state0 = np.asarray(INIT_STATE, dtype=np.float64) if i == 0 else np.concatenate(
            [rng.uniform(-hi, hi) * np.array([1, 1, 0.4]) + np.array([0, 0, 0.5]),
             rng.normal(0.0, 0.4, size=3)]
        )
        rx, ry = _rollout_frames(layout_scene, int(n_rollout), state0)
        xs.extend(rx)
        ys.extend(ry)

    from .reference import RefPoint
    for _ in range(int(n_random)):
        layout_scene = pool[int(rng.integers(len(pool)))]
        goal = layout_scene.goal_np if rng.random() < 0.5 else _sample_goal(rng, scene)
        pos = rng.uniform(-hi, hi)
        pos[2] = rng.uniform(0.0, hi[2])
        state = np.concatenate([pos, rng.normal(0.0, 0.8, size=3)])
        rp = RefPoint(pos=np.asarray(goal, dtype=np.float64), vel=np.zeros(3),
                      acc=np.zeros(3), meta={"scene": layout_scene})
        xs.append(_input(state, rp, layout_scene, SENSOR, rng))
        ys.append(teacher_coeffs(layout_scene, state, _DEFAULT_FIELD))

    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def train_field(
    scene,
    *,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 2e-3,
    seed: int = FIELD_NET["seed"],
    log_every: int = 50,
    n_random: int = 3000,
    n_rollout: int = 300,
    n_scenes: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ConnectomeTopology, dict]:
    import torch

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    topo = ConnectomeTopology(
        n_neurons=FIELD_NET["n_neurons"], k=FIELD_NET["k"], seed=int(seed)
    )
    n = topo.n_neurons
    x, y = build_field_dataset(scene, n_random=n_random, n_rollout=n_rollout,
                               n_scenes=n_scenes, seed=seed)
    n_in, k_out = int(x.shape[1]), int(y.shape[1])

    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    idx = torch.tensor(topo.edges.astype(np.int64))
    polarity_src = torch.tensor(topo.polarity[topo.edges[1]].astype(np.float32))
    init_mag = torch.tensor(topo.init_mag.astype(np.float32))

    w_in = torch.nn.Parameter(torch.randn(n, n_in, dtype=torch.float32) * 0.1)
    raw = torch.nn.Parameter(init_mag.clone())
    w_out = torch.nn.Parameter(torch.randn(k_out, n, dtype=torch.float32) * 0.1)

    def forward(xb):
        wrec = torch.sparse_coo_tensor(idx, raw.abs() * polarity_src, (n, n)).coalesce()
        h = torch.zeros(xb.shape[0], n)
        drive = xb @ w_in.t()
        for _ in range(3):
            h = torch.relu(drive + torch.sparse.mm(wrec, h.t()).t())
        return torch.relu(h @ w_out.t())     # coefficients are non-negative

    optim = torch.optim.Adam([w_in, raw, w_out], lr=float(lr))
    loss_fn = torch.nn.MSELoss()
    losses = []
    t0 = time.time()
    for epoch in range(1, int(epochs) + 1):
        perm = torch.randperm(x_t.shape[0])
        ep, nb = 0.0, 0
        for s in range(0, x_t.shape[0], int(batch_size)):
            b = perm[s:s + int(batch_size)]
            loss = loss_fn(forward(x_t[b]), y_t[b])
            optim.zero_grad()
            loss.backward()
            optim.step()
            ep += float(loss.item())
            nb += 1
        losses.append(ep / max(nb, 1))
        if log_every and (epoch % int(log_every) == 0 or epoch == 1):
            print(f"epoch {epoch:4d}  mse {losses[-1]:.6f}  ({time.time() - t0:.1f}s)")

    report = {"epochs": int(epochs), "samples": int(x_t.shape[0]),
              "final_mse": losses[-1], "losses": losses, "k": k_out}
    return (
        w_in.detach().numpy().astype(np.float64),
        w_out.detach().numpy().astype(np.float64),
        raw.detach().abs().numpy().astype(np.float64),
        topo,
        report,
    )


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Train the structured LiDAR->field network")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=FIELD_NET["seed"])
    ap.add_argument("--random-samples", type=int, default=3000)
    ap.add_argument("--rollout-frames", type=int, default=300)
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--out", default=str(DEFAULT_FIELD_PATH))
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args(argv)

    scene = build_scene(seed=0)
    print(f"training structured field net ({args.epochs} epochs, K={_DEFAULT_FIELD.k}) ...")
    w_in, w_out, w_mag, topo, report = train_field(
        scene, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
        n_random=args.random_samples, n_rollout=args.rollout_frames, n_scenes=args.scenes,
    )
    fields = default_field_fields(seed=int(topo.seed))
    save_field_weights(
        args.out, w_in=w_in, w_out=w_out, w_mag=w_mag, edges=topo.edges,
        polarity=topo.polarity, fields=fields, seed=int(topo.seed),
    )
    print(f"wrote {args.out}  (final mse {report['final_mse']:.6f}, K={report['k']})")

    bundle = {
        "w_in": w_in, "w_out": w_out, "w_mag": w_mag, "edges": topo.edges,
        "polarity": topo.polarity, "n_neurons": topo.n_neurons, "k": topo.k,
        "connectome_steps": 3, "micro_steps": 10, "v_th": 1.0,
        "field": FieldConfig().to_dict(),
    }
    # reference io for the no-torch parity test
    from .field import PotentialBasis
    x_ref, _ = build_field_dataset(scene, n_random=0, n_rollout=64, n_scenes=1, seed=0)
    import torch

    idx = torch.tensor(np.asarray(bundle["edges"], dtype=np.int64))
    signed = np.abs(bundle["w_mag"]) * np.asarray(bundle["polarity"])[
        np.asarray(bundle["edges"][1], dtype=np.int64)
    ]
    wrec = torch.sparse_coo_tensor(idx, torch.tensor(signed, dtype=torch.float32),
                                   (topo.n_neurons, topo.n_neurons)).coalesce()
    wi = torch.tensor(bundle["w_in"], dtype=torch.float32)
    wo = torch.tensor(bundle["w_out"], dtype=torch.float32)
    h = torch.zeros(1, topo.n_neurons)
    outs = []
    for row in x_ref[:64]:
        xb = torch.tensor(row, dtype=torch.float32).reshape(1, -1)
        drive = xb @ wi.t()
        for _ in range(3):
            h = torch.relu(drive + torch.sparse.mm(wrec, h.t()).t())
        outs.append(torch.relu(h @ wo.t())[0].detach().numpy())
    np.savez(_FIELD_REF_PATH, inputs=x_ref[:64], ann_out=np.asarray(outs))
    print(f"wrote {_FIELD_REF_PATH}")

    if not args.no_check:
        print("closed-loop check (estimate-only):")
        from .cli import make_controller
        for name in ("ds_guidance", "ann", "snn", "field_ann", "field_snn"):
            ctrl = make_controller(name, scene, CONTROL_LIMIT)
            m = _closed_loop_metrics(scene, ctrl, steps=STEPS)
            print(f"  {name:>10s}: collisions={m['collisions']:2d} "
                  f"clearance_min={m['clearance_min_m']:+.3f} "
                  f"final={m['final_goal_dist_m']:.3f} reached={m['reached_goal']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
