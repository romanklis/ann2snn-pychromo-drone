"""Distil the connectome ANN from the DS teacher and export a numpy bundle.

This is the *only* module that needs torch.  It is run in a dedicated training
image (``Dockerfile.train`` / ``make train``); the demo image runs the exported
numpy bundle with no torch.

    python -m drone6dof.train --epochs 300 --out weights/quad6dof_connectome.npz

The teacher is the analytic DS-guidance controller, evaluated on the **true**
plant state (the same convention as the ported controllers).  The policy is
memoryless, so a coverage sample of states across the flight box plus the nominal
DS rollout is enough to clone the field; the closed-loop check at the end
confirms the distilled network actually flies the obstacle course.
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
    GOAL_BOUNDS,
    INIT_STATE,
    PLANT_GAIN,
    STEPS,
    build_scene,
)
from .connectome import ConnectomeController, ConnectomeTopology, DEFAULTS
from .control import DSGuidanceController
from .dynamics import NumpyPlantBackend
from .params import DT, QuadParams
from .policy import error_vector
from .reference import RefPoint, goal_reference
from .sim import Simulation
from .task import ObstacleGoalTask
from .weights import DEFAULT_REF_IO_PATH, DEFAULT_WEIGHTS_PATH, default_fields, save_weights

__all__ = ["main", "build_dataset", "train_connectome"]

#: flight box the policy must cover for distillation (matches the example bounds)
BOUNDS_HIGH = (2.5, 2.5, 3.0)


def _input(state: np.ndarray, ref, scene, task: ObstacleGoalTask) -> np.ndarray:
    """13-D policy input with ``u_ff = 0`` (the constant-goal feed-forward)."""
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    err = error_vector(state, ref, 3)
    feats = task._from_positions(state[:3].reshape(1, 3), scene)[0]
    return np.concatenate([err, np.zeros(3), feats]).astype(np.float64)


def _sample_goal(rng: np.random.Generator, scene) -> np.ndarray:
    """Uniform goal in the allowed box, kept outside the pillar core."""
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = GOAL_BOUNDS
    obs = scene.obstacle_np
    while True:
        goal = np.array(
            [rng.uniform(xlo, xhi), rng.uniform(ylo, yhi), rng.uniform(zlo, zhi)]
        )
        if obs is None:
            return goal
        if np.hypot(goal[0] - obs[0], goal[1] - obs[1]) > scene.core_radius + 0.15:
            return goal


def build_dataset(
    scene,
    *,
    n_random: int = 40000,
    n_rollout: int = 500,
    seed: int = 0,
    n_rollout_goals: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Teacher ``(state -> accel)`` samples over **randomised goals**.

    Goal randomisation is what lets the learned arms be commanded to arbitrary
    locations interactively (the teacher is goal-relative; the students must see
    that distribution).  ``n_rollout`` frames are rolled out for each of
    ``n_rollout_goals`` goals (the shipped goal first), and the coverage samples
    each get their own sampled goal.
    """
    task = ObstacleGoalTask(goal_tolerance=0.30)
    teacher = DSGuidanceController(
        action_limit=CONTROL_LIMIT, scene=scene, task=task, **DS_TEACHER_KWARGS
    )
    rng = np.random.default_rng(int(seed))
    xs, ys = [], []

    goals = [scene.goal_np] + [
        _sample_goal(rng, scene) for _ in range(max(1, int(n_rollout_goals)) - 1)
    ]
    for goal in goals:
        ref = goal_reference(steps=n_rollout, goal=goal, dt=DT, meta={"scene": scene})
        backend = NumpyPlantBackend(QuadParams(), heading_target=goal)
        state = backend.reset(np.asarray(INIT_STATE, dtype=np.float64))
        for k in range(int(n_rollout)):
            rp = ref.at(k)
            u = teacher.act(state, rp)
            xs.append(_input(state, rp, scene, task))
            ys.append(u)
            state = backend.step(u, dt=DT, limit=CONTROL_LIMIT, gain=PLANT_GAIN)

    # coverage: uniform positions in the flight box, each with a sampled goal
    hi = np.asarray(BOUNDS_HIGH, dtype=np.float64)
    pos = rng.uniform(-hi, hi, size=(n_random, 3))
    pos[:, 2] = rng.uniform(0.0, hi[2], size=n_random)
    vel = rng.normal(0.0, 0.8, size=(n_random, 3))
    zeros3 = np.zeros(3, dtype=np.float64)
    for i in range(int(n_random)):
        goal = _sample_goal(rng, scene)
        rp = RefPoint(pos=goal, vel=zeros3, acc=zeros3, meta={"scene": scene})
        s = np.concatenate([pos[i], vel[i]])
        xs.append(_input(s, rp, scene, task))
        ys.append(teacher.act(s, rp))

    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def build_sequences(
    scene,
    *,
    n_traj: int = 8,
    traj_len: int = 500,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Teacher-driven trajectories ``(n_traj, traj_len, 13)`` / ``(…, 3)``.

    One nominal DS rollout plus perturbed initial conditions, each toward a
    **randomly sampled goal** (the shipped goal for the first trajectory).  Training
    on these with the hidden state carried matches deployment and teaches the
    students the goal-relative field they need for interactive goal commands.
    """
    task = ObstacleGoalTask(goal_tolerance=0.30)
    teacher = DSGuidanceController(
        action_limit=CONTROL_LIMIT, scene=scene, task=task, **DS_TEACHER_KWARGS
    )
    rng = np.random.default_rng(int(seed))
    hi = np.asarray(BOUNDS_HIGH, dtype=np.float64)

    xs_all, ys_all = [], []
    for i in range(int(n_traj)):
        goal = scene.goal_np if i == 0 else _sample_goal(rng, scene)
        ref = goal_reference(steps=traj_len, goal=goal, dt=DT, meta={"scene": scene})
        if i == 0:
            state = np.asarray(INIT_STATE, dtype=np.float64)
        else:
            pos = rng.uniform(-hi, hi)
            pos[2] = rng.uniform(0.0, hi[2])
            vel = rng.normal(0.0, 0.5, size=3)
            state = np.concatenate([pos, vel])
        backend = NumpyPlantBackend(QuadParams(), heading_target=goal)
        state = backend.reset(state)
        xs, ys = [], []
        for k in range(int(traj_len)):
            rp = ref.at(k)
            xs.append(_input(state, rp, scene, task))
            ys.append(teacher.act(state, rp))
            state = backend.step(ys[-1], dt=DT, limit=CONTROL_LIMIT, gain=PLANT_GAIN)
        xs_all.append(xs)
        ys_all.append(ys)
    return np.asarray(xs_all, dtype=np.float64), np.asarray(ys_all, dtype=np.float64)


def train_connectome(
    scene,
    *,
    epochs: int = 80,
    batch_size: int = 1024,
    lr: float = 0.006,
    n_random: int = 3000,
    n_rollout: int = 500,
    n_traj: int = 6,
    traj_len: int = 500,
    window: int = 100,
    seed: int = 0,
    topology_seed: int = 0,
    micro_steps: int = DEFAULTS["micro_steps"],
    connectome_steps: int = DEFAULTS["connectome_steps"],
    v_th: float = DEFAULTS["v_th"],
    log_every: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ConnectomeTopology, dict]:
    """Distil the ANN. Returns ``(w_in, w_out, w_mag, topology, report)``."""
    import torch

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    # The anatomy seed is recorded in the bundle (consumers reuse the bundle's
    # edge list, not the seed); ``seed`` only seeds data/shuffle/weight init.
    topo = ConnectomeTopology(seed=int(topology_seed))
    n = topo.n_neurons
    x_rand, y_rand = build_dataset(scene, n_random=n_random, n_rollout=n_rollout, seed=seed)
    x_seq, y_seq = build_sequences(scene, n_traj=n_traj, traj_len=traj_len, seed=seed)
    n_in, n_out = int(x_rand.shape[1]), int(y_rand.shape[1])

    xr = torch.tensor(x_rand, dtype=torch.float32)
    yr = torch.tensor(y_rand, dtype=torch.float32)
    xs = torch.tensor(x_seq, dtype=torch.float32)
    ys = torch.tensor(y_seq, dtype=torch.float32)

    idx = torch.tensor(topo.edges.astype(np.int64))
    polarity_src = torch.tensor(topo.polarity[topo.edges[1]].astype(np.float32))
    init_mag = torch.tensor(topo.init_mag.astype(np.float32))

    w_in = torch.nn.Parameter(torch.randn(n, n_in, dtype=torch.float32) * 0.1)
    raw = torch.nn.Parameter(init_mag.clone())
    w_out = torch.nn.Parameter(torch.randn(n_out, n, dtype=torch.float32) * 0.1)

    def recurrence():
        w = raw.abs() * polarity_src
        return torch.sparse_coo_tensor(idx, w, (n, n)).coalesce()

    def forward_seq(xb, h):
        """``xb`` is ``(B, L, n_in)``; returns ``(B, L, n_out)`` and the last ``h``."""
        wrec = recurrence()
        outs = []
        for t in range(xb.shape[1]):
            h = torch.relu(xb[:, t, :] @ w_in.t() + torch.sparse.mm(wrec, h.t()).t())
            outs.append(h)
        hidden = torch.stack(outs, dim=1)
        return torch.clamp(hidden @ w_out.t(), -CONTROL_LIMIT, CONTROL_LIMIT), h

    optim = torch.optim.Adam([w_in, raw, w_out], lr=float(lr))
    loss_fn = torch.nn.MSELoss()

    losses = []
    t0 = time.time()
    n_seq = int(xs.shape[0])
    seq_batch = max(1, int(batch_size) // max(1, int(window)))
    for epoch in range(1, int(epochs) + 1):
        epoch_loss, n_batches = 0.0, 0

        # sequence pass: hidden state carried across the trajectory (deployment)
        order = torch.randperm(n_seq)
        for start in range(0, n_seq, seq_batch):
            b = order[start:start + seq_batch]
            h = torch.zeros(len(b), n, dtype=torch.float32)
            for c in range(0, int(traj_len), int(window)):
                xb = xs[b, c:c + int(window)]
                yb = ys[b, c:c + int(window)]
                pred, h = forward_seq(xb, h)
                loss = loss_fn(pred, yb)
                optim.zero_grad()
                loss.backward()
                optim.step()
                h = h.detach()
                epoch_loss += float(loss.item())
                n_batches += 1

        # coverage pass: random states from h = 0
        perm = torch.randperm(xr.shape[0])
        for start in range(0, xr.shape[0], int(batch_size)):
            b = perm[start:start + int(batch_size)]
            xb = xr[b].unsqueeze(1)
            yb = yr[b].unsqueeze(1)
            h = torch.zeros(xb.shape[0], n, dtype=torch.float32)
            pred, _ = forward_seq(xb, h)
            loss = loss_fn(pred, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        mean_loss = epoch_loss / max(n_batches, 1)
        losses.append(mean_loss)
        if log_every and (epoch % int(log_every) == 0 or epoch == 1):
            print(f"epoch {epoch:4d}  mse {mean_loss:.6f}  ({time.time() - t0:.1f}s)")

    report = {
        "epochs": int(epochs),
        "samples": int(xr.shape[0]) + int(xs.shape[0]) * int(traj_len),
        "final_mse": losses[-1],
        "losses": losses,
        "connectome_steps": int(connectome_steps),
        "micro_steps": int(micro_steps),
        "v_th": float(v_th),
    }
    return (
        w_in.detach().cpu().numpy().astype(np.float64),
        w_out.detach().cpu().numpy().astype(np.float64),
        raw.detach().abs().cpu().numpy().astype(np.float64),
        topo,
        report,
    )


def _reference_io(scene, bundle: dict, *, frames: int = 64, seed: int = 0):
    """Torch ANN outputs for a DS-rollout input sequence (deployment semantics)."""
    import torch

    task = ObstacleGoalTask(goal_tolerance=0.30)
    ref = goal_reference(steps=STEPS, goal=scene.goal, dt=DT, meta={"scene": scene})
    teacher = DSGuidanceController(
        action_limit=CONTROL_LIMIT, scene=scene, task=task, **DS_TEACHER_KWARGS
    )
    backend = NumpyPlantBackend(QuadParams(), heading_target=scene.goal_np)
    state = backend.reset(np.asarray(INIT_STATE, dtype=np.float64))

    inputs = []
    for k in range(frames):
        rp = ref.at(k)
        inputs.append(_input(state, rp, scene, task))
        u = teacher.act(state, rp)
        state = backend.step(u, dt=DT, limit=CONTROL_LIMIT, gain=PLANT_GAIN)
    x = np.asarray(inputs, dtype=np.float64)

    n = int(bundle["n_neurons"])
    idx = torch.tensor(np.asarray(bundle["edges"], dtype=np.int64))
    signed = np.abs(bundle["w_mag"]) * np.asarray(bundle["polarity"])[
        np.asarray(bundle["edges"][1], dtype=np.int64)
    ]
    w_in = torch.tensor(bundle["w_in"], dtype=torch.float32)
    w_out = torch.tensor(bundle["w_out"], dtype=torch.float32)
    wrec = torch.sparse_coo_tensor(idx, torch.tensor(signed, dtype=torch.float32), (n, n)).coalesce()

    h = torch.zeros(1, n, dtype=torch.float32)
    outs = []
    for row in x:
        xb = torch.tensor(row, dtype=torch.float32).reshape(1, -1)
        drive = xb @ w_in.t()
        for _ in range(int(bundle.get("connectome_steps", 3))):
            h = torch.relu(drive + torch.sparse.mm(wrec, h.t()).t())
        outs.append(torch.clamp(h @ w_out.t(), -CONTROL_LIMIT, CONTROL_LIMIT).detach().numpy()[0])
    return x, np.asarray(outs, dtype=np.float64)


def _closed_loop_metrics(scene, controller, *, steps: int = STEPS) -> dict:
    backend = NumpyPlantBackend(QuadParams(), heading_target=scene.goal_np)
    sim = Simulation(
        backend, controller, scene, steps=steps, initial_state=INIT_STATE,
        gain=PLANT_GAIN, control_limit=CONTROL_LIMIT,
    )
    sim.run()
    return sim.metrics()


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Distil and export the connectome ANN/SNN bundle")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.006)
    ap.add_argument("--random-samples", type=int, default=3000)
    ap.add_argument("--rollout-frames", type=int, default=500)
    ap.add_argument("--trajectories", type=int, default=6,
                    help="teacher rollouts trained with the hidden state carried")
    ap.add_argument("--trajectory-length", type=int, default=500)
    ap.add_argument("--window", type=int, default=100, help="truncated BPTT window")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topology-seed", type=int, default=0,
                    help="connectome anatomy seed (stored in the bundle)")
    ap.add_argument("--out", default=str(DEFAULT_WEIGHTS_PATH))
    ap.add_argument("--ref-io", default=str(DEFAULT_REF_IO_PATH))
    ap.add_argument("--no-check", action="store_true", help="skip the closed-loop sanity check")
    args = ap.parse_args(argv)

    scene = build_scene(seed=0)
    print(f"distilling connectome ANN ({args.epochs} epochs, seed {args.seed}) ...")
    w_in, w_out, w_mag, topo, report = train_connectome(
        scene,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_random=args.random_samples,
        n_rollout=args.rollout_frames,
        n_traj=args.trajectories,
        traj_len=args.trajectory_length,
        window=args.window,
        seed=args.seed,
        topology_seed=args.topology_seed,
    )

    fields = default_fields()
    bundle_path = save_weights(
        args.out,
        w_in=w_in,
        w_out=w_out,
        w_mag=w_mag,
        edges=topo.edges,
        polarity=topo.polarity,
        fields=fields,
        seed=int(topo.seed),
    )
    print(f"wrote {bundle_path}  (final mse {report['final_mse']:.6f})")

    bundle = {
        "w_in": w_in, "w_out": w_out, "w_mag": w_mag, "edges": topo.edges,
        "polarity": topo.polarity, "n_neurons": topo.n_neurons, "k": topo.k,
        "connectome_steps": DEFAULTS["connectome_steps"],
        "micro_steps": DEFAULTS["micro_steps"], "v_th": DEFAULTS["v_th"],
        "seed": topo.seed,
    }
    x, ann_out = _reference_io(scene, bundle)
    Path(args.ref_io).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.ref_io, inputs=x, ann_out=ann_out, connectome_steps=DEFAULTS["connectome_steps"])
    print(f"wrote {args.ref_io}  ({x.shape[0]} reference frames)")

    if not args.no_check:
        print("closed-loop check:")
        for name in ("ds_guidance", "pid", "ann", "snn"):
            if name in ("ds_guidance", "pid"):
                from .cli import make_controller
                ctrl = make_controller(name, scene, CONTROL_LIMIT)
            else:
                ctrl = ConnectomeController(name, bundle, action_limit=CONTROL_LIMIT)
            m = _closed_loop_metrics(scene, ctrl)
            print(
                f"  {name:>12s}: collisions={m['collisions']:2d}  "
                f"clearance_min={m['clearance_min_m']:+.3f}  "
                f"final_goal_dist={m['final_goal_dist_m']:.3f}  "
                f"reached_goal={m['reached_goal']}"
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
