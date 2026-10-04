# ANN2SNN 6-DoF drone in PyChrono

A self-contained, Dockerized port of the 6-DoF quadcopter example from the
[ANN2SNN](https://github.com/romanklis/ANN2SNN) project (`drone-example` branch):
a numpy plant with the analytic DS-guidance teacher, the PID baseline, the
trained **connectome ANN**, and its **spiking (integrate-and-fire) transfer** —
rendered with [Project Chrono](https://projectchrono.org) / PyChrono, plus a
separate browser dashboard for quantitative comparison.

The flight task is the source's `quad6dof` navigation problem: start at
`(-2.2, 0, 0.5)`, reach the goal `(0, 0, 2.5)`, avoiding a vertical pillar at
`(-1.0, 0, 0.8)` (core radius 0.35 m). The teacher detours around the pillar; the
PID baseline flies straight into it.

Two surfaces, both Docker-only:

- **PyChrono view** (`make demo`) — the 3-D visualization (drone body, spinning
  rotors, body frame, trail, ground grid, pillar, goal, camera, HUD).
- **Dashboard** (`make dashboard`) — a Flask + Vite/Plotly page that runs the
  brains on the same scene and compares their flight properties (trajectories,
  spike raster, control output, goal distance, clearance, telemetry, metrics).
  It is data-only; the visualization stays in Chrono.

## Controllers

| `--controller` | what it is |
|---|---|
| `ds_guidance` | analytic dynamical-system teacher — the distillation label source |
| `pid` | classical PD + feed-forward baseline (collides with the pillar) |
| `ann` | 1000-neuron sparse recurrent connectome ANN, distilled from the teacher |
| `snn` | integrate-and-fire transfer of the connectome ANN (rate-coded, spiking) |

The SNN topology/dynamics match upstream: `N=1000`, fan-in `K=40` (40 000 edges),
20 % inhibitory scaled ×4 (Dale's law), `seed=42`; the ANN is evaluated with 3
recurrent steps per frame; the SNN runs 10 IF micro-steps per frame with
`v_th=1.0`. Policy input is `[e(3), ė(3), u_ff(3), task features(4)] = 13` →
3 accel commands. Inference is **numpy only** at runtime; torch is used only to
distil (in a separate training image).

## Quick start

```bash
make build        # PyChrono demo image
make demo         # interactive Irrlicht view (needs an X server)
make headless     # headless run -> out/telemetry.csv (+ trajectory.npz)
make test         # full test suite in-container under Xvfb

make train        # distil the connectome ANN/SNN -> weights/ (torch image)
make dashboard    # build + run the comparison dashboard on :8080
```

Override variables: `make demo CONTROLLER=snn CAMERA=follow`,
`make headless CONTROLLER=ann`, `make dashboard PORT=9000`.

### Interactive Chrono view

The container uses the host X server via `/tmp/.X11-unix` and `DISPLAY` (wired
into `make demo`/`make pid`). If the window does not open, allow the container
once: `xhost +local:root`. Software Mesa is used by default; add `--device
/dev/dri` for hardware GL. Camera modes: `fixed|orbit|follow`; the HUD shows
time, controller, goal distance, clearance, g-force, SoC, saturation, and — for
the SNN/ANN — firing rate and active-neuron fraction.

`make frames` records PNGs, but Irrlicht's framebuffer read returns black frames
in this base image (the live window is fine).

### Expected results

`ds_guidance` → `collisions=0`, `clearance_min_m>0`, `reached_goal=true`;
`pid` → `collisions>0`, `clearance_min_m<0`. The distilled `ann`/`snn` should also
clear the pillar and reach the goal; the dashboard reports the SNN−ANN delta.

## Obstacle avoidance (boxes + LiDAR)

The policy is **sensor-conditioned**: it never sees obstacle identities or
extents, only a horizontal LiDAR scan (`k=32` beams, `R_max=3 m`) plus cues
(nearest range, bearing, slope, curvature), with range noise, missed returns and
dropout. The analytic DS teacher is *privileged* (it uses the true geometry) and
computes the modulation from each obstacle's closest-point normal, generalising
the original cylinder formula to oriented boxes and cylinders; the student is
distilled from it (teacher-student).

- Geometry: `geometry.py` — signed distance, closest-point normal, ray casting.
- Sensor: `sensor.py` — scan model + cues.
- Teacher: `control.py` — modulation composed over the scene's obstacles.
- Training: randomised box/cylinder layouts + goals, sensor noise in the loop
  (`train.py`); the bundle fingerprint records the sensor config and layout spec.

Preset benchmark scenes (`pillar` = shipped cylinder, `boxes`, `wall`, `slalom`)
are selectable in the dashboard header and via `POST /api/benchmark {"scene": …}`.
Expected: the DS teacher clears every preset; the learned ANN/SNN generalise
within the trained distribution (report the sensor gap honestly).

## Dashboard

`make dashboard` builds `Dockerfile.dashboard` (node builds the Vite frontend,
then a slim Python runtime serves it) and runs it on `http://localhost:8080`.

- **API** (Flask): `GET /api/health`, `GET /api/controllers`,
  `POST /api/simulate` (`{controller, steps?, seed?, goal?}`),
  `POST /api/benchmark` (`{controllers?, steps?, seed?, goal?}` → per-controller
  traces, telemetry, spike events, metrics, and `stats` with the SNN−ANN deltas).
- **Hero page** (`/`): Plotly 3-D trajectories (goal + pillar), spike raster,
  control-output and tracking/clearance charts, a metrics/result bar, and a
  shared play cursor.
- **Extended page** (`/extended`): multi-lane traces (position, velocity,
  attitude, tracking error, command, telemetry), a frame readout, and
  metrics/weights tables.

### Interactive goal

The hero header has x/y/z goal inputs plus **Set** / **Reset**. Setting a goal
re-runs the selected brains to that location (batch re-run; custom goals use a
16 s horizon so far targets settle). The goal is stored in the URL
(`?gx=&gy=&gz=`) and shared with the extended page; bounds are
`x,y ∈ [-3,3]`, `z ∈ [0.2,3.5]` (out-of-bounds is a 400).

- The **DS teacher** is goal-relative, so it retargets exactly — it reliably
  reaches reachable goals.
- The **learned ANN/SNN** are distilled across randomised goals, so they follow
  the command too, but behaviour cloning generalises only partially: on some
  goals they orbit a little short of the 0.30 m tolerance. The dashboard reports
  `reached` / closest approach honestly rather than hiding it.

Note the DS is a *reactive* field, not a planner: there is no trajectory
re-optimisation, just a reshaped velocity field.

The dashboard has no in-browser training; it loads the committed
`weights/quad6dof_connectome.npz`. If that bundle is missing, `ann`/`snn` are
reported as unavailable and `ds_guidance`/`pid` still work.

## Weights and training

`weights/quad6dof_connectome.npz` is committed (self-contained: topology +
weights + a configuration fingerprint) so `--controller snn|ann` works out of the
box. `weights/quad6dof_reference_io.npz` holds a torch-reference input/output
sequence for the no-torch parity test. Regenerate both deterministically with:

```bash
make train-image     # build the CPU-torch training image (once)
make train           # distil + export into weights/
```

A bundle trained for a different plant/scene/teacher is refused by the
fingerprint guard, with instructions to rerun `make train`.

## Tests

`make test` runs the suite in the PyChrono image under Xvfb. Flask API tests run
in the dashboard image (`docker run --rm --entrypoint sh drone6dof-dashboard
-c 'pytest server/tests -q'` or `make dashboard-shell`).

- `tests/test_plant.py` — ported upstream plant cases; the `slow` test tracks the
  vendored prototype within 0.25 m.
- `tests/test_controllers.py`, `tests/test_scene_reference.py` — DS field, PID
  formula, clearance, reference/task.
- `tests/test_connectome.py` — topology determinism, Dale's law, E/I fraction,
  degree, ANN/SNN shapes, IF invariants.
- `tests/test_weights.py` — bundle loads, shape checks, fingerprint/missing guard.
- `tests/test_snn_parity.py` — numpy ANN matches the torch reference; SNN tracks
  the ANN (gain/r²) — no torch needed.
- `tests/test_end_to_end.py` — DS clears and reaches the goal; PID collides;
  ANN/SNN clear and reach; CSV export shape.
- `tests/test_benchmark.py` — dashboard report contract.
- `tests/test_viz_smoke.py` — builds the Chrono scene and renders one frame.
- `server/tests/test_api.py` — health, catalogue, simulate, benchmark, static.

## CLI

```
python -m drone6dof [--controller pid|ds_guidance|ann|snn] [--vis irrlicht|none]
                    [--seconds 10] [--dt 0.02] [--gain 1.0] [--damping 0.0]
                    [--weights weights/quad6dof_connectome.npz]
                    [--camera fixed|orbit|follow] [--out telemetry.csv]
                    [--traj trajectory.npz] [--record-dir frames/]
                    [--max-frames N] [--json]
```

## Architecture

```
Quad6DoF (numpy) ─┬─> Simulation ─> ChronoViz (Chrono/Irrlicht)
                  ├─> benchmark.py ─> Flask API ─> Vite/Plotly dashboard
controller ───────┘
```

- The plant's 500 Hz inner substepping is preserved; one `Simulation.step()` is
  one 20 ms frame.
- Controllers act on the **true** plant state (no Kalman estimator) — a
  deliberate simplification, recorded in the weight fingerprint.
- `ChronoDynamicsBackend` is a stub for a future Chrono-integrated rigid body.
- Coordinates are z-up; the Irrlicht camera requests Z-up
  (`SetCameraVertical`), falling back to an `R_x(-90°)` geometry rotation.

## Layout

```
src/drone6dof/       plant, controllers (ds/pid/ann/snn), policy, geometry, sensor,
                     weights, train, scene/reference/task, sim, benchmark, viz, cli
server/              Flask API (app.py, wsgi.py) + tests
web/                 Vite + Plotly frontend (hero + extended)
weights/             committed connectome bundle + torch reference I/O
tools/               vendored pure-numpy prototype (parity oracle)
Dockerfile           PyChrono+irrlicht demo image
Dockerfile.train     CPU-torch distillation image
Dockerfile.dashboard node build + Flask runtime
docker-compose.yml   interactive / headless / dashboard / test services
```

## Why a prebuilt PyChrono base image

No prebuilt PyChrono package includes a renderer: conda-forge builds with
`CH_ENABLE_MODULE_IRRLICHT=OFF` (only `core/fea/robot`), and the official
`projectchrono` channel ships only `core/fea/robot` too. The demo image therefore
builds on the third-party `lucamarchiano/pychrono_simulator:1.0` (pinned by
digest), which is source-built with `pychrono.irrlicht`. It is third-party and
not audited here; see `NOTICE`.

## Out of scope

Environment presets / robustness sweeps, multi-example selection, WebM/MP4
recording, in-dashboard training, the dense ANN, `pychrono.sensor` / ROS 2, and
the Kalman estimator.

## Attribution

Ported from the ANN2SNN project (`drone-example` branch, commit `bafe4cf`), MIT
licensed. Container base image: `lucamarchiano/pychrono_simulator:1.0`. See
`LICENSE` and `NOTICE` for full provenance.
