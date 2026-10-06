# ANN2SNN 6-DoF drone in PyChrono

[![repo](https://img.shields.io/badge/github-ann2snn--pychrono--drone-181717?logo=github)](https://github.com/romanklis/ann2snn-pychrono-drone)
[![docs](https://img.shields.io/badge/docs-src--cited-22b8a6)](docs/index.md)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11-3776AB)](https://www.python.org)
[![runtime](https://img.shields.io/badge/runtime-docker-2496ED)](https://docs.docker.com)
[![3-D](https://img.shields.io/badge/3--D-Project%20Chrono-c98a2b)](https://projectchrono.org)

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

![Demo: PyChrono flight view and the comparison dashboard](docs/assets/demo.gif)

*The interactive PyChrono view and the browser dashboard (`docs/assets/demo.mp4`
is the higher-quality capture).*

![Drone control architecture: ANN-to-SNN learning and obstacle avoidance](docs/assets/architecture.png)

*System architecture — global DS / A\* guidance, the local LiDAR obstacle field,
the ANN→SNN transfer, and the UKF + PyChrono simulation loop. See
[Architecture](docs/architecture.md) for the module-level detail.*

## Contents

**Surfaces** — [Interactive Chrono view](#interactive-chrono-view) ·
[Dashboard](#dashboard) · [Interactive goal](#interactive-goal)

**Methods** — [Obstacle avoidance (LiDAR + field)](#obstacle-avoidance-boxes--lidar) ·
[State estimation (UKF)](#state-estimation-ukf-estimate-only-control) ·
[Structured SNN field + DS](#structured-snn-field--ds-modulation) ·
[SLAM-lite map](#slam-lite-map-dashboard)

**Project** — [Controllers](#controllers) · [Quick start](#quick-start) ·
[Weights & training](#weights-and-training) · [Tests](#tests) · [CLI](#cli) ·
[Architecture](#architecture) · [Layout](#layout) ·
[Why a prebuilt PyChrono base image](#why-a-prebuilt-pychrono-base-image) ·
[Out of scope](#out-of-scope) · [Attribution](#attribution) · [Author](#author)

**Deep dives (`docs/`)** — [Architecture](docs/architecture.md) ·
[Physics](docs/physics.md) · [Estimation](docs/estimation.md) ·
[Sensing](docs/sensing.md) · [Field](docs/field.md) · [Control](docs/control.md) ·
[SLAM](docs/slam.md) · [Limitations](docs/limitations.md)

## Documentation

Detailed, source-cited documentation lives in [`docs/`](docs/index.md):

- [Architecture](docs/architecture.md) — modules, layers, per-frame data flow
- [Physics](docs/physics.md) — equations of motion, actuator/battery model
- [Estimation](docs/estimation.md) — sensor suite and the error-state UKF
- [Sensing](docs/sensing.md) — obstacle geometry and the LiDAR scan + cues
- [Potential field](docs/field.md) — field methodology, structured ANN → SNN
- [Control](docs/control.md) — controllers and the execution loop

Schematics are tracked as replaceable placeholders under
[`docs/assets/schematics/`](docs/assets/schematics/README.md).

## Controllers

| `--controller` | what it is |
|---|---|
| `ds_guidance` | analytic dynamical-system teacher — the distillation label source |
| `pid` | classical PD + feed-forward baseline (collides with the pillar) |
| `ann` | 1000-neuron sparse recurrent connectome ANN, distilled from the teacher |
| `snn` | integrate-and-fire transfer of the connectome ANN (rate-coded, spiking) |
| `field_ann` | small ANN → trajectory-anchored obstacle-field coefficients + DS modulation |
| `field_snn` | spiking transfer of the field ANN; the SNN never outputs control commands |

The SNN topology/dynamics match upstream: `N=1000`, fan-in `K=40` (40 000 edges),
20 % inhibitory scaled ×4 (Dale's law), `seed=42`; the ANN is evaluated with 3
recurrent steps per frame; the SNN runs 10 IF micro-steps per frame with
`v_th=1.0`. Policy input is **sensor-conditioned**,
`[e(3), ė(3), u_ff(3), LiDAR ranges(32), cues(5)] = 46` → 3 accel commands (or
`K=24` field coefficients for the structured arms). Inference is **numpy only**
at runtime; torch is used only to distil (in a separate training image). The SNN
is a *rate-coded finite-window approximation* of the ANN — not a lossless
transfer — and there is **no independent safety layer**: safety comes only from
command clipping and plant saturation, never from the network. See
[docs/limitations.md](docs/limitations.md) for per-claim status.

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

`pid` collides (`collisions>0`, `clearance_min_m<0`); `ds_guidance` and the
structured `field_*` arms clear the obstacle on the evaluated scenes.
Estimate-only, the DS teacher and the end-to-end `ann`/`snn` clear but often stop
short of the 0.30 m goal tolerance — the dashboard reports `reached` and closest
approach honestly rather than hiding it. Exact numbers:
[docs/field.md §8](docs/field.md#8-measured-result).

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
- **Hero page** (`/`): Plotly 3-D trajectories (goal + pillar/boxes), an **oriented
  drone** per brain (body cross + rotors + body-frame triad, from the recorded
  attitude) and the **LiDAR rays/returns** for the selected sensor brain, a spike
  raster, control-output and tracking/clearance charts, a metrics/result bar, and a
  shared play cursor. The header is grouped into two dropdown menus: `Brains`
  (which models run) and `View` (the `drone`/`rays`/`spikes` toggles, per-model
  ray and spike checkboxes, and a `channels` selector for the raster). PID is
  kept as a CLI/compare baseline but hidden from the dashboard. The spike raster
  scales its height to the displayed channel count (64/128/200, uniform
  subsampling with an `M/N` label when capped). The camera/zoom persist while the
  animation plays, and obstacles are drawn as see-through boxes with an outline.
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

## State estimation (UKF, estimate-only control)

The plant is the **hidden truth**; every controller — the DS teacher, PID and the
learned ANN/SNN — sees only a fused state estimate. Each frame the loop runs
`plant → sensors → UKF → controller`:

- **Sensors** (`sensors.py`): GPS position (10 Hz, noise/dropouts — no bias
  term), IMU specific force + gyro (50 Hz, biases/noise/outliers; the specific
  force is the true body-frame non-gravitational force, so gravity is not
  measurable), INS velocity/attitude, barometer altitude, compass heading, and
  the LiDAR scan — with a seeded RNG for determinism.
- **UKF** (`estimator.py`): error-state (multiplicative) unscented filter over
  `[p, v, quaternion, ω, rotor speeds, accel/gyro biases]`; the process model is a
  simplified nonlinear quad (attitude kinematics, thrust mapping, rotor
  first-order lag + saturation) driven by the applied rotor speeds.
- **Rotor/actuator dynamics** already exist in the plant (first-order electrical
  lag `tau_e`, motor speed dynamics, current/speed saturation); the estimator
  mirrors a simplified version.

Controllers consume `Observation(state=estimate)`; the true state is never passed
to them. The learned arms are trained with the estimator in the loop (teacher
labels on the estimate), so training matches deployment. The bundle fingerprint
records the sensor suite + estimator config.

**Honest limitation:** estimate-only control is harder than the truth-based demo.
With the current estimator the teacher and ANN clear the pillar but the closed
loop orbits short of the 0.30 m tolerance. The estimator is deliberately simpler
than the plant (50 ms rotor lag, truth-plus-noise INS, GPS σ ≈ the goal
tolerance); it models the thrust + H-drag + fuselage-drag specific force but not
the rotor in-flow loss. See [docs/limitations.md](docs/limitations.md) for the
full per-claim status.

## Structured SNN field + DS modulation

Besides the end-to-end connectome controllers (`ann`/`snn`), the demo implements a
**structured** architecture where the network does **not** learn the control law:

```
LiDAR (k ranges + cues)
  → small SNN/ANN → a ∈ R_+^K        (compact obstacle-field coefficients, K=24)
  → U_obs(x)=Σ a_k φ_k(x−c_k)        (trajectory-anchored Gaussian RBF potential)
  → F_obs = −∇U_obs                  (analytic, conservative — never learned)
  → Billard-style DS modulation       (M→I away from obstacles: goal attractor kept)
  → nominal DS v_nom (stable)         (kept analytic)
  → existing impedance → u → plant    (reused low-level controller)
```

- `field.py` — the potential basis (centres anchored to a short **nominal-DS
  trajectory preview**, so the field is trajectory-relative), the privileged
  teacher barrier potential `U*=Σ 0.5·max(0, d_inf−clearance)²`, analytic
  gradient, modulation, and trajectory-cost/grid helpers.
- `field_control.py` — `FieldDSController` (`field_ann`, `field_snn`); the SNN
  outputs only the ``K`` coefficients.
- `train_field.py` / `make train-field` — distils a **small** network (128
  neurons, ~10k params) to the teacher potential sampled at the trajectory
  centres; exports `weights/quad6dof_field.npz`.
- `tools/compare_controllers.py` / `make compare` — the evaluation table.
- `tools/field_viz.py` / `make field-viz` — a self-contained Plotly HTML showing
  the LiDAR, learned `U_obs`/`F_obs`, nominal vs modulated DS and the trajectory.

Result (estimate-only, `make compare`, 500 steps, seed 0). `pillar`:

```
controller  coll  clr_min   final  reach  params
ds_guidance    0    0.019   1.317  False       -
pid           11   -0.223   0.106   True       -
ann            0    0.133   0.772  False   89000
snn            0    0.061   0.812  False   89000
field_ann      0    0.338   0.073   True    9984
field_snn      0    0.114   0.098   True    9984
```

The structured field arms now split roles (plan D): a **global A\* planner**
provides the path-tracking guidance `v_nom`, and the **LiDAR field is a gated
local residual** trained by gradient matching to the teacher field. This fixed
the previously reported failures — `wall` went from 44/42 collisions to **0** and
`slalom` from 28/11 to **0**, with the field arms reaching on 7 of 8
scene/arm combinations. The end-to-end `ann`/`snn` baselines are unchanged and
still fail on `wall`/`slalom`. Caveats: the planner uses the privileged scene
(a map), and the field **SNN** transfer fidelity is low (Pearson ≈0.45 after
readout calibration), so its residual is coarse. Per-scene table and per-claim
status: [docs/field.md §8](docs/field.md#8-measured-result) and
[docs/limitations.md](docs/limitations.md).

## SLAM-lite map (dashboard)

The hero dashboard is three columns: **left** telemetry charts (‖u‖, goal
distance, clearance), **centre** the 3-D view, **right** the **"SLAM map · what
the drone knows"** panel plus the spike raster. The map is drawn top-down (grey
unknown / dark free / amber occupied) with the ground-truth obstacle outline
overlaid and the trajectory in the owning brain's colour; **replanning events**
are marked on the trajectory and on a timeline strip, and light up when the
playback cursor passes them. Metrics: explored %, surface coverage, IoU (secondary
tooltip), replans + last time. A header **map** selector switches `SLAM`⇄`truth`.
Under SLAM the A\* planner consumes the live map with **event-driven** replanning
(path invalidation / goal change / slow refresh) and a frontier fallback; on the
four preset scenes the map-based runs stay collision-free and reach the goal. See
[docs/slam.md](docs/slam.md). `POST /api/benchmark` accepts
`"map_source": "truth" | "slam"`.

A **hidden** URL window `?from=<s>&to=<s>` (no UI control, combined with the
existing `gx/gy/gz/scene` params) trims the whole run to that time range: every
path, plot and the playback are sliced server-side, metrics are recomputed, and
the simulation is capped at `to` so it is not simulated past the window.

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
- `tests/test_snn_parity.py` — numpy ANN matches the torch reference; the
  rate-coded SNN is scored with MAE/RMSE/NRMSE/R²/Pearson/cosine/gain and the
  error-vs-micro-step-window check (`tools/snn_transfer_eval.py`) — no torch.
- `tests/test_estimator.py` — UKF tracking/determinism, covariance
  positive-definiteness, zero-bias-initialisation stability.
- `tests/test_imu_model.py` — the accelerometer reads body specific force
  (~1g at hover, regression guard against the old ~2g model).
- `tests/test_ds_stability.py` — nominal-DS eigenvalues and Lyapunov argument.
- `tests/test_control.py` — controller saturation and plant numerical robustness.
- `tests/test_end_to_end.py` — DS clears; PID collides; ANN/SNN clear (goal
  reach reported honestly); CSV export shape.
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

See [docs/architecture.md](docs/architecture.md) for the full module map and the
per-frame data flow.

- The plant's 500 Hz inner substepping is preserved; one `Simulation.step()` is
  one 20 ms frame.
- Controllers act on the **UKF state estimate**, not the hidden plant truth:
  every frame runs `plant → sensors → UKF → controller` (see
  [docs/estimation.md](docs/estimation.md)). The learned arms are distilled with
  the estimator in the loop, so training matches deployment.
- `ChronoDynamicsBackend` is a stub for a future Chrono-integrated rigid body.
- Coordinates are z-up; the Irrlicht camera requests Z-up
  (`SetCameraVertical`), falling back to an `R_x(-90°)` geometry rotation.

## Layout

```
src/drone6dof/       plant, estimator (UKF), sensors, controllers
                     (ds/pid/ann/snn/field_ann/field_snn), planner (A*), guidance,
                     slam (occupancy mapper), field, policy, geometry,
                     scene/reference/task, weights, train, sim, benchmark, viz, cli
server/              Flask API (app.py, wsgi.py) + tests
web/                 Vite + Plotly frontend (hero + extended)
weights/             committed connectome + field bundles + torch reference I/O
tools/               compare_controllers.py, field_viz.py (parity oracle snippet)
docs/                architecture / physics / estimation / sensing / field / control
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
recording, in-dashboard training, the dense ANN, and `pychrono.sensor` / ROS 2.

## Attribution

Ported from the ANN2SNN project (`drone-example` branch, commit `bafe4cf`), MIT
licensed. Container base image: `lucamarchiano/pychrono_simulator:1.0`. See
`LICENSE` and `NOTICE` for full provenance.

## Author

**Roman Pawel Klis** — Senior Manager, TOGAF-certified Data Architect and AI
Program Leader · PhD, ETH Zürich ·
[LinkedIn](https://www.linkedin.com/in/roman-pawel-klis-3811994)

12+ years across manufacturing, R&D and FMCG, bridging urgent business needs and
complex technical execution. Selected work:

- **Johnson Electric** — spearheaded the digital transformation of the global
  manufacturing shop floor: authored the data/AI strategy, built the large-scale
  organizational processes, and led global teams deploying AI initiatives
  (GenAI, LLMs, predictive maintenance) that cut testing times by 60%.
- **Philip Morris International** — during the global IQOS rollout, architected
  the data pipelines and AI-driven analytical backends that let market research
  and executive management capture, process and act on multi-market feedback in
  near real time, under FDA-preparatory governance.

**Core expertise**

- **AI leadership & strategy** — digital transformation, Lean/Agile, offshore
  team building, tech transfer.
- **Advanced AI** — agentic AI, generative AI, LLM fine-tuning (RAG), computer
  vision, predictive maintenance.
- **Data architecture** — high-throughput ETL/ELT pipelines, Spark, Hadoop,
  Azure/GCP, Docker/Kubernetes, IoT & sensor fusion.
- **Governance** — ISO-compliant data regimes, high-security access controls,
  FDA regulatory environments.

Passionate about mentoring the next generation of tech talent — most recently by
guiding the 1st-prize-winning team at the NASA Space Apps Challenge 2025 (Zurich).
Open to connecting with peers, leaders and recruiters on senior data, AI
leadership and R&D management roles.
