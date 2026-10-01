# ANN2SNN 6-DoF drone in PyChrono

A self-contained, Dockerized visualization of the 6-DoF quadcopter example from
the [ANN2SNN](https://github.com/romanklis/ANN2SNN) project (`drone-example`
branch), rendered with [Project Chrono](https://projectchrono.org) / PyChrono.

Your own numpy integrator computes the flight dynamics; **Chrono is the
mechanical visualization layer** (drone body, spinning rotors, body frame,
trajectory trail, ground grid, pillar, goal marker, camera, HUD). A small
`DynamicsBackend` interface leaves room to move the dynamics into Chrono later,
and to attach `pychrono.sensor` / ROS 2.

The flight task is the source's `quad6dof` navigation problem: start at
`(-2.2, 0, 0.5)`, reach the goal `(0, 0, 2.5)`, avoiding a vertical pillar at
`(-1.0, 0, 0.8)` (core radius 0.35 m). The analytic DS-guidance teacher detours
around the pillar; the PID baseline flies straight into it.

## Why a prebuilt base image

No prebuilt PyChrono package includes a renderer or sensors:

- conda-forge builds PyChrono with `CH_ENABLE_MODULE_IRRLICHT=OFF`,
  `VEHICLE=OFF`, `ROS=OFF` (only `core`, `fea`, `robot`);
- the official `projectchrono` channel also ships only `core`, `fea`, `robot`
  for 9.0.1 and 10.0.0.

The Dockerfile therefore builds on the third-party image
`lucamarchiano/pychrono_simulator:1.0` (pinned by digest), a source-built
PyChrono that includes `pychrono.irrlicht` (plus ROS 2 Humble). The base image is
third-party and not audited here; see `NOTICE`.

## Everything runs in Docker

The `drone6dof` image is the only runtime. You do **not** need PyChrono, numpy,
Xvfb or conda on the host — only Docker (and an X server for the interactive
view).

```bash
make build        # build the image
make demo         # interactive Irrlicht view (needs an X server)
make pid          # interactive view; the PID baseline collides with the pillar
make headless     # headless run -> out/telemetry.csv + out/trajectory.npz
make test         # full test suite in-container under Xvfb
make help         # list all targets
```

Override variables on the command line:

```bash
make demo CONTROLLER=pid CAMERA=follow
make headless SECONDS=20
```

`environment.yml` is not used; the container environment is the base image.

## Interactive mode

The container uses the host X server through `/tmp/.X11-unix` and `DISPLAY`
(already wired into `make demo` / `make pid`). On a host with X access control,
allow the container's root user first:

```bash
xhost +local:root
make demo
```

If the window opens but the view is empty/software-rendered, that is expected:
`LIBGL_ALWAYS_SOFTWARE=1` uses Mesa llvmpipe. For hardware GL, add
`--device /dev/dri` to the run command.

Camera modes: `make demo CAMERA=fixed|orbit|follow`. The window stays open at
the end of the episode; close it to exit. The HUD shows time, controller,
goal distance, clearance, g-force, SoC and saturation.

## Headless runs

```bash
make headless                      # DS guidance -> out/telemetry.csv
make headless CONTROLLER=pid       # PID baseline (collides)
make frames                        # PNG snapshots -> out/frames (headless)
```

Expected metrics: `ds_guidance` → `collisions=0`, `clearance_min_m>0`,
`reached_goal=true`; `pid` → `collisions>0`, `clearance_min_m<0`.

Note: PNG capture (`make frames`) uses Irrlicht's framebuffer read, which returns
black frames in this base image (the interactive window renders correctly, as
shown by an external screen capture). Use the interactive view for visual
inspection; `make frames` is provided for the wiring, not reliable output.

## Tests

`make test` runs the whole suite inside the image under Xvfb (via
`scripts/xvfb.sh`, because `xvfb-run` hangs in this base image), so the Chrono
smoke test executes. `make test-fast` skips the slow prototype-parity test.

- `tests/test_plant.py` — ported upstream plant cases (hover at 1 g, tilt
  direction, SO(3) hygiene, determinism, inner-rate convergence, authority
  saturation, damping). The `slow` test runs the vendored prototype for 8 s and
  requires the ported plant to track it within 0.25 m.
- `tests/test_controllers.py` — DS-guidance field and PID formula.
- `tests/test_scene_reference.py` — clearance signs, goal reference, task mask.
- `tests/test_end_to_end.py` — DS guidance clears the pillar and reaches the
  goal; PID collides; CSV export shape.
- `tests/test_viz_smoke.py` — builds the Chrono scene, syncs 20 frames and
  renders one frame.

## CLI

Inside the container:

```
python -m drone6dof [--controller pid|ds_guidance] [--vis irrlicht|none]
                    [--seconds 10] [--dt 0.02] [--gain 1.0] [--damping 0.0]
                    [--camera fixed|orbit|follow] [--out telemetry.csv]
                    [--traj trajectory.npz] [--record-dir frames/]
                    [--max-frames N] [--json]
```

## Architecture

```
Quad6DoF (numpy)  ->  DynamicsBackend  ->  Simulation  ->  ChronoViz (Chrono/Irrlicht)
                          ^                    ^
                       controller            telemetry
```

- The plant's 500 Hz inner substepping is preserved; one `Simulation.step()` is
  one 20 ms frame.
- Controllers act on the **true** plant state (no Kalman estimator). This matches
  the vendored prototype and is a deliberate simplification for the demo.
- `ChronoDynamicsBackend` is a documented stub for a future milestone where
  Chrono integrates the quad as a rigid body driven by rotor thrust.
- Coordinates: the plant is z-up and the demo requests a Z-up Irrlicht camera
  (`SetCameraVertical`), falling back to rotating rendered geometry by
  `R_x(-90°)` if that is unavailable.

## Layout

```
src/drone6dof/       plant, controllers, scene/reference/task, simulation, viz, CLI
tools/               vendored pure-numpy prototype (parity oracle)
tests/               plant/controller/end-to-end tests + Chrono smoke test
scripts/xvfb.sh      temporary Xvfb runner for headless tests
Dockerfile           PyChrono+irrlicht base + Xvfb/GL/ffmpeg
docker-compose.yml   interactive / headless / test services
Makefile             all targets (container-only)
```

## Out of scope (future work)

Chrono-integrated rigid-body dynamics, `pychrono.sensor`
(camera/LiDAR/GPS/IMU), ROS 2, ANN/SNN controllers and training, the web
dashboard, and the Kalman estimator / embodiment noise models.

## Attribution

Ported from the ANN2SNN project (`drone-example` branch, commit `bafe4cf`), MIT
licensed. Container base image: `lucamarchiano/pychrono_simulator:1.0`. See
`LICENSE` and `NOTICE` for full provenance.
