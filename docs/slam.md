# SLAM-lite: mapping, replanning and what the drone knows

The planner no longer reads the privileged scene online. A light mapper builds an
occupancy grid from the LiDAR scan and the UKF's estimated pose, and the planner
consumes that live map. This is **SLAM-lite**: localization comes from the
existing estimator, mapping from the scan, and there is **no loop closure**.

Source: `src/drone6dof/slam.py`, `planner.plan_path_grid`, `sim.Simulation`
(`map_source`), `field_control.FieldDSController`, `benchmark.py`.

## Map representation

`OccupancyMap` stores per-cell **log-odds** on a regular 2.5-D grid (`SLAM.res`,
`SLAM.bounds`). Each frame the mapper fuses one scan with an inverse sensor model:

- cells along `[0, r)` along a beam are updated **free**;
- the cell at the hit (`r < r_max`) is updated **occupied**;
- cells beyond the hit (or beyond `r_max` on a miss) stay **unknown**.

Because the LiDAR beams are **world-fixed** (`sensor.scan` uses `config.angles()`
and the world position), the mapper needs only the estimated `(x, y)` to place
the sensor, not the yaw. `occupancy()` thresholds the log-odds to
`0 = unknown, 1 = free, 2 = occupied` for compact transport and plotting.

## Planning, replanning and exploration

- `OccupancyMap.plan` runs `planner.plan_path_grid` (A* on the grid, with a grid
  line-of-sight string-pull). Replanning is **event-driven**: the controller
  replans when the current path is **invalidated** by a newly-observed obstacle
  (only *known* obstacles invalidate; unknown space does not), on a goal change,
  or after a slow `SLAM.refresh` floor (~40 frames), with a small anti-thrash gap.
  Each event is recorded (`replan_steps`) and marked in the UI. This replaced the
  original validity-blind cadence that produced ~79 replans per run.
- **Conservative** search: while planning, unknown cells count as **blocked**, so
  the planner never routes through unseen space. A blocked/unknown goal is
  **snapped to the nearest free cell**, which keeps a safe route toward the goal
  (advancing along known-free space and discovering more) rather than stalling;
  the nearest **frontier** is the explicit fallback.
- The learned LiDAR field remains the local residual on top of `v_nom`, with the
  distance gate and goal-capture fade unchanged.

## What the map learned — metrics

| Metric | Meaning |
|---|---|
| `explored_frac` | fraction of grid cells that have been observed at all |
| `entropy_bits` | mean binary entropy of observed cells (0 = certain) |
| `occupied_cells` | number of cells the map believes are obstacles |
| `replans` | number of successful A* plans (`goal_replans` + `frontier_replans`) |
| `replan_steps` | frame indices at which a replan happened (event markers) |
| `path_found_step` | first frame at which a goal route existed (`-1` = never) |
| `surface_coverage` | **recall** of the truth obstacle-shell cells the sensor could see (the fair "discovery" score) |
| `iou_vs_truth` | IoU of discovered occupied cells vs the *whole* truth shell (penalises never-observed far faces; secondary) |

`surface_coverage` restricts the truth shell to cells within `r_max` of the
trajectory, because a horizontal scan observes surfaces, not obstacle interiors —
comparing against the full shell understates discovery.

## Measured result (`field_ann`, 500 steps, seed 0)

| scene | map | coll | clearance min | final | reached | replans | surface cov |
|---|---|---|---|---|---|---|---|
| pillar | truth | 0 | 0.338 | 0.073 | yes | – | – |
| pillar | **SLAM** | 0 | 0.427 | 0.081 | yes | 12 | 0.50 |
| boxes | truth | 0 | 0.328 | 0.116 | yes | – | – |
| boxes | **SLAM** | 0 | 0.363 | 0.068 | yes | 12 | 0.68 |
| wall | truth | 0 | 0.250 | 0.471 | no | – | – |
| wall | **SLAM** | 0 | 0.077 | 0.221 | yes | 39 | 1.00 |
| slalom | truth | 0 | 0.338 | 0.162 | yes | – | – |
| slalom | **SLAM** | 0 | 0.329 | 0.095 | yes | 14 | 0.65 |

The online map matches the ground-truth-map runs: collision-free on all four
scenes, all reaching the goal, with a handful of meaningful replans. `wall` needs
the most (39) because its long barrier is only discovered on approach; the others
settle after ~12.

## Dashboard

The hero page is a **three-column** layout: **left** the telemetry charts
(‖u‖, goal distance, clearance), **centre** the (largest) 3-D view, **right** the
**"SLAM map · what the drone knows"** panel (`web/src/slamviz.js`) plus the spike
raster. The map is drawn top-down (grey unknown, dark free, amber occupied) with
the **ground-truth obstacle outline** overlaid, the trajectory and drone in the
owning **brain's colour**, and a legend. A **brain** selector in the panel chooses
whose map to show (both `FIELD·ANN` and `FIELD·SNN` are available when selected;
it defaults to `FIELD·SNN`). **Replanning events** are marked as ticks
on the trajectory and on a timeline strip under the map; when the playback cursor
is on a replan the strip highlights and the metrics show `⟳ replanning`. Metrics:
`explored`, `surface` (recall of the visible truth shell), `IoU` (secondary,
tooltip), and `replans` with the last replan time. A header **map** selector
switches between `SLAM` and `truth`. The **brain** selector in the panel chooses
which arm's map is shown (default `FIELD·SNN`), so its `surface`/`IoU` are that
brain's numbers.

## Limitations

- **2.5-D**: a single horizontal slice; obstacle height is ignored (obstacles are
  2.8 m, the goal 2.5 m, so nothing is flown over).
- **No loop closure**: the map accumulates at the (UKF-estimated) pose; there is
  no pose-graph optimisation.
- **Short range**: `r_max = 3 m` limits how far the map grows from the drone.
- **Distribution shift**: the field was distilled against truth-planner rollouts;
  map-based routes (with frontier detours) differ. Open-field results hold here,
  but a regeneration of field rollouts with map-based guidance is the mitigation
  if degradation appears.
- The teacher (modulation labels) stays **privileged/offline**; only the online
  planner was de-privileged. See [limitations.md](limitations.md).

## Hidden trim window

The dashboard also honours a **hidden** URL window `?from=<s>&to=<s>` (seconds,
combined with `gx/gy/gz/scene`; it is intentionally not in the UI). `run_benchmark`
slices every per-frame series (paths, state, scans, spikes, SLAM map frames and
series) to `[from, to]`, **recomputes** the metrics for that window, and — when
`to` is given — caps `steps` so the simulation is not run past it. Playback, the
charts and the SLAM panel then operate on the trimmed report (the map timeline
looks the frame up by its absolute time). Example:
`/?gx=0&gy=0&gz=1.5&scene=slalom&from=1.0&to=4.0`.
