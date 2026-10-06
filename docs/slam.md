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
  line-of-sight string-pull) and **replans** every `SLAM.replan_period` frames.
- **Conservative** by default: unknown cells are treated as **blocked**, so the
  planner never routes through space it has not seen. When no goal route exists,
  the controller plans to the nearest **frontier** (a free cell next to unknown),
  driving exploration until a corridor to the goal is discovered
  (`field_control._ensure_tracker`).
- The learned LiDAR field remains the local residual on top of `v_nom`, with the
  distance gate and goal-capture fade unchanged.

## What the map learned — metrics

| Metric | Meaning |
|---|---|
| `explored_frac` | fraction of grid cells that have been observed at all |
| `entropy_bits` | mean binary entropy of observed cells (0 = certain) |
| `occupied_cells` | number of cells the map believes are obstacles |
| `replans` | number of A* replans over the run |
| `path_found_step` | first frame at which a goal route existed (`-1` = never) |
| `surface_coverage` | **recall** of the truth obstacle-shell cells the sensor could see (the fair "discovery" score) |
| `iou_vs_truth` | IoU of discovered occupied cells vs the *whole* truth shell (penalises never-observed far faces; secondary) |

`surface_coverage` restricts the truth shell to cells within `r_max` of the
trajectory, because a horizontal scan observes surfaces, not obstacle interiors —
comparing against the full shell understates discovery.

## Measured result (`field_ann`, 500 steps, seed 0)

| scene | map | coll | clearance min | final | reached | explored | surface cov |
|---|---|---|---|---|---|---|---|
| pillar | truth | 0 | 0.338 | 0.073 | yes | – | – |
| pillar | **SLAM** | 0 | 0.400 | 0.172 | yes | 0.75 | 0.50 |
| boxes | truth | 0 | 0.328 | 0.116 | yes | – | – |
| boxes | **SLAM** | 0 | 0.331 | 0.164 | yes | 0.75 | 0.45 |
| wall | truth | 0 | 0.250 | 0.471 | no | – | – |
| wall | **SLAM** | 0 | 0.079 | 0.062 | yes | 0.81 | 0.77 |
| slalom | truth | 0 | 0.338 | 0.162 | yes | – | – |
| slalom | **SLAM** | 0 | 0.271 | 0.197 | yes | 0.69 | 0.65 |

The online map matches the ground-truth-map runs: collision-free on all four
scenes, all reaching the goal. `wall` is the tightest under SLAM
(clearance 0.079 m) because the long wall is only discovered on approach.

## Dashboard

The hero page's right column shows **"SLAM map · what the drone knows"**
(`web/src/slamviz.js`): the occupancy grid drawn top-down (grey unknown, dark
free, amber occupied), the **ground-truth obstacle outline** overlaid for
comparison, the estimated trajectory so far, and the drone marker. Below it are
the live metrics (`explored`, `surface`, `IoU`, `replans`, `goal path`). It
animates with the playback cursor, so you watch the map fill in. A header **map**
selector switches between `SLAM` and `truth`.

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
