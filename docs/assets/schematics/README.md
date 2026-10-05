# Schematics

Each file here is a **placeholder**: a labelled dark-theme box describing what
the final artwork should show. Replace the SVG contents (keep the filename) and
every doc link keeps working. If you switch to raster art, add the new file and
update the single reference in the page listed below.

Common style: dark background `#0e141d`, panel `#111823`, border `#22303c`,
text `#c9d1d9`, accent `#22b8a6`, highlight `#f72585`, secondary `#4f8cff`.

| File | Referenced from | What it must show |
|---|---|---|
| `architecture.svg` | [index.md](../../index.md), [architecture.md](../../architecture.md) | The three runtime layers (entry points → core → surfaces), the estimator-in-the-loop chain `plant → sensors → UKF → controller`, the offline torch training path, and where the weight bundles enter. |
| `plant-control-loop.svg` | [physics.md](../../physics.md) | The plant signal chain: `u` → `F_des`/`R_d` → SO(3) PD → mixer `B⁻¹`/tanh → motor + battery → rotor aero → rigid body → integrate (500 Hz substeps). |
| `ukf.svg` | [estimation.md](../../estimation.md) | The 23-D nominal / 22-D error state, sigma-point generation, the predict model (thrust + kinematics + attitude + rotor lag), and the stacked-measurement update (`S`, `P_xz`, `K`, NIS). |
| `lidar.svg` | [sensing.md](../../sensing.md) | A drone with a horizontal beam fan, an obstacle with the nearest return `i*`, and the five cues (min range, bearing, slope, curvature). |
| `field.svg` | [field.md](../../field.md) | The nominal-DS preview with trajectory-anchored RBF centres, the learned `U_obs`/`F_obs=−∇U`, and the Billard modulation geometry (`n`, `t`, `v_nom` → `v_des`). |
| `control-execution.svg` | [control.md](../../control.md) | The `make_controller` dispatch table for the six arms, and the per-frame loop `act → plant.step → sensors.measure → UKF.step`, plus the dashboard batch entry point. |

The editable captions live in the SVG `<text>` elements, so a final figure can be
produced directly in a vector editor from these files.
