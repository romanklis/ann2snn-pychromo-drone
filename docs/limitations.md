# Limitations and claim status

This page states, per major claim, whether it is established and how. Labels:

- **[PROVEN]** — a mathematical proof is given.
- **[DERIVED FROM IMPLEMENTATION]** — a direct consequence of the code, checkable
  by reading it or by a finite-difference/numeric test.
- **[APPROXIMATION]** — a deliberate simplification, with the gap described.
- **[EMPIRICALLY VALIDATED]** — supported by simulation results, scoped to the
  scenarios run.
- **[NOT ESTABLISHED]** — not shown; do not rely on it.

## Established

- **[PROVEN]** Nominal DS stability (obstacle-free, uncapped): the horizontal
  linearisation has eigenvalues `-g_r ± i·g_c` with `g_r>0`, so `V=½‖e‖²` gives
  `V̇ = -g_r(e_x²+e_y²) - g_climb e_z² < 0`. See
  [field.md §3](field.md#3-nominal-ds-analytic-kept) and
  `tests/test_ds_stability.py`. This is stability of the *nominal DS only* — it
  is **not** a proof for the saturated nonlinear drone closed loop
  (**[NOT ESTABLISHED]**).
- **[DERIVED FROM IMPLEMENTATION]** The normalized-RBF gradient
  `∇_x U` is the exact partial derivative with centres frozen; verified by
  finite differences in `tests/test_field.py`.
- **[DERIVED FROM IMPLEMENTATION]** The teacher potential
  `U* = Σ ½ max(0, r_infl − d)²` has `∇U* = −Σ(r_infl − d)n_out`; the field sign
  pushes outward.
- **[DERIVED FROM IMPLEMENTATION]** Mixer, hover speed `√(mg/4C_T)`, hover
  current, battery OCV endpoints, and parameter counts.

## Approximations

- **[APPROXIMATION]** The SNN is a **rate-coded finite-window IF approximation**
  of the ANN, not a lossless transfer. Measured on the deployment sequence
  (`tools/snn_transfer_eval.py`): for the connectome SNN Pearson ≈0.92 but
  gain ≈0.56 and R²<0, with ~92 % of neurons silent; for the **field SNN** the
  fidelity is lower still (Pearson ≈0.45, cosine ≈0.40 even after the readout
  calibration), so its residual is coarse. Exact ANN/SNN equivalence is
  **[NOT ESTABLISHED]**.
- **[APPROXIMATION]** The field target is **gradient-matched** to the teacher
  (`F* = −∇U*`) but value-only terms are gone; a full Sobolev/velocity loss is
  future work.
- **[APPROXIMATION]** Global guidance is an **A\* planner**. With
  `map_source="truth"` it reads the privileged scene; with `map_source="slam"`
  it reads a live **SLAM-lite** occupancy map (UKF localization + LiDAR mapping,
  **no loop closure**, 2.5-D, 3 m range). The basis centres are still anchored to
  the nominal-DS preview (not the path). See [slam.md](slam.md).
- **[APPROXIMATION]** Sensing/geometry: obstacles are 2.5-D, the clearance metric
  treats the drone as a point, and the LiDAR beams are world-fixed (not
  body-yaw-relative).
- **[APPROXIMATION]** The estimator is simpler than the plant: it models the
  accelerometer as thrust + in-plane H-drag + fuselage drag (residual mismatch:
  rotor in-flow loss, `thrust_min`), uses a 50 ms rotor lag vs the plant's
  ≈2–4 ms, truth-plus-white-noise INS, and GPS `σ = 0.25 m` comparable to the
  0.30 m tolerance. See
  [estimation.md §6](estimation.md#6-estimator--plant-model-mismatches).

## Empirical (scoped to the scenarios run)

- **[EMPIRICALLY VALIDATED]** The structured field controllers are
  collision-free and reach the goal on the evaluated scene(s). Phrase as
  "collision-free in the evaluated scenarios", never "formally collision-free"
  (**[NOT ESTABLISHED]** as a guarantee).
- **[EMPIRICALLY VALIDATED]** The UKF tracks the hidden plant to sub-0.5 m RMSE
  on the shipped run. The default bias initialisation is **privileged** (seeded
  with the true biases); the zero-bias case is tested for stability.
- **[EMPIRICALLY VALIDATED]** The SNN output correlates with the ANN rate code on
  the deployment sequence (`tools/snn_transfer_eval.py`).

## Not established

- **[NOT ESTABLISHED]** Global closed-loop stability of the saturated drone with
  the DS + modulation + impedance stack; the time-scale separation is weak.
- **[NOT ESTABLISHED]** A formal collision-free / CBF-style safety guarantee for
  the obstacle modulation.
- **[NOT ESTABLISHED]** Any safety guarantee from the SNN. There is **no
  independent safety layer** (`docs/control.md §8`).
- **[NOT ESTABLISHED]** That the RBF field representation is more
  parameter-efficient than the end-to-end net. The comparison is confounded: the
  structured arms keep the analytic goal-reaching / modulation / impedance, so it
  shows that *structure helps*, not representation efficiency.
- **[NOT ESTABLISHED]** Multi-seed / multi-scene robustness; reported tables are
  single-scene, often single-seed.
- **[NOT ESTABLISHED]** A computational-efficiency advantage of the SNN; it is
  not claimed and not benchmarked.

## What this revision changed

- **Plan D — global guidance + local residual**: added an A\* planner
  (`planner.py`), a path-tracking guidance (`guidance.PathTracker`), retargeted
  the field to gradient matching, gated the residual to free space, added a
  goal-capture fade, and calibrated the SNN readout. `wall`/`slalom` collisions
  went to 0; the `field_*` bundle format is now `field@3`.
- Corrected the accelerometer model to a physically standard body-frame specific
  force (the old model double-counted gravity, reading ≈2g at hover).
- Corrected the battery-current, arm-length, rotation-axis and impedance
  equations in the docs; reconciled README claims with the code.
- Renamed the SNN to reflect that the transfer is rate-coded, not lossless.
- Added a covariance positive-definiteness guard and richer ANN↔SNN metrics.
