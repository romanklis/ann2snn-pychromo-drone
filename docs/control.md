# Control: where the controllers live and how they run

Source of truth: `src/drone6dof/control.py` (DS teacher + PID),
`src/drone6dof/connectome.py` (end-to-end ANN/SNN),
`src/drone6dof/field_control.py` (structured field),
`src/drone6dof/cli.py` (dispatch), `src/drone6dof/sim.py` (execution loop),
`src/drone6dof/benchmark.py` + `server/app.py` (dashboard execution).

## 1. The controller contract

Every controller exposes

```python
def act(self, state, ref) -> np.ndarray:  # (3,) acceleration demand u [m/s²]
```

It consumes an `Observation` (`state` = the fused estimate `[p, v]`, plus the
LiDAR scan) and a `RefPoint` (`pos`, `vel`, `acc`). It **never** receives the
hidden plant truth (`sim.py:126-129`). Controllers optionally expose `reset()`,
`last_telemetry`, and (for sensor/SNN arms) `last_scan`, `scan_angles`,
`last_spikes`.

## 2. Dispatch

`make_controller(name, scene, control_limit, weights_path)` selects the arm
(`cli.py:72-97`):

| Name | Class | File |
|---|---|---|
| `ds_guidance` | `DSGuidanceController` | `control.py` |
| `pid` | `ClassicalPDController` | `control.py` |
| `ann`, `snn` | `ConnectomeController` | `connectome.py` |
| `field_ann`, `field_snn` | `FieldDSController` | `field_control.py` |

The `ann`/`snn` arms load `weights/quad6dof_connectome.npz`; the field arms load
`weights/quad6dof_field.npz`, both validated against the configuration
fingerprint (see [architecture.md](architecture.md#5-configuration-and-the-fingerprint-guard)).

## 3. `ds_guidance` — analytic teacher

Billard-modulated dynamical-system guidance followed by a passive impedance
(`control.py:81-158`). The nominal DS is the same radial/curl/climb field as
[field.md](field.md#3-nominal-ds-analytic-kept), with the example's teacher gains.

For each obstacle whose reference distance `d_ref = sdf + core_offset` is inside
its influence radius, the field is modulated using the closest-point outward
normal `n` and tangent `t` (flipped so `v_des·t ≥ 0`):

$$
\gamma = \left(\frac{d_\text{ref}}{d_\text{dead}}\right)^2,\quad
\lambda_r = 1 - \frac{1}{\max(\gamma,0.05)},\quad
\lambda_t = 1 + \frac{0.8}{\max(\gamma,0.05)},
$$
$$
s = \mathrm{clip}\!\left(\frac{r_\text{infl}-d_\text{ref}}{r_\text{infl}-d_\text{dead}},0,1\right),\quad
w = \sin^2\!\left(\frac{\pi}{2}s\right),
$$
$$
v_\text{des} \leftarrow v_\text{des} + w\big[(\lambda_r-1)(v_\text{des}\cdot n)n + (\lambda_t-1)(v_\text{des}\cdot t)t\big],
$$

composed over the scene's obstacles with the same law for boxes and cylinders
(`control.py:92-119`). The acceleration demand comes from a passive impedance
toward $v_\text{des}$ with separate along/across damping
(`damping_along = 1.2 [1/s]`, `damping_across = 4.5 [1/s]`). With
$dv = v - v_\text{des}$, $\hat v = v_\text{des}/\lVert v_\text{des}\rVert$,
$a_\parallel = dv\cdot\hat v$ and $a_\perp = dv - a_\parallel\hat v$:

$$
u = -\frac{1}{m}\big(d_\parallel\, a_\parallel\, \hat v + d_\perp\, a_\perp\big)\quad [\text{m/s}^2],
$$

clipped to `action_limit` (`control.py:121-129,157-158`).

## 4. `pid` — classical baseline

PD with acceleration feed-forward (`control.py:168-215`). With $e = p-r$ and
$\dot e = v-\dot r$:

$$
u = -\!\left(\omega_n^2 e + 2\zeta\omega_n \dot e\right)/G + a_\text{ff}/G,\qquad
k_p = \frac{\omega_n^2}{|G|},\quad k_d = \frac{2\zeta\omega_n}{|G|},
$$

defaults `omega_n = 3.5`, `zeta = 0.85`, `plant_gain = 1`. It is deliberately
obstacle-blind, so it flies straight into the pillar — the negative control.
Note it is a **PD** law (no integral term), despite the `pid` name kept for
parity with upstream.

## 5. `ann` / `snn` — end-to-end connectome

Both consume the 46-D sensor-conditioned policy input
(`policy_input_with_scan`, [sensing.md](sensing.md#4-policy-feature-vector)) and
return a 3-D acceleration demand (`connectome.py:250-265`). The feed-forward
reference acceleration comes from `ReferenceAccelEstimator`, a 3-point second
difference of `x - e`; for a constant goal it settles to zero
(`policy.py:41-72`).

- `ann` — sparse recurrent ReLU network, 1000 neurons / fan-in 40 / 3 recurrent
  steps (`connectome.py:102-135`).
- `snn` — rate-coded integrate-and-fire **approximation** of the ANN, 10
  micro-steps, `v_th = 1.0` (`connectome.py:138-190`); see
  [field.md §7](field.md#7-what-the-ann-field-model-is-and-what-the-transferred-snn-is).

The full network description is in
[field.md §7](field.md#7-what-the-ann-field-model-is-and-what-the-transferred-snn-is).

## 6. `field_ann` / `field_snn` — structured field (global planner + local residual)

Global guidance is an A\* path-tracking DS (`planner.py` → `reference.path_reference`
→ `guidance.PathTracker`, `v_nom = ṙ_ref + k(r_ref − p)`); the LiDAR network
supplies a **gated local residual** (`field_control.py:106-170`):

$$
a = \mathrm{clip}\big(\text{net}(\phi), 0, \text{field\_limit}\big),\quad
F = -\nabla U_a,\quad
v_\text{des} = \mathrm{modulate}\big(v_\text{nom},\ \rho\,F\big),
$$

with the gate $\rho = \rho_\text{dist}\cdot\rho_\text{goal}$ (distance gate ×
goal-capture fade) so the residual vanishes in free space and near the goal.
Training labels are the teacher's modulated flow on the planner's `v_nom`
(`guidance.SupervisorController`, privileged).

**Terminology.** `F_obs = −∇U_obs` is an **obstacle modulation (repulsion)
vector**, not a physical Newtonian force: it reshapes the desired velocity. The
word "force" is reserved for the plant dynamics in [physics.md](physics.md).

## 7. Execution

### Simulation loop
`Simulation` wires one backend + one controller + one reference
(`sim.py:39-90`) and steps frame by frame (`sim.py:124-163`):

1. `action = controller.act(Observation(state=estimate), reference.at(k))`
2. `backend.step(action, dt, limit, gain, damping, disturbance)` — the plant runs
   its 500 Hz inner substeps ([physics.md](physics.md#7-inner-substepping)).
3. `SensorSuite.measure(k+1, truth)` → `ErrorStateUKF.step(meas, rotor_target)`
   ([estimation.md](estimation.md)).
4. Record, advance `k`.

`history` holds state, rpy, command, telemetry, goal distance, clearance, spikes
(first 200 neurons), raw scans, estimate and estimate error (`sim.py:108-119`,
`sim.py:193-217`). `metrics()`, `to_csv()` and `to_npz()` export the run
(`sim.py:241-323`).

### CLI
`python -m drone6dof --controller …` builds the scene, backend and controller,
then either renders with `ChronoViz` or runs headless (`cli.py:100-162`). The
PyChrono view calls `viz.render_loop(sim, …)`, which steps the simulation once
per draw and syncs the kinematic drone/rotors/trail (`viz.py:421-458`).

### Dashboard
`POST /api/benchmark {"controllers": [...], "scene": …, "goal": …, "steps": …}`
→ `run_benchmark` (`server/app.py:93-99`, `benchmark.py:259-341`). Each
controller runs on a **fresh plant** on the same scene, so runs are directly
comparable (`benchmark.py:195-207`). The report carries per-controller
trajectory, attitude, command, telemetry, spikes, scans/angles, estimate and
metrics, plus `stats` with the SNN−ANN delta and a ranking (`benchmark.py:234-340`).

## 8. Safety layer

There is **no independent safety layer**. Safety in these runs comes only from
(i) clipping the acceleration demand to `action_limit`, (ii) the plant's
rotor/thrust/current saturation ([physics.md](physics.md)), and (iii) the
operator bounds on the interactive goal. The learned `snn`/`field_snn` provide
**no** safety guarantee — they are one learned component inside the structured
loop. An explicit limiter / recovery layer that can override or blend the
learned output is future work (see [limitations.md](limitations.md)).

## 9. Limits and gains

| Quantity | Value | Source |
|---|---|---|
| Outer timestep `dt` | 0.02 s (50 Hz) | `params.py:20` |
| Inner substep rate | 500 Hz | `params.py:81` |
| `CONTROL_LIMIT` (command space) | 12 m/s² | `config.py:47` |
| Plant thrust envelope | ~7.8 m/s² at full rotor efficiency | `config.py:45-46` |
| `GOAL_TOLERANCE` | 0.30 m | `config.py:52` |
| Teacher gains | `speed_cap=1.4, ds_radial_gain=0.6` | `config.py:66` |
| PID | `omega_n=3.5, zeta=0.85` | `control.py:179-180` |
| Action limits | teacher 20, others `control_limit` | `control.py:57,183`, `cli.py:74-96` |

`CONTROL_LIMIT` (12 m/s²) exceeds the instantaneous thrust envelope
(≈7.8 m/s²), so demands can saturate; there is no explicit anti-windup. The DS
bandwidth, impedance poles (1.2 and 4.5 s⁻¹) and attitude loop (≈6 rad/s) are
only a few times apart, so the time-scale-separation assumption is weak.

## 10. Schematics

- [assets/schematics/control-execution.svg](assets/schematics/control-execution.svg)
  — `make_controller` dispatch plus the per-frame `plant → sensors → UKF →
  controller` loop.
