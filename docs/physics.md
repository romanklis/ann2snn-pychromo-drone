# Physics: equations of motion and the plant model

Source of truth: `src/drone6dof/plant.py` (the 6-DoF plant), `src/drone6dof/so3.py`
(rotation utilities) and `src/drone6dof/params.py` (parameters). The plant is a
verbatim numpy port of the ANN2SNN `sim_engine/plants/quad6dof.py`; only the torch
boundary was replaced with numpy arrays.

## 1. State and conventions

The observable state handed around the rest of the stack is the translational
state

$$
x = \begin{bmatrix} p \\ v \end{bmatrix} \in \mathbb{R}^6,
$$

while the attitude, body rates, four rotor speeds, rotor currents and battery
state of charge live inside `Quad6DoF` as hidden state (`plant.py:9-17`).

The command is an **acceleration demand** $\bar u \in \mathbb{R}^3$ (m/s²) that
**excludes gravity**. The plant closes the loop with a desired force

$$
F_\text{des} = m\,(\bar u + g\,\hat z), \qquad g = 9.81,\ \hat z = (0,0,1),
$$

so hover is $\bar u = 0$ (`plant.py:14-16,131`). The demand is clipped component
wise to `limit` (the policy command space, `CONTROL_LIMIT = 12 m/s²`), and a
`gain` ("rotor efficiency", `authority`) scales the achieved rotor thrust.

World frame is z-up (`so3.py`, `viz.py:129-131`).

## 2. Desired attitude (outer loop)

The required thrust direction is the normalized desired force,

$$
\hat z_{b,\text{des}} = \frac{F_\text{des}}{\lVert F_\text{des}\rVert},
$$

and the remaining two body axes are built from a heading direction `cam`
(`plant.py:133-171`). When the example supplies `heading_target` (the goal, from
`NumpyPlantBackend`), the look-at direction blends a fixed forward
$(1,0,0)$ with the direction to the target using a smoothstep of the remaining
distance between `heading_d_min = 0.15 m` and `heading_d_max = 0.60 m`:

$$
\text{look} = \frac{r_\text{target}-p}{\lVert r_\text{target}-p\rVert},\quad
s = \mathrm{clamp}\!\left(\frac{d-d_\text{min}}{d_\text{max}-d_\text{min}},0,1\right),\quad
w_b = 3s^2-2s^3,
$$
$$
\text{cam} = \frac{w_b\,\text{look} + (1-w_b)(1,0,0)}{\lVert \cdot \rVert}.
$$

Otherwise `cam = (1,0,0)`. The columns of the desired rotation (the body axes) are

$$
y_b = \frac{\hat z_{b,\text{des}} \times \text{cam}}{\lVert \cdot \rVert},\qquad
x_b = y_b \times \hat z_{b,\text{des}},\qquad
R_d = \begin{bmatrix} x_b & y_b & \hat z_{b,\text{des}} \end{bmatrix}.
$$

## 3. Attitude control (SO(3))

The rotation error follows Lee et al. (`plant.py:173-200`):

$$
e_R = \tfrac12\,\mathrm{vee}\!\left(R_d^\top R - R^\top R_d\right).
$$

A proportional torque per axis (yaw uses a smaller gain `att_kp_yaw`) is
low-pass filtered at `att_filter_tau`:

$$
\tau_{\text{att},i} \mathrel{+}= \left(-k_{p,i}\,e_{R,i} - \tau_{\text{att},i}\right)\left(1 - e^{-h/\tau_f}\right).
$$

The final torque adds a rate term plus the gyroscopic coupling
$\omega \times J\omega$:

$$
\tau_i = \tau_{\text{att},i} - k_{d,i}\left(\omega_i + \omega_{\text{ref}}\,e_{R,i}\right) + (\omega \times J\omega)_i,
$$

with the yaw rate term saturated to `±tau_yaw_limit` (`plant.py:188-200`).

## 4. Mixer, rotor targets and electrical dynamics

The scalar command axis is the **demanded achieved collective thrust**

$$
T_\text{demand} = F_\text{des} \cdot R[:,2],
\qquad
T_\text{rotor} = \mathrm{clip}\!\left(\frac{T_\text{demand}}{\text{authority}},\ T_\text{min}^{\text{cmd}},\ s_\text{max}\,m g\right).
$$

A quad-X mixer maps the wrench $[T, \tau_\text{roll}, \tau_\text{pitch}, \tau_\text{yaw}]$
to four rotor thrust targets through the matrix $B$ (`plant.py:54-66,213-217`):

$$
B = \begin{bmatrix}
1 & 1 & 1 & 1\\
-d_a & d_a & d_a & -d_a\\
-d_a & d_a & -d_a & d_a\\
-c_r & -c_r & c_r & c_r
\end{bmatrix},\quad
d_a = \frac{d}{\sqrt 2},\quad c_r = \frac{C_Q}{C_T},
$$

with $B^{-1}$ used to invert it. A `tanh` saturation limits the mix, then the
rotor speed is recovered as $\omega_\text{target} = \sqrt{\max(\text{target},0)/C_T}$.

The propulsion electronics use an exact zero-order-hold current update and an
explicit-Euler mechanical loop (`plant.py:219-247`):

$$
i \mathrel{+}= (i_\text{target}-i)\left(1-e^{-h/\tau_e}\right), \qquad
i_\text{target} = \mathrm{clip}\!\left(\frac{d_m \omega_\text{target} + C_Q \omega_\text{target}^2 + 0.003(\omega_\text{target}-\omega_m)}{K_t},0,i_\text{max}\right),
$$

$$
\omega_m \mathrel{+}= \frac{K_t i - (d_m \omega_m + C_Q \omega_m^2)}{J_m}\,h,
\qquad \omega_m \in [\omega_\text{min},\ \omega_\text{max}^v].
$$

The speed ceiling comes from the battery (`plant.py:226-247`):

$$
i_\text{tot} = \sum_k i_k + I_\text{av},\quad
v_\text{ocv} = V_\text{min} + 3\,\mathrm{soc} + 0.6\,\mathrm{soc}^2,\quad
v_\text{term} = v_\text{ocv} - i_\text{tot} R_\text{int},
$$
$$
\mathrm{soc} \mathrel{-}= \frac{i_\text{tot}}{Q_\text{Ah}\cdot3600}\,h,\qquad
\omega_\text{max}^v = \max\!\left(\frac{v_\text{term} - i R_\text{coil}}{K_e},\ \omega_\text{floor}^{v}\right).
$$

## 5. Rotor and fuselage aerodynamics

With body-frame velocity $v_b = R^\top v$ (`plant.py:249-256`):

$$
f_{\text{rotor},k} = \max\!\left(C_T \omega_{m,k}^2 - k_{vz}\,\omega_{m,k}\,v_{b,z},\ f_{\text{min}}\right)\cdot\text{authority},
$$

$$
F_H = -k_h \Big(\textstyle\sum_k \omega_{m,k}\Big)\,(v_{b,x}, v_{b,y}, 0),
\qquad
F_\text{drag} = -\tfrac12 \rho\, C_{D,\text{body}} \odot (v_b \odot |v_b|).
$$

The thrust/rotor-drag wrench is $\text{wrench} = B f_\text{rotor}$, and the torque
model subtracts rotor aerodynamic damping $D_\text{rot} \odot \omega$:

$$
\tau_\text{aero} = \text{wrench}[1{:}4] - D_\text{rot}\odot\omega,\qquad
\tau_\text{gyro} = -\,\omega \times J\omega.
$$

## 6. Rigid-body equations of motion

Body rates and attitude are integrated over the substep $h$ (`plant.py:258-277`):

$$
\dot\omega = J^{-1}(\tau_\text{aero} + \tau_\text{gyro}),\qquad
R \leftarrow R\,\exp\!\big([\omega]_\times h\big),
$$

where $\exp([\omega]_\times h)$ is the exact Rodrigues rotation for a constant
body rate (`so3.py:23-29`). The net world force and translational integration are

$$
F_\text{net} = T_\text{actual}\,R[:,2] + R(F_H + F_\text{drag}) + m(0,0,-g) + F_\text{dist} - m\,\text{damping}\,v,
$$
$$
v \leftarrow v + \frac{F_\text{net}}{m}\,h,\qquad p \leftarrow p + v h,
$$

with $T_\text{actual} = \text{wrench}[0]$. The `damping` term is an optional extra
linear drag (embodiment), and `disturbance` is an optional per-step world-force
injection (both default to zero). After the substeps, $R$ is re-orthonormalized by
an SVD projection onto SO(3) to remove long-horizon drift (`plant.py:314`,
`so3.py:32-40`).

## 7. Inner substepping

One control frame is `dt = 0.02 s` (50 Hz, `params.py:20`). The plant integrates

$$
n = \max\!\left(1,\ \mathrm{round}(dt\cdot f_\text{inner})\right),\qquad h = \frac{dt}{n},
$$

at `inner_hz = 500 Hz` (validated; `params.py:81`, `plant.py:124-129`). The
outer command is zero-order held across the $n$ substeps; the motor/attitude
state evolves inside them. `Simulation.step` = one 20 ms frame.

## 8. Parameters (`params.py`)

| Symbol / field | Value | Meaning |
|---|---|---|
| `m` | 0.5 kg | mass |
| `g` | 9.81 m/s² | gravity |
| `J` | (2.3e-3, 2.3e-3, 4.0e-3) kg·m² | body inertia |
| `d` | 0.15 m | centre-to-motor arm length |
| `C_T` | 1.5e-5 | rotor thrust coefficient |
| `C_Q` | 2.5e-7 | rotor drag-torque coefficient |
| `k_vz` | 1.4e-4 | axial climb inflow penalty |
| `k_h` | 1.6e-6 | rotor in-plane drag |
| `rho` | 1.225 | air density |
| `C_D_body` | (0.022, 0.022, 0.048) | fuselage drag |
| `J_m`, `K_t`, `K_e`, `d_m`, `R_coil`, `tau_e` | 1.2e-5, 0.015, 0.015, 1.5e-5, 0.18, 0.002 | motor |
| `capacity_Ah`, `R_int`, `V_min_ocv`, `V_max_ocv`, `I_avionics`, `soc0` | 1.3, 0.035, 13.2, 16.8, 0.25, 0.95 | 4S battery |
| `thrust_cmd_min`, `thrust_cmd_max_scale`, `tau_yaw_limit`, `current_max`, `omega_min` | 1.0, 2.0, 0.015, 6.5, 10.0 | limits |
| `att_kp`, `att_kp_yaw`, `att_kd`, `att_kd_yaw`, `att_rate_ref`, `att_filter_tau` | 0.08, 0.015, 0.022, 0.008, 6.0, 0.015 | autopilot |
| `inner_hz` | 500 | inner substep rate |
| `initial_omega_m`, `initial_current` | 286, 1.65 | reset values |

The hover rotor speed is $\omega_\text{hover} = \sqrt{mg/(4C_T)}$
(`params.py:89-92`).

## 9. Telemetry

Once per frame the plant reports (`plant.py:280-312`): `g_force`, `thrust_n`,
`power_w`, `soc_pct`, `v_term`, `v_ocv`, `advance_ratio`, `omega_mean`,
`current_mean`, `tau_aero_norm`, `tau_gyro_norm`, `inflow_loss`,
`thrust_demand_n`, `thrust_achieved_n`, `thrust_margin_frac`,
`thrust_saturated`, and per-rotor `rpm{1..4}` / `current{1..4}`. These flow into
the CSV/NPZ export and the dashboard's telemetry charts.

## 10. Specific force (ideal accelerometer)

The plant also reports the body-frame **specific force** an ideal accelerometer
would measure: the non-gravitational force per unit mass,

$$
f_b = \frac{1}{m} R^\top \big(F_\text{thrust} + R(F_H + F_\text{drag}) + F_\text{dist} + F_\text{damp}\big)
= \frac{T_\text{actual}}{m}\hat z + \frac{F_H + F_\text{drag}}{m} + \dots
$$

(`plant.py`, telemetry `sf_bx/sf_by/sf_bz`). Gravity is excluded — an
accelerometer cannot measure it — so a level hover reads `+g` along body z
(≈9.81 m/s²), not `2g`. This is the sensor model consumed by `sensors.py`; the
estimator models the same thrust + H-drag + fuselage-drag terms (residual
mismatch: rotor in-flow loss) — see
[estimation.md](estimation.md#6-estimator--plant-model-mismatches).

## 11. Modelling choices and known omissions

- The motor **reaction torque** `J_m ω̇` on the body is not included in the
  rigid-body torque (only the rotor aerodynamic drag `d_m ω_m + C_Q ω_m²`).
- `C_D_body` is effectively a drag-area (`CdA`); the reference area is folded in.
- Obstacles are **2.5-D** (horizontal cross-section with a stored height). The
  clearance metric is horizontal and treats the drone as a **point**, so it does
  not account for arm/prop radius.
- The battery-voltage speed ceiling `omega_max_v` (≈1000 rad/s) is far above the
  hover speed (≈286 rad/s) and effectively never binds in these runs.
- The physics here is a numpy model; PyChrono is only a visualization layer
  (see [architecture.md](architecture.md)).

## 12. Schematics

- [assets/schematics/plant-control-loop.svg](assets/schematics/plant-control-loop.svg)
  — desired attitude → mixer → motor/battery → rigid body → telemetry.
