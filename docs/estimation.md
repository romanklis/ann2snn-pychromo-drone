# Estimation: sensor suite and the error-state UKF

Source of truth: `src/drone6dof/sensors.py` (measurement models),
`src/drone6dof/estimator.py` (the filter) and `src/drone6dof/config.py:71-75`
(configuration).

The plant is the **hidden truth**. Every controller consumes only a fused state
estimate: the loop is `plant → sensor suite → UKF → controller`
(`sim.py:140-158`). The learned controllers are distilled with the estimator in
the loop, so training matches deployment.

## 1. Sensor suite

`SensorSuite.measure(step, truth)` produces a `Measurement` with rate-limited
channels and realistic failure modes (`sensors.py:122-167`). Rates are converted
to integer periods in frames (`sensors.py:53-54`).

| Channel | Rate (default) | Model (default) |
|---|---|---|
| IMU specific force | 50 Hz | $\tilde f_b = R^\top(a - g) + b_a + \mathcal N(0,\sigma_a^2)$, outliers at 1 % scaled ×8 |
| IMU gyro | 50 Hz | $\tilde\omega = \omega + b_g + \mathcal N(0,\sigma_g^2)$, outliers at 1 % |
| GPS position | 10 Hz | $\tilde p = p + \mathcal N(0,\sigma_p^2)$, 5 % dropout |
| INS velocity | 50 Hz | $\tilde v = v + \mathcal N(0,\sigma_v^2)$ |
| INS attitude | 50 Hz | $\widetilde{\text{rpy}} = \text{rpy}(R) + \mathcal N(0,\sigma_\theta^2)$ |
| Barometer altitude | 25 Hz | $\tilde z = p_z + \mathcal N(0,\sigma_b^2)$ |
| Compass heading | 25 Hz | $\tilde\psi = \text{yaw}(R) + \mathcal N(0,\sigma_m^2)$ |
| LiDAR | 50 Hz | noisy range scan (see [sensing.md](sensing.md)) |

Defaults (`sensors.py:30-51`): $\sigma_p = 0.25$ m, $\sigma_v = 0.08$,
$\sigma_\theta = 0.02$ rad, $\sigma_a = 0.12$ m/s², $\sigma_g = 0.015$ rad/s,
$\sigma_b = 0.25$ m, $\sigma_m = 0.04$ rad; accelerometer bias
$(0.05,-0.04,0.06)$, gyro bias $(0.012,-0.010,0.008)$; seeded RNG for
determinism. The IMU specific force uses the true thrust
$T = C_T\sum_k \omega_{m,k}^2$ (`sensors.py:135-146`).

The controller receives an `Observation` carrying the estimate plus the LiDAR
scan (`sensors.py:85-93`).

## 2. Error-state formulation

`ErrorStateUKF` maintains a nominal 23-vector and a **22-dimensional error
covariance** (`estimator.py:80-83,151-184`):

$$
x = \big[\,p(3),\ v(3),\ q(4),\ \omega(3),\ \omega_m(4),\ b_a(3),\ b_g(3)\,\big],
\qquad \delta x \in \mathbb{R}^{22},
$$

with the attitude error a 3-vector (rotation vector). The boxplus and diff
operators are multiplicative on the quaternion and additive on the rest:

$$
x \boxplus \delta = \big[\,p+\delta_p,\ \dots,\ q\otimes\exp(\tfrac12\delta_\theta),\ \dots\,\big],\qquad
\delta_\theta = \mathrm{Log}\!\left(q_0^{-1}\otimes q_1\right).
$$

Keeping the quaternion out of the covariance avoids the unit-norm singularity
(`estimator.py:2-8`).

## 3. Sigma points

With $\alpha=0.3$, $\beta=2$, $\kappa=0$ and $n = 22$ (`estimator.py:115-123`):

$$
\lambda = \alpha^2(n+\kappa) - n,\qquad
\chi_0 = x,\quad
\chi_{i} = x \boxplus \big[(n+\lambda)P\big]^{1/2}_i,\quad
\chi_{i+n} = x \boxplus -\big[(n+\lambda)P\big]^{1/2}_i,
$$

$$
W_0^m = \frac{\lambda}{n+\lambda},\quad
W_0^c = W_0^m + (1-\alpha^2+\beta),\quad
W_i^m = W_i^c = \frac{1}{2(n+\lambda)}.
$$

The matrix square root uses Cholesky with an eigen-decomposition fallback
(`estimator.py:237-241`).

## 4. Predict (process model)

The process model is a deliberately **simplified** nonlinear quad driven by the
applied rotor target — never the full plant — so $\sigma$-point propagation stays
cheap (`estimator.py:203-219`):

$$
R = R(q),\qquad T = C_T \textstyle\sum_k \omega_{m,k}^2,\qquad
a_w = \frac{T}{m}\,R[:,2] + g,
$$
$$
p \leftarrow p + v\,\Delta t + \tfrac12 a_w \Delta t^2,\qquad
v \leftarrow v + a_w \Delta t,\qquad
q \leftarrow q \otimes \exp(\omega\,\Delta t),
$$
$$
\omega \leftarrow \omega + J^{-1}\!\left(\text{wrench}[1{:}4] - \omega \times J\omega\right)\Delta t,\quad
\omega_m \leftarrow \mathrm{clip}\!\left(\omega_m + (\omega_\text{target}-\omega_m)\min(1,\tfrac{\Delta t}{\tau_r}),\ 1,\ \omega_{m,\max}\right),
$$

where `wrench = B (C_T ω_m²)`, `rotor_tau = 0.05 s`, `rotor_omega_max = 1400 rad/s`
(`config.py:75`). After propagation, the mean is formed with a weighted
quaternion composition about the nominal attitude, and

$$
P \leftarrow Q + \sum_i W_i^c\,(\chi_i \ominus \bar x)(\chi_i \ominus \bar x)^\top,
$$

symmetric-ized (`estimator.py:246-265`).

## 5. Update

All available measurements are stacked into one update. The measurement
functions match the sensor models (`estimator.py:268-296`):

$$
h_\text{gps}=p,\quad
h_\text{acc}=\frac{T}{m}(0,0,1) - R^\top g + b_a,\quad
h_\text{gyro}=\omega+b_g,\quad
h_\text{ins,v}=v,\quad
h_\text{ins,att}=\text{rpy}(R),\quad
h_\text{baro}=p_z,\quad
h_\text{mag}=\text{yaw}(R).
$$

Then (`estimator.py:307-348`):

$$
S = R_\text{noise} + \sum_i W_i^c (\chi^z_i-\bar z)(\chi^z_i-\bar z)^\top,\qquad
P_{xz} = \sum_i W_i^c (\chi_i-\bar x)(\chi^z_i-\bar z)^\top,
$$
$$
K = P_{xz}S^{-1},\qquad
x \leftarrow x \boxplus K(z-\bar z),\qquad
P \leftarrow P - K S K^\top,
$$

with per-channel noise blocks $\sigma^2 I$ (`estimator.py:272-296`). The NIS
$\nu = (z-\bar z)^\top S^{-1}(z-\bar z)$ is recorded as telemetry. There is
deliberately **no batch gate**: rejecting a whole frame on one outlier would
discard every sensor that frame, so robustness comes from the noise model (and
per-sensor gating if needed) — see the comment at `estimator.py:335-336`.

Initial covariance and process noise are `estimator.py:136-145` and
`estimator.py:221-230` (position 1e-4, velocity 5e-3, attitude 1e-4, rates
2e-3, rotor 1e-1, accel bias 1e-6, gyro bias 1e-7, all diagonal).

## 6. Honest limitation

Estimate-only control is harder than a truth-based demo. With the current
estimator the teacher and the ANN clear the pillar and approach the goal, but the
closed loop orbits a little short of the 0.30 m tolerance and the SNN can graze
the pillar. Improving the estimator / DS tuning is follow-up work
(`README.md`, "State estimation").

## 7. Schematics

- [assets/schematics/ukf.svg](assets/schematics/ukf.svg) — nominal state +
  22-D error covariance, sigma-point predict, stacked-measurement update.
