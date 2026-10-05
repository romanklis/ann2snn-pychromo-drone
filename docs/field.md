# Potential field, and the structured ANN → SNN

Source of truth: `src/drone6dof/field.py` (basis, teacher, modulation),
`src/drone6dof/field_control.py` (`FieldDSController`),
`src/drone6dof/train_field.py` (distillation), `src/drone6dof/connectome.py`
(the network).

This page explains the **structured** architecture: the network does **not learn
the control law**. It predicts a compact obstacle-field representation; the
gradient, the modulation and the low-level control are analytic and never
learned. This is what distinguishes `field_ann`/`field_snn` from the end-to-end
`ann`/`snn` (see the contrast in §7).

## 1. Pipeline

```mermaid
flowchart LR
  L["LiDAR (k ranges + cues)"] --> N["small SNN/ANN"]
  N --> A["a ∈ R_+^K<br/>field coefficients"]
  A --> U["U_obs(x)=Σ a_k φ_k(x−c_k)"]
  U --> F["F_obs = −∇U_obs<br/>(analytic, conservative)"]
  F --> M["Billard-style DS modulation"]
  DS["nominal DS v_nom (stable)"] --> M
  M --> V["v_des"]
  V --> I["passive impedance → u"]
  I --> P["plant"]
```

The net outputs only the ``K`` non-negative coefficients ``a``; everything
downstream is analytic (`field.py:1-14`, `field_control.py:1-8`).

## 2. Field configuration

`FieldConfig` (`field.py:47-76`):

| Field | Default | Meaning |
|---|---|---|
| `h` | 8 | nominal-DS preview points |
| `m` | 3 | centres per preview point (on-axis + both lateral) |
| `K = h·m` | 24 | basis size |
| `preview_dt` | 0.05 s | preview integration step |
| `lateral` | 0.4 m | lateral offset for the side centres |
| `sigma` | 0.5 m | RBF bandwidth |
| `r_influence` | 1.5 m | teacher barrier influence radius |
| `field_limit` | 2.0 | coefficient/field clip |
| `f_ref` | 1.0 | field magnitude giving full modulation weight |
| `lam_t` | 0.8 | tangential stretch at full weight |
| `gains` | see §3 | nominal-DS gains |

## 3. Nominal DS (analytic, kept)

The stable nominal velocity field (`field.py:79-89`) is a radial attractor with
curl in the horizontal plane and a vertical climb term:

$$
v_x = -g_r e_x - g_c e_y,\qquad
v_y = -g_r e_y + g_c e_x,\qquad
v_z = -g_\text{climb} e_z,
\qquad
v_\text{nom} = v\,\min\!\left(\frac{v_\text{cap}}{\lVert v\rVert},1\right),
$$

with $e = p - r$, $g_r = 0.6$, $g_c = 1.35$, $g_\text{climb} = 0.45$,
$v_\text{cap} = 1.4$ (`field.py:38-44`). This matches the example teacher
(`config.py:66`).

The basis centres are **anchored to a short preview of this nominal DS**
(`field.py:92-126`): starting at the current position, integrate $h$ points at
`preview_dt`, take unit headings $v/\lVert v\rVert$, and per point place three
centres — the point itself and offsets $\pm\lambda_\text{lat}\,\hat\perp$ with
$\hat\perp = (-h_y, h_x, 0)$ (`field.py:118-126`). The field is therefore
trajectory-relative, not world-global.

## 4. Normalized Gaussian RBF basis

For a point $x$ and centres $c_k$ (horizontal components only),
`field.py:128-156`:

$$
\phi_k(x) = \exp\!\left(-\frac{\lVert x_{xy} - c_{k,xy}\rVert^2}{2\sigma^2}\right),\qquad
s = \sum_k \phi_k,\qquad
U_\text{obs}(x) = \sum_k a_k \frac{\phi_k(x)}{s(x)}.
$$

The obstacle vector field is the **analytic gradient** of the normalized basis
(quotient rule), never a separately learned quantity:

$$
\nabla U = \frac{\sum_k a_k \nabla\phi_k}{s} - \frac{\left(\sum_k a_k\phi_k\right)\left(\sum_k \nabla\phi_k\right)}{s^2},
\qquad
F_\text{obs} = -\nabla U,
$$

with $\partial\phi_k/\partial x = -\phi_k (x-c_k)/\sigma^2$. The vertical
component is zero — the field acts in the horizontal plane only
(`field.py:142-153`).

## 5. Privileged teacher potential (training target only)

The DS teacher is privileged (it uses true geometry). Its barrier potential and
gradient (`field.py:159-192`) are

$$
U^*(x) = \sum_{\text{obs}} \tfrac12 \max\!\big(0,\ r_\text{infl} - d(x)\big)^2,
\qquad
\nabla U^* = -\sum_{\text{obs}} \big(r_\text{infl} - d\big)\,n_\text{out}\,\mathbf 1[d < r_\text{infl}],
$$

where $d$ is the obstacle signed distance and $n_\text{out}$ the outward normal.
`teacher_coeffs` samples $U^*$ at the trajectory-anchored centres and returns the
value vector used as the **supervised target** (`field.py:187-192`).

## 6. Billard-style DS modulation

Given $v_\text{nom}$ and $F_\text{obs}$ (`field.py:195-210`):

$$
n = \frac{F_{\text{obs},xy}}{\lVert F_{\text{obs},xy}\rVert},\quad
t = (-n_y, n_x)\ \text{(flipped so } v_\text{nom}\cdot t \ge 0),\quad
w = \mathrm{clip}\!\left(\frac{\lVert F_{\text{obs}}\rVert}{f_\text{ref}},0,1\right),
$$

$$
v_\text{des} = v_\text{nom} - w\,(v_\text{nom}\cdot n)\,n + \lambda_t\,w\,(v_\text{nom}\cdot t)\,t.
$$

When $F_\text{obs}\to 0$ (no obstacle nearby) the modulation matrix tends to the
identity, so the goal attractor is preserved. `FieldDSController` then reuses the
same passive impedance as the teacher to produce the acceleration demand
(`field_control.py:132-139`).

## 7. What the ANN field model is, and what the transferred SNN is

### The field ANN
`FIELD_NET` is deliberately small: `n_neurons = 128`, fan-in `k = 8`,
`seed = 42` (`train_field.py:53-54`). It is a sparse recurrent ReLU network with
three recurrent evaluations per frame:

$$
h \leftarrow \mathrm{ReLU}\!\big(W_\text{in}x + W_\text{rec}h\big)\ \times 3,
\qquad
a = \mathrm{ReLU}\!\big(W_\text{out}h\big) \in \mathbb R_+^K,
$$

`train_field.py:166-172`. It is distilled by MSE to `teacher_coeffs` on a dataset
that combines closed-loop teacher rollouts (through the plant and UKF) with random
coverage states, across randomised box/cylinder scenes
(`build_field_dataset`, `train_field.py:94-129`). The network sees only the LiDAR
features; it learns the **field coefficients**, not forces or accelerations.

### The transferred SNN
The transfer is `LosslessConnectomeSNN` over the **same topology and weights**
(`connectome.py:138-190`): integrate-and-fire with `micro_steps = 10` per frame,
threshold `v_th = 1.0`, non-negative membrane, soft reset, and the command taken
as the mean output rate over the micro-steps. `last_spikes` records neurons that
fired at any micro-step within the frame. For `field_snn`, the spiking network
outputs the `K` field coefficients exactly like `field_ann` — the SNN never
outputs a control command (`field_control.py:85-98,148-151`).

### Contrast with the end-to-end connectome controllers

| | end-to-end `ann`/`snn` | structured `field_ann`/`field_snn` |
|---|---|---|
| Network output | acceleration demand (3) | field coefficients ($K=24$) |
| Network size | `N=1000`, `k=40` (~89 000 params) | `N=128`, `k=8` (~9 984 params) |
| Distillation target | teacher acceleration `u` | teacher potential $U^*$ at centres |
| What is learned | the control law | a compact obstacle-field |
| Gradient / modulation / impedance | learned implicitly | analytic, never learned |
| Training file | `train.py` | `train_field.py` |
| Bundle | `quad6dof_connectome.npz` | `quad6dof_field.npz` |

The end-to-end net carries the whole policy in its weights; the structured net
only parameterizes the field, which is why it needs ~9× fewer parameters while
being collision-free and goal-reaching (result table below).

## 8. Measured result

Estimate-only, `make compare SCENE=pillar`:

| controller | collisions | clearance min [m] | final [m] | reached | params |
|---|---|---|---|---|---|
| `ann` | 0 | 0.597 | 0.308 | no | 89 000 |
| `snn` | 0 | 0.727 | 1.944 | no | 89 000 |
| `field_ann` | 0 | 0.228 | 0.252 | yes | 9 984 |
| `field_snn` | 0 | 0.229 | 0.251 | yes | 9 984 |

The end-to-end nets clear the obstacle but under-fit the goal tolerance; the
structured field nets both clear and reach. See
[`tools/compare_controllers.py`](../tools/compare_controllers.py) and
[`tools/field_viz.py`](../tools/field_viz.py) (writes a self-contained Plotly HTML
showing the LiDAR, learned $U_\text{obs}/F_\text{obs}$, nominal vs modulated DS
and the trajectory).

## 9. Schematics

- [assets/schematics/field.svg](assets/schematics/field.svg) — RBF basis anchored
  to the preview, analytic gradient, and the modulation geometry.
