# Sensing: obstacle geometry and the LiDAR model

Source of truth: `src/drone6dof/geometry.py` (obstacle primitives),
`src/drone6dof/sensor.py` (scan model + cues) and `src/drone6dof/policy.py`
(feature assembly).

The policy is **sensor-conditioned**: it never sees obstacle identities or
extents, only a horizontal LiDAR scan plus a few cues derived from the profile.
The same scan feeds the dashboard's ray visualization.

## 1. Obstacle geometry

Shapes live in the horizontal plane with a stored `height`/`z0`; the current
teacher and clearance metric use the 2-D cross-section (a 2.5-D world). Each
shape exposes the three primitives the DS teacher and the sensor need
(`geometry.py:1-16`).

### Oriented box
Let `loc = R(-angle)·(p - center)` be the point in the box frame and
`q = |loc| - half`. The signed distance is the standard box SDF

$$
\text{sdf}(p) = \big\lVert \max(q,0) \big\rVert + \min\!\big(\max(q_x,q_y),0\big)
$$

(negative inside; `geometry.py:63-76`). `closest_point_normal` clips to the box
when outside; when inside it pushes out through the nearest face
(`geometry.py:78-103`). Ray intersection uses the slab method
(`geometry.py:105-122`).

### Cylinder
The legacy pillar shape (`geometry.py:135-200`):

$$
\text{sdf}(p) = \lVert p - c \rVert - r,\qquad
n = \frac{p-c}{\lVert p-c\rVert},\qquad
\text{ray: } \lVert o + t\,d - c\rVert = r.
$$

The cylinder's `core_offset` defaults to its radius, so distance thresholds are
measured from the axis and the ported single-pillar behaviour is reproduced
exactly (`geometry.py:143-150`, `geometry.py:8-15`).

### Shared helpers
`clearance(obstacles, p)` = minimum signed distance (∞ with no obstacles);
`ray_cast(obstacles, origin, angles, r_max)` returns, per beam, the nearest
entry distance or `r_max` on a miss (`geometry.py:203-229`).

## 2. LiDAR scan

`SensorConfig` (`sensor.py:28-51`):

| Field | Default | Meaning |
|---|---|---|
| `k` | 32 | beams over 360° |
| `r_max` | 3.0 m | max range |
| `range_sigma` | 0.02 m | additive range noise |
| `dropout_p` | 0.02 | random beam dropout |
| `max_range_miss_p` | 0.01 | random max-range/missed return |
| `seed` | 0 | RNG seed |

Beams are `linspace(0, 2π, k, endpoint=False)`. The ideal profile comes from
`ray_cast`; then (`sensor.py:54-70`)

$$
r_k \leftarrow r_k + \mathcal N(0,\sigma_r^2),\quad
\text{miss/dropout} \Rightarrow r_k \leftarrow r_\text{max},\quad
r_k \leftarrow \mathrm{clip}(r_k, 0, r_\text{max}).
$$

## 3. Shape cues

Five cues are appended after the beams (`SENSOR_CUE_DIM = 5`,
`sensor.py:73-85`). With `i* = argmin r`, `Δ = 2π/k`:

$$
\text{min range} = \frac{r_{i*}}{r_\text{max}},\quad
\text{bearing} = (\cos\theta_{i*},\ \sin\theta_{i*}),
$$
$$
\text{slope} = \frac{r_{i^*+1} - r_{i^*-1}}{2\Delta\, r_\text{max}},\quad
\text{curvature} = \frac{|r_{i^*-1} - 2r_{i^*} + r_{i^*+1}|}{r_\text{max}}.
$$

Indices wrap around the ring. The cues are normalized so a fixed network
generalizes across sensor scales.

## 4. Policy feature vector

The sensor-conditioned input is (`policy.py:75-99`)

$$
\phi = \big[\ \underbrace{e(3),\ \dot e(3)}_{} ,\ u_\text{ff}(3),\ \underbrace{r_k/r_\text{max}}_{k},\ \underbrace{\text{cues}(5)}_{} \ \big],
$$

so the width is

$$
\text{policy\_input\_dim} = 3\,\text{pos\_dim} + (k + 5) = 9 + 37 = 46
$$

for the defaults (`sensor.py:40-42`, `policy.py:75-77`). Here $e = p - r$ and
$\dot e = v - \dot r$ are the tracking errors, and `u_ff` is the reference
feed-forward (zero for the constant-goal task). The scan is a horizontal slice
of the 2.5-D geometry, so the same code becomes a 3-D slice later
(`sensor.py:1-8`). Two simplifications to keep in mind: the beams are cast at
**world-fixed** angles (not body-yaw-relative) — `scan()` uses `config.angles()`
and the world position — and obstacle height is ignored, so a drone flying above
a pillar is avoided as if at ground level.

## 5. Scenes and presets

Obstacles are described by `kind` (`box`/`cylinder`) with `center`, extents,
`height`, `z0`, and the `dead_radius`/`influence_radius` used by the DS teacher
(`geometry.py:35-200`). Named presets (`config.py:140-152`):

| Scene | Obstacles |
|---|---|
| `pillar` | the shipped single cylinder `(-1.0, 0, 0.8)`, radius 0.35 m |
| `boxes` | two rotated boxes |
| `wall` | one long thin box |
| `slalom` | three boxes in a zig-zag |

Presets are selectable in the dashboard header and via
`POST /api/benchmark {"scene": …}`.

## 6. Schematics

- [assets/schematics/lidar.svg](assets/schematics/lidar.svg) — beam fan, nearest
  return, and the derived min-range/bearing/slope/curvature cues.
