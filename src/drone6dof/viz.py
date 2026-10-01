"""PyChrono / Irrlicht visualization layer (visualization-only).

The drone body is a kinematic Chrono body whose pose is set from the numpy plant
state each frame; no Chrono dynamics are integrated.  The scene has a ground
grid, the pillar, a goal marker, the drone (body, arms, four spinning rotors and
a body-frame triad), a trajectory trail, a configurable camera and a HUD.

Chrono's Irrlicht camera defaults to Y-up; we request a Z-up camera
(:meth:`ChVisualSystemIrrlicht.SetCameraVertical`) so the plant's native z-up
world is used directly.  If the installed Chrono cannot switch camera vertical,
we fall back to rotating all rendered geometry by ``R_x(-90°)``
(``v_vis = (x, z, -y)``).
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple

import numpy as np

from .params import QuadParams

__all__ = ["ChronoViz", "chrono_available"]

#: source-frame rotation used only if the Z-up camera cannot be requested
_RX_MINUS_90 = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64
)


def chrono_available() -> bool:
    """Whether the real PyChrono (Project Chrono) can be imported."""
    try:
        import pychrono.core  # noqa: F401
        import pychrono.irrlicht  # noqa: F401
    except Exception:
        return False
    return True


def _quat_from_matrix(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Rotation matrix -> quaternion ``(w, x, y, z)`` (Shepperd's method)."""
    m = np.asarray(R, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = np.linalg.norm(q)
    return tuple(float(v) for v in (q / n))


def _quat_mul(a, b):
    """Quaternion product ``a ⊗ b`` for ``(w, x, y, z)`` tuples."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_about_z(angle: float) -> Tuple[float, float, float, float]:
    h = 0.5 * float(angle)
    return (float(np.cos(h)), 0.0, 0.0, float(np.sin(h)))


class ChronoViz:
    """Build and drive the Irrlicht scene for the 6-DoF drone demo."""

    def __init__(
        self,
        params: Optional[QuadParams] = None,
        *,
        window_size: Tuple[int, int] = (1280, 720),
        title: str = "ANN2SNN 6-DoF drone - PyChrono",
        camera_mode: str = "orbit",
        trail_stride: int = 2,
    ) -> None:
        if not chrono_available():
            raise ImportError(
                "PyChrono (Project Chrono) is not importable. Install it from "
                "conda-forge (`conda install -c conda-forge pychrono`) or run "
                "with `--vis none`."
            )
        import pychrono.core as chrono
        import pychrono.irrlicht as chronoirr

        self.chrono = chrono
        self.chronoirr = chronoirr
        self.params = params or QuadParams()
        self.camera_mode = camera_mode
        self.trail_stride = max(1, int(trail_stride))

        self.system = chrono.ChSystemNSC()
        self.system.SetGravitationalAcceleration(chrono.ChVector3d(0.0, 0.0, -9.81))

        self.vis = chronoirr.ChVisualSystemIrrlicht()
        self.vis.AttachSystem(self.system)
        self.vis.SetWindowSize(int(window_size[0]), int(window_size[1]))
        self.vis.SetWindowTitle(title)

        # Z-up camera if supported; otherwise rotate rendered geometry.
        self._z_up = self._try_set_z_up()
        self._map = np.eye(3) if self._z_up else _RX_MINUS_90

        self._material = chrono.ChContactMaterialNSC()
        self._rotor_angles = np.zeros(4, dtype=np.float64)
        self._spin_sign = np.array([1.0, -1.0, 1.0, -1.0])
        self._last_hud = ""
        self._hud_element = None
        self._trail_bodies = []
        self._trail_capacity = 1
        self._goal = np.zeros(3)
        self._is_built = False

    # -- camera vertical ---------------------------------------------------- #
    def _try_set_z_up(self) -> bool:
        chrono = self.chrono
        candidates = []
        for owner in (chrono, getattr(chrono, "ChVisualSystem", None)):
            enum = getattr(owner, "CameraVerticalDir", None) if owner is not None else None
            if enum is not None:
                candidates.append(getattr(enum, "Z", None))
        candidates.extend([getattr(chrono, "CameraVerticalDir_Z", None), 1])
        for value in candidates:
            if value is None:
                continue
            try:
                self.vis.SetCameraVertical(value)
                return True
            except Exception:
                continue
        return False

    # -- helpers ------------------------------------------------------------ #
    def to_vis(self, v: np.ndarray) -> np.ndarray:
        return self._map @ np.asarray(v, dtype=np.float64).reshape(3)

    def to_vis_R(self, R: np.ndarray) -> np.ndarray:
        return self._map @ np.asarray(R, dtype=np.float64).reshape(3, 3)

    def _vec(self, v):
        return self.chrono.ChVector3d(float(v[0]), float(v[1]), float(v[2]))

    def _color(self, body, color) -> None:
        for getter in (
            lambda: body.GetVisualShape(0),
            lambda: body.GetVisualModel().GetShape(0),
        ):
            try:
                getter().SetColor(color)
                return
            except Exception:
                continue

    def _quat_z(self, angle: float):
        try:
            return self.chrono.QuatFromAngleZ(angle)
        except Exception:
            return self.chrono.QuatFromAngleAxis(angle, self.chrono.VECT_Z)

    # -- scene construction ------------------------------------------------- #
    def build(self, scene, init_state, steps: int = 500) -> "ChronoViz":
        """Create ground, pillar, goal marker, drone and trail. Call once."""
        if self._is_built:
            return self
        chrono = self.chrono
        p = self.params
        self._trail_capacity = max(1, int(steps) // self.trail_stride + 1)
        self._goal = np.asarray(scene.goal, dtype=np.float64)

        # ---- ground -------------------------------------------------------- #
        ground = chrono.ChBodyEasyBox(16.0, 16.0, 0.04, 1000.0, False, True, self._material)
        ground.SetFixed(True)
        ground.SetPos(self._vec(self.to_vis([0.0, 0.0, -0.02])))
        self._color(ground, chrono.ChColor(0.35, 0.38, 0.42))
        self.system.AddBody(ground)

        # ---- pillar -------------------------------------------------------- #
        obs = scene.obstacle_np
        if obs is not None:
            pillar = chrono.ChBodyEasyCylinder(
                chrono.ChAxis_Z,
                float(scene.core_radius),
                float(scene.obstacle_height),
                500.0,
                False,
                True,
                self._material,
            )
            center = [float(obs[0]), float(obs[1]), float(scene.obstacle_height) / 2.0]
            pillar.SetFixed(True)
            pillar.SetPos(self._vec(self.to_vis(center)))
            self._color(pillar, chrono.ChColor(0.85, 0.6, 0.2))
            self.system.AddBody(pillar)

        # ---- goal marker --------------------------------------------------- #
        goal = chrono.ChBodyEasySphere(0.09, 1.0, False, True, self._material)
        goal.SetFixed(True)
        goal.SetPos(self._vec(self.to_vis(self._goal)))
        self._color(goal, chrono.ChColor(0.1, 0.85, 0.3))
        self.system.AddBody(goal)

        # ---- drone --------------------------------------------------------- #
        self.drone = chrono.ChBody()
        self.drone.SetFixed(True)
        self.drone.SetPos(self._vec(self.to_vis(np.asarray(init_state)[:3])))
        self.system.AddBody(self.drone)

        body_shape = chrono.ChVisualShapeBox(0.13, 0.13, 0.045)
        body_shape.SetColor(chrono.ChColor(0.15, 0.15, 0.18))
        self.drone.AddVisualShape(body_shape)

        a = float(p.d) / np.sqrt(2.0)          # motor offset in x and y
        arm_len = float(np.hypot(a, a))
        motors = np.array(
            [[a, a, 0.0], [-a, a, 0.0], [-a, -a, 0.0], [a, -a, 0.0]], dtype=np.float64
        )
        self._motor_offsets = motors
        for i, m in enumerate(motors):
            angle = float(np.arctan2(m[1], m[0]))
            arm = chrono.ChVisualShapeBox(arm_len, 0.014, 0.012)
            arm.SetColor(chrono.ChColor(0.25, 0.27, 0.32))
            self.drone.AddVisualShape(
                arm, chrono.ChFramed(self._vec(m * 0.5), self._quat_z(angle))
            )

        # body-frame triad (thin colored boxes; no line-geometry binding needed)
        triad = (
            ("x", np.array([0.22, 0.0, 0.0]), (0.22, 0.012, 0.012), chrono.ChColor(0.9, 0.2, 0.2)),
            ("y", np.array([0.0, 0.22, 0.0]), (0.012, 0.22, 0.012), chrono.ChColor(0.2, 0.9, 0.2)),
            ("z", np.array([0.0, 0.0, 0.22]), (0.012, 0.012, 0.22), chrono.ChColor(0.25, 0.45, 1.0)),
        )
        for _, tip, dims, color in triad:
            axis = chrono.ChVisualShapeBox(*dims)
            axis.SetColor(color)
            self.drone.AddVisualShape(axis, chrono.ChFramed(self._vec(tip * 0.5), chrono.QUNIT))

        # rotor bodies
        self.rotors = []
        for i, m in enumerate(motors):
            rotor = chrono.ChBody()
            rotor.SetFixed(True)
            rotor.SetPos(self._vec(self.to_vis(np.asarray(init_state)[:3])))
            self.system.AddBody(rotor)
            disc = chrono.ChVisualShapeCylinder(0.065, 0.008)
            disc.SetColor(chrono.ChColor(0.6, 0.62, 0.66))
            rotor.AddVisualShape(disc)
            blade = chrono.ChVisualShapeBox(0.12, 0.014, 0.006)
            blade.SetColor(chrono.ChColor(0.1, 0.1, 0.12))
            rotor.AddVisualShape(blade)
            self.rotors.append(rotor)

        # ---- trail (pre-allocated markers; positioned as the flight proceeds) - #
        init_pos = np.asarray(init_state, dtype=np.float64)[:3]
        self._trail_bodies = []
        for _ in range(self._trail_capacity):
            marker = chrono.ChBodyEasySphere(0.012, 1.0, False, True, self._material)
            marker.SetFixed(True)
            marker.SetPos(self._vec(self.to_vis(init_pos)))
            self._color(marker, chrono.ChColor(0.9, 0.25, 0.2))
            self.system.AddBody(marker)
            self._trail_bodies.append(marker)

        self.vis.Initialize()
        self.vis.AddLogo()
        self.vis.AddSkyBox()
        self._add_grid()
        self._add_camera(init_state)
        self.vis.AddTypicalLights()
        self._make_hud()
        self._is_built = True
        return self

    def _add_grid(self) -> None:
        try:
            self.vis.AddGrid(
                0.5,
                0.5,
                12,
                12,
                self.chrono.ChCoordsysd(self.chrono.ChVector3d(0, 0, 0), self.chrono.QUNIT),
                self.chrono.ChColor(0.28, 0.30, 0.34),
            )
        except Exception:
            pass

    def _add_camera(self, init_state) -> None:
        init = np.asarray(init_state, dtype=np.float64)[:3]
        center = self.to_vis(init)
        if self.camera_mode == "follow":
            eye = center + self.to_vis(np.array([-1.6, -1.6, 1.1]))
        else:
            eye = self.to_vis(np.array([3.6, -3.6, 2.8]))
        self._cam_eye = eye
        self._cam_center = center
        self.vis.AddCamera(self._vec(eye), self._vec(self.to_vis(self._goal)))
        self._orbit_radius = float(np.linalg.norm(eye[:2] - center[:2]) + 1e-6)
        self._orbit_angle = 0.0

    def _make_hud(self) -> None:
        try:
            gui = self.vis.GetGUIEnvironment()
            rect = self.chronoirr.recti(10, 10, 560, 150)
            self._hud_element = gui.addStaticText("", rect)
        except Exception:
            self._hud_element = None

    # -- per-frame sync ----------------------------------------------------- #
    def sync(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        telemetry: Dict[str, float],
        dt: float,
        step: int = 0,
    ) -> None:
        chrono = self.chrono
        pos_vis = self.to_vis(position)
        R_vis = self.to_vis_R(rotation)
        q = _quat_from_matrix(R_vis)

        self.drone.SetPos(self._vec(pos_vis))
        self.drone.SetRot(chrono.ChQuaterniond(*q))

        # rotors: world pose = drone pose ∘ local spin
        for i, rotor in enumerate(self.rotors):
            rpm = float(telemetry.get(f"rpm{i + 1}", 0.0))
            omega = rpm * 2.0 * np.pi / 60.0
            self._rotor_angles[i] += self._spin_sign[i] * omega * dt
            offset = R_vis @ self._motor_offsets[i]
            rpos = pos_vis + offset
            qr = _quat_mul(q, _quat_about_z(self._rotor_angles[i]))
            rotor.SetPos(self._vec(rpos))
            rotor.SetRot(chrono.ChQuaterniond(*qr))

        # trail
        if step % self.trail_stride == 0:
            idx = step // self.trail_stride
            if idx < len(self._trail_bodies):
                self._trail_bodies[idx].SetPos(self._vec(pos_vis))

        # camera
        if self.camera_mode == "follow":
            eye = pos_vis + self.to_vis(np.array([-1.6, -1.6, 1.1]))
            self.vis.SetCameraPosition(self._vec(eye))
            self.vis.SetCameraTarget(self._vec(pos_vis))
        elif self.camera_mode == "orbit":
            self._orbit_angle += 0.15 * dt
            c = self.to_vis(self._goal)
            eye = np.array(
                [
                    c[0] + self._orbit_radius * np.cos(self._orbit_angle),
                    c[1] + self._orbit_radius * np.sin(self._orbit_angle),
                    c[2] + 1.6,
                ]
            )
            self.vis.SetCameraPosition(self._vec(eye))
            self.vis.SetCameraTarget(self._vec(c))

        self.system.SetChTime(float(step) * dt)

    def update_hud(self, sim) -> None:
        tele = sim.backend.telemetry
        goal_dist = sim.history["goal_dist"][-1] if sim.history["goal_dist"] else float("nan")
        clearance = sim.history["clearance"][-1] if sim.history["clearance"] else float("nan")
        text = (
            f"t = {sim.k * sim.dt:5.2f} s   step {sim.k}/{sim.steps}   "
            f"{getattr(sim.controller, 'name', 'controller')}\n"
            f"goal dist = {goal_dist:.2f} m   "
            f"clearance = {clearance:+.2f} m\n"
            f"g = {tele.get('g_force', float('nan')):.2f}   "
            f"SoC = {tele.get('soc_pct', float('nan')):.1f}%   "
            f"sat = {int(tele.get('thrust_saturated', 0))}"
        )
        self._last_hud = text
        if self._hud_element is not None:
            try:
                self._hud_element.setText(text)
            except Exception:
                pass

    # -- render loop -------------------------------------------------------- #
    def render_loop(
        self,
        sim,
        *,
        realtime: bool = True,
        max_frames: Optional[int] = None,
        record_dir: Optional[str] = None,
        record_every: int = 1,
    ) -> None:
        """Run the simulation while rendering. ``sim`` steps one frame per draw."""
        if not self._is_built:
            raise RuntimeError("ChronoViz.build(...) must be called before render_loop")
        if record_dir:
            os.makedirs(record_dir, exist_ok=True)
        timer = self._make_timer()
        while self.vis.Run():
            if not sim.done:
                sim.step()
            self.sync(sim.position, sim.rotation, sim.backend.telemetry, sim.dt, sim.k)
            self.update_hud(sim)

            self.vis.BeginScene()
            self.vis.Render()
            # Capture inside the scene (after Render, before EndScene): the
            # framebuffer is only readable before the buffer swap.
            if record_dir and sim.k % record_every == 0:
                try:
                    self.vis.WriteImageToFile(
                        os.path.join(record_dir, f"frame_{sim.k:04d}.png")
                    )
                except Exception:
                    pass
            self.vis.EndScene()

            if realtime:
                self._pace(timer, sim.dt)
            if max_frames is not None and sim.k >= max_frames:
                break

    def _make_timer(self):
        try:
            return self.chrono.ChRealtimeStepTimer()
        except Exception:
            return None

    def _pace(self, timer, dt: float) -> None:
        if timer is not None:
            try:
                timer.Spin(dt)
                return
            except Exception:
                pass
        time.sleep(max(0.0, dt))

    def close(self) -> None:
        try:
            self.vis.Quit()
        except Exception:
            pass
