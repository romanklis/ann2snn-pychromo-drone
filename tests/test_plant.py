"""Plant fidelity tests (ported from ``tests/test_quad6dof.py``).

These exercise the numpy plant only; no PyChrono required.
"""

from __future__ import annotations

import numpy as np
import pytest

from drone6dof.params import QuadParams
from drone6dof.plant import Quad6DoF

DT = 0.02


def _rollout(plant, u, frames=100, dt=DT, state0=(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)):
    state = np.array(state0, dtype=np.float64)
    u = np.array(u, dtype=np.float64)
    for _ in range(frames):
        state = plant(state, u, dt=dt, limit=12.0, gain=1.0, damping=0.0, disturbance=None)
    return state


def test_hover_holds_altitude_at_one_g():
    plant = Quad6DoF()
    state = _rollout(plant, [0.0, 0.0, 0.0])
    assert abs(float(state[2]) - 1.0) < 0.05
    assert abs(float(state[5])) < 0.02
    assert plant.last_telemetry["g_force"] == pytest.approx(1.0, abs=0.02)
    rpy = np.degrees(plant.attitude_rpy)
    assert np.all(np.abs(rpy[:2]) < 1.0)


def test_lateral_demand_tilts_and_accelerates_the_right_way():
    plant = Quad6DoF()
    state = _rollout(plant, [3.0, 0.0, 0.0], frames=150)
    assert float(state[0]) > 0.5
    assert float(state[3]) > 0.5
    assert float(plant.attitude_rpy[1]) > 0.0
    # tilting spends part of the capped thrust sideways, so the drone descends
    assert float(state[2]) < 1.0


def test_rotation_stays_on_so3_and_battery_drains():
    plant = Quad6DoF()
    state = _rollout(plant, [2.0, 1.0, 0.5], frames=250)
    R = plant.R
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    tele = plant.last_telemetry
    assert tele["soc_pct"] < 95.0
    assert tele["v_term"] < tele["v_ocv"]
    for key in ("g_force", "rpm1", "current1", "power_w", "advance_ratio"):
        assert key in tele
    assert np.isfinite(state).all()


def test_plant_is_deterministic():
    def run():
        plant = Quad6DoF()
        return _rollout(plant, [1.5, -0.5, 0.8], frames=80)

    assert np.array_equal(run(), run())


def test_inner_rate_convergence():
    """250 Hz and 1 kHz substeps must agree closely on the same command."""

    def rollout(inner_hz, seconds=6.0):
        plant = Quad6DoF(QuadParams(inner_hz=inner_hz))
        state = np.array([-2.2, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float64)
        for k in range(int(seconds / DT)):
            t = k * DT
            u = np.array(
                [
                    0.8 * np.sin(2 * np.pi * 0.5 * t),
                    0.5 * np.sin(2 * np.pi * 0.7 * t),
                    0.4 * np.sin(2 * np.pi * 0.3 * t),
                ],
                dtype=np.float64,
            )
            state = plant(state, u, dt=DT, limit=12.0, gain=1.0, damping=0.0, disturbance=None)
        return state

    coarse = rollout(250.0)
    mid = rollout(500.0)
    fine = rollout(1000.0)
    assert np.allclose(coarse, mid, atol=0.05), float(np.abs(coarse - mid).max())
    assert np.allclose(mid, fine, atol=0.05), float(np.abs(mid - fine).max())


def _hover_drift(gain: float, frames: int = 100) -> float:
    plant = Quad6DoF()
    state = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    zeros = np.zeros(3)
    for _ in range(frames):
        state = plant(state, zeros, dt=DT, limit=20.0, gain=gain, damping=0.0, disturbance=None)
    return float(state[2]) - 1.0


def test_hover_has_no_authority_bias():
    base = _hover_drift(1.0)
    assert abs(base) < 0.10, base
    for gain in (0.95, 0.9, 0.8):
        drift = _hover_drift(gain)
        assert abs(drift - base) < 0.05, (gain, drift, base)
        assert abs(drift) < 0.15, (gain, drift)


def test_demand_limit_is_inside_the_achievable_envelope():
    plant = Quad6DoF()
    tip = {}
    for gain in (1.0, 0.9, 0.8):
        # reset once per gain; the motor state then ramps up across the demand
        # sweep, exactly as in the upstream test
        plant.reset()
        best = 0.0
        for demand in (2.0, 4.0, 6.0, 8.0, 10.0, 12.0):
            local = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            u = np.array([0.0, 0.0, demand], dtype=np.float64)
            v0 = float(local[5])
            nxt = plant(local, u, dt=DT, limit=20.0, gain=gain, damping=0.0, disturbance=None)
            best = max(best, (float(nxt[5]) - v0) / DT)
        tip[gain] = best
    assert tip[1.0] >= 7.5, tip
    assert tip[0.9] < tip[1.0] and tip[0.8] < tip[0.9], tip
    assert tip[1.0] < 12.0, tip


def test_damping_acts_as_extra_drag():
    def final_speed(damping: float) -> float:
        plant = Quad6DoF()
        state = np.array([0.0, 0.0, 1.0, 0.4, 0.0, 0.0], dtype=np.float64)
        zeros = np.zeros(3)
        for _ in range(50):
            state = plant(state, zeros, dt=DT, limit=20.0, gain=1.0,
                          damping=damping, disturbance=None)
        return float(np.linalg.norm(state[3:6]))

    undamped = final_speed(0.0)
    damped = final_speed(0.6)
    assert damped < undamped * 0.75, (damped, undamped)


def test_hidden_state_is_exposed_for_the_dashboard():
    plant = Quad6DoF()
    state = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    plant(state, np.array([0.0, 0.0, 12.0]), dt=DT, limit=20.0, gain=0.8,
          damping=0.0, disturbance=None)
    tele = plant.last_telemetry
    for key in ("thrust_demand_n", "thrust_achieved_n", "thrust_margin_frac",
                "thrust_saturated"):
        assert key in tele, key
    assert tele["thrust_saturated"] == pytest.approx(1.0)
    assert tele["thrust_margin_frac"] < 0.0


@pytest.mark.slow
def test_short_horizon_parity_with_the_original_prototype():
    """8 s of DS guidance + ported plant tracks the vendored prototype.

    Uses the prototype's own task (goal (0,0,2.5), pillar (1.35,0,1.5)) and gains,
    so this measures the plant/controller port rather than the example's re-tuned
    task.
    """
    from drone6dof.control import DSGuidanceController
    from drone6dof.dynamics import NumpyPlantBackend
    from drone6dof.reference import goal_reference
    from drone6dof.scene import SceneSpec
    from quad_snippet_original import FullPhysicsQuadcopter10kHz

    proto = FullPhysicsQuadcopter10kHz()
    tel = proto.run(sim_time=8.0, dt_outer=0.005, f_inner=10000)

    scene = SceneSpec(goal=(0.0, 0.0, 2.5), obstacle=(1.35, 0.0, 1.5)).instantiate()
    ref = goal_reference(steps=1600, goal=scene.goal, dt=0.005, meta={"scene": scene})
    teacher = DSGuidanceController(
        speed_cap=2.2, ds_radial_gain=0.35, action_limit=12.0, scene=scene
    )
    plant = NumpyPlantBackend(QuadParams(inner_hz=10000.0), heading_target=scene.goal_np)
    plant.reset(np.array([-2.2, 0.0, 0.5, 0.0, 0.0, 0.0]))

    peak_g = 0.0
    pos = []
    for k in range(1600):
        u = teacher.act(plant.state, ref.at(k))
        plant.step(u, dt=0.005, limit=12.0, gain=1.0, damping=0.0)
        pos.append(plant.position)
        peak_g = max(peak_g, plant.telemetry["g_force"])

    drift = float(np.linalg.norm(np.asarray(pos)[-1] - tel["pos"][-1]))
    assert drift < 0.25, f"final position drifted {drift:.3f} m from the prototype"
    assert peak_g == pytest.approx(float(tel["g_force"].max()), rel=0.15)
    assert abs(float(np.linalg.det(plant.rotation)) - 1.0) < 1e-9
