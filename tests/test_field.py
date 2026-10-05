"""Structured field representation: basis, teacher potential, modulation, controller."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drone6dof.config import CONTROL_LIMIT, INIT_STATE, build_scene, preset_scene
from drone6dof.field import (
    FieldConfig,
    PotentialBasis,
    modulate_ds,
    nominal_ds,
    teacher_coeffs,
    teacher_potential,
)
from drone6dof.geometry import BoxObstacle
from drone6dof.scene import Scene
from drone6dof.weights import DEFAULT_FIELD_PATH, DEFAULT_WEIGHTS_PATH

_FIELD_REF = Path(DEFAULT_FIELD_PATH).with_name("quad6dof_field_ref.npz")


def test_basis_size_and_analytic_gradient():
    cfg = FieldConfig()
    basis = PotentialBasis(cfg)
    scene = preset_scene("pillar")
    state = np.array([-2.2, 0.0, 0.5, 0.0, 0.0, 0.0])
    centers = basis.centers(state, scene.goal_np)
    assert centers.shape == (cfg.k, 3)
    coeffs = teacher_coeffs(scene, state, cfg)
    p = np.array([-1.5, 0.0, 0.5])
    g = basis.grad_U(coeffs, p, centers)
    eps = 1e-6
    fd = np.array([
        (basis.eval_U(coeffs, p + [eps, 0, 0], centers)
         - basis.eval_U(coeffs, p - [eps, 0, 0], centers)) / (2 * eps),
        (basis.eval_U(coeffs, p + [0, eps, 0], centers)
         - basis.eval_U(coeffs, p - [0, eps, 0], centers)) / (2 * eps),
    ])
    assert np.allclose(g[:2], fd, atol=1e-3), (g[:2], fd)


def test_teacher_potential_is_positive_local_and_superposes():
    scene = preset_scene("pillar")
    assert teacher_potential(scene, np.array([-1.0, 0.0, 0.5])) > 0.0
    assert teacher_potential(scene, np.array([3.0, 3.0, 0.5])) == 0.0

    box1 = BoxObstacle(center=(0.0, 0.0), half=(0.2, 0.2))
    box2 = BoxObstacle(center=(2.0, 0.0), half=(0.2, 0.2))
    p = np.array([-0.5, 0.0, 0.5])
    u1 = teacher_potential(Scene(goal=(0, 0, 2.5), obstacles=(box1,)), p)
    u2 = teacher_potential(Scene(goal=(0, 0, 2.5), obstacles=(box2,)), p)
    u12 = teacher_potential(Scene(goal=(0, 0, 2.5), obstacles=(box1, box2)), p)
    assert u12 == pytest.approx(u1 + u2)


def test_modulation_is_identity_without_field():
    cfg = FieldConfig()
    v = nominal_ds(np.array([1.0, 0.0, 0.5]))
    assert np.allclose(modulate_ds(v, np.zeros(3), cfg), v)
    near = modulate_ds(v, np.array([1.0, 0.0, 0.0]), cfg)
    assert not np.allclose(near, v)          # deflected when the field is present
    assert near[0] == pytest.approx(0.0, abs=1e-9)  # radial push into obstacle removed


def test_field_bundle_has_fewer_parameters():
    from drone6dof.benchmark import field_weights_info, weights_info
    from drone6dof.weights import load_field_weights, load_weights

    if not weights_info().get("loaded") or not field_weights_info().get("loaded"):
        pytest.skip("bundles not built")
    fb = load_field_weights(DEFAULT_FIELD_PATH)
    cb = load_weights(DEFAULT_WEIGHTS_PATH)
    fp = fb["w_in"].size + fb["w_out"].size + fb["w_mag"].size
    cp = cb["w_in"].size + cb["w_out"].size + cb["w_mag"].size
    assert fp < cp, (fp, cp)


def test_field_ann_matches_torch_reference():
    from drone6dof.benchmark import field_weights_info
    from drone6dof.connectome import ConnectomeANN, SparseRecurrence
    from drone6dof.weights import load_field_weights

    if not field_weights_info().get("loaded") or not _FIELD_REF.exists():
        pytest.skip("field bundle/reference not built")
    b = load_field_weights(DEFAULT_FIELD_PATH)
    io = np.load(_FIELD_REF)
    edges = np.asarray(b["edges"], dtype=np.int64)
    signed = np.abs(b["w_mag"]) * np.asarray(b["polarity"])[edges[1]]
    rec = SparseRecurrence(edges, signed, int(b["n_neurons"]))
    net = ConnectomeANN(b["w_in"], b["w_out"], rec,
                        steps_per_frame=int(b["connectome_steps"]),
                        limit=float(b["field"].get("field_limit", 2.0)))
    out = np.asarray([np.clip(net.forward_input(row), 0.0, b["field"].get("field_limit", 2.0))
                      for row in np.asarray(io["inputs"])])
    assert np.allclose(out, np.asarray(io["ann_out"]), atol=1e-4)


def test_field_controllers_run_and_avoid():
    from drone6dof.benchmark import field_weights_info
    from drone6dof.cli import make_controller
    from drone6dof.dynamics import NumpyPlantBackend
    from drone6dof.sim import Simulation

    if not field_weights_info().get("loaded"):
        pytest.skip("field bundle not built")
    scene = build_scene()
    for name in ("field_ann", "field_snn"):
        backend = NumpyPlantBackend(heading_target=scene.goal_np)
        ctrl = make_controller(name, scene, CONTROL_LIMIT)
        sim = Simulation(backend, ctrl, scene, steps=300, initial_state=INIT_STATE)
        sim.run()
        m = sim.metrics()
        assert m["collisions"] == 0, (name, m)
        assert np.isfinite(m["final_goal_dist_m"])
