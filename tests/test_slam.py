"""SLAM-lite occupancy mapper: inverse sensor model, metrics, planning."""

from __future__ import annotations

import numpy as np
import pytest

from drone6dof.config import SLAM, build_scene
from drone6dof.slam import OccupancyMap, SlamConfig, truth_shell


def _map():
    return OccupancyMap(SlamConfig(bounds=(-2.0, 2.0, -2.0, 2.0), res=0.1))


def test_inverse_sensor_model_free_occupied_unknown():
    m = _map()
    # one beam straight ahead hitting a wall at 1.0 m
    m.update((0.0, 0.0), np.array([1.0]), np.array([0.0]), r_max=3.0)
    occ = m.occupancy()
    assert occ[m._cell((1.0, 0.0))] == 2          # hit -> occupied
    assert occ[m._cell((0.5, 0.0))] == 1          # swept -> free
    assert occ[m._cell((1.5, 0.0))] == 0          # beyond -> unknown


def test_miss_is_free_not_occupied():
    m = _map()
    m.update((0.0, 0.0), np.array([3.0]), np.array([0.0]), r_max=3.0)  # max range
    occ = m.occupancy()
    assert occ[m._cell((1.0, 0.0))] == 1           # swept -> free
    assert m.occupied_cells() == 0                 # a miss never marks occupied
    assert occ[m._cell((0.0, 1.0))] == 0           # perpendicular -> untouched


def test_coverage_grows_with_updates():
    m = _map()
    before = m.explored_frac()
    for i in range(5):
        m.update((0.0, 0.0), np.array([1.0]), np.array([0.05 * i]), r_max=3.0)
    assert m.explored_frac() > before


def test_dense_shape_and_values():
    m = _map()
    m.update((0.0, 0.0), np.array([1.0, 3.0]), np.array([0.0, np.pi]), r_max=3.0)
    d = m.dense(20)
    assert d.shape == (20, 20)
    assert d.dtype == np.uint8
    assert set(np.unique(d)).issubset({0, 1, 2})


def test_surface_recall_bounds():
    scene = build_scene()
    m = OccupancyMap(SlamConfig.from_dict(SLAM))
    shell = truth_shell(scene, m)
    assert int(np.sum(shell == 2)) > 0
    assert m.surface_recall(shell) == 0.0
    m.log_odds[shell == 2] = 4.0                  # pretend we discovered it
    assert m.surface_recall(shell) == pytest.approx(1.0, abs=1e-6)


def test_plan_conservative_empty_is_none_optimistic_is_path():
    m = _map()
    # empty map: conservative (unknown blocked) has no route; optimistic does
    assert m.plan((-1.5, 0.0), (1.5, 0.0), conservative=True) is None
    way = m.plan((-1.5, 0.0), (1.5, 0.0), conservative=False)
    assert way is not None and way[0] == pytest.approx((-1.5, 0.0))


def test_slam_config_roundtrip():
    cfg = SlamConfig.from_dict(SLAM)
    assert cfg.res == pytest.approx(SLAM["res"])
    assert list(cfg.to_dict()["bounds"]) == [float(v) for v in SLAM["bounds"]]


def test_slam_closed_loop_smoke():
    from drone6dof.benchmark import run_benchmark

    r = run_benchmark(["field_ann"], steps=200, map_source="slam")
    res = r["results"].get("field_ann")
    if res is None:
        pytest.skip("field bundle not available")
    assert res["map"] is not None and res["slam"] is not None
    assert res["metrics"]["collisions"] == 0
    assert res["slam"]["replans"] >= 1
    assert 0.0 <= res["slam"]["surface_coverage"] <= 1.0


def test_replan_events_are_recorded_and_event_driven():
    from drone6dof.benchmark import run_benchmark

    r = run_benchmark(["field_ann"], steps=400, map_source="slam")
    res = r["results"].get("field_ann")
    if res is None:
        pytest.skip("field bundle not available")
    s = res["slam"]
    steps = s["replan_steps"]
    assert steps == sorted(steps)
    assert all(0 <= v <= 400 for v in steps)
    # event-driven + refresh~40: far fewer than a blind period-10 policy (~40+)
    assert s["replans"] < 60, s["replans"]
    assert s["goal_replans"] + s["frontier_replans"] == s["replans"]
    assert res["metrics"]["collisions"] == 0
