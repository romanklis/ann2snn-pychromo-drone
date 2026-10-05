"""Benchmark serialisation contract (numpy only)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from drone6dof.benchmark import (
    available_controllers,
    run_benchmark,
    weights_info,
)
from drone6dof.config import validate_goal

_HAS_WEIGHTS = bool(weights_info().get("loaded"))


def _names():
    return ["ds_guidance", "pid"] + (["ann", "snn"] if _HAS_WEIGHTS else [])


def test_report_shapes_and_keys():
    report = run_benchmark(_names(), steps=20)
    assert report["example"] == "quad6dof"
    assert report["steps"] == 20
    assert report["seed"] == 0
    assert set(report["controllers"]) == set(_names())
    for name in report["controllers"]:
        res = report["results"][name]
        assert len(res["t"]) == 21
        assert len(res["trajectory"]) == 21
        assert len(res["state"]) == 21
        assert len(res["attitude"]) == 21
        assert len(res["command"]) == 21
        assert len(res["goal_dist"]) == 21
        assert len(res["clearance"]) == 21
        assert len(res["success"]) == 21
        assert np.isfinite(np.asarray(res["trajectory"])).all()
        assert "g_force" in res["telemetry"]
        assert isinstance(res["metrics"], dict)
        assert res["metrics"]["steps"] == 20


def test_report_is_strict_json_safe():
    """The payload must parse with a strict browser parser (no NaN/Infinity)."""
    report = run_benchmark(_names(), steps=20)
    json.dumps(report, allow_nan=False)  # raises ValueError on NaN/Inf
    tele = report["results"]["ds_guidance"]["telemetry"]
    assert any(v is None for v in tele.get("g_force", [])[:1])  # initial frame -> null


def test_teacher_reaches_and_pid_collides():
    report = run_benchmark(["ds_guidance", "pid"], steps=500)
    ds = report["results"]["ds_guidance"]["metrics"]
    pid = report["results"]["pid"]["metrics"]
    assert ds["collisions"] == 0 and ds["closest_goal_dist_m"] < 0.6
    assert pid["collisions"] > 0 and pid["clearance_min_m"] < 0.0


def test_interactive_goal_is_used():
    goal = (2.0, 2.0, 1.5)  # straight path from the start clears the pillar
    report = run_benchmark(["ds_guidance"], steps=800, goal=goal)
    assert report["goal"] == list(goal)
    assert report["scene"]["goal"] == list(goal)
    metrics = report["results"]["ds_guidance"]["metrics"]
    assert metrics["reached_goal"] is True, metrics
    assert metrics["collisions"] == 0, metrics


def test_preset_box_scene():
    report = run_benchmark(["ds_guidance"], steps=400, scene_name="boxes")
    assert report["scene_name"] == "boxes"
    assert len(report["scene"]["obstacles"]) == 2
    metrics = report["results"]["ds_guidance"]["metrics"]
    assert metrics["collisions"] == 0, metrics
    with pytest.raises(ValueError):
        run_benchmark(["ds_guidance"], steps=10, scene_name="nope")


def test_goal_validation():
    assert validate_goal(None) is None
    assert validate_goal([1, 2, 3]) == (1.0, 2.0, 3.0)
    for bad in ([1, 2], [0, 0, float("nan")], [0, 0, 99.0], ["a", 0, 0]):
        with pytest.raises(ValueError):
            validate_goal(bad)


def test_benchmark_is_deterministic():
    names = ["ds_guidance"]
    a = run_benchmark(names, steps=40)["results"]["ds_guidance"]["trajectory"]
    b = run_benchmark(names, steps=40)["results"]["ds_guidance"]["trajectory"]
    assert a == b


@pytest.mark.skipif(not _HAS_WEIGHTS, reason="connectome weights bundle not present")
def test_snn_reports_spikes_and_stats():
    report = run_benchmark(["snn", "ann"], steps=40)
    snn = report["results"]["snn"]
    assert snn["spiking"] is True
    assert snn["spikes"] is not None
    assert len(snn["spikes"]) == 41
    assert len(snn["spikes"][0]) == 200
    assert "spike_rate_hz" in snn["telemetry"]
    assert "h_norm" in report["results"]["ann"]["telemetry"]
    assert report["stats"]["snn_minus_ann"] is not None


def test_missing_weights_marks_ann_snn_unavailable(tmp_path):
    missing = tmp_path / "nope.npz"
    report = run_benchmark(["ds_guidance", "snn", "ann"], steps=10, weights_path=str(missing))
    assert "snn" in report["unavailable"] and "ann" in report["unavailable"]
    assert "snn" not in report["results"]
    assert "ds_guidance" in report["results"]


@pytest.mark.skipif(not _HAS_WEIGHTS, reason="connectome weights bundle not present")
def test_scan_is_recorded_for_sensor_brains():
    report = run_benchmark(["ds_guidance", "snn"], steps=20)
    assert report["results"]["ds_guidance"]["scan"] is None       # teacher is oracle
    scan = report["results"]["snn"]["scan"]
    assert len(scan) == 21 and len(scan[0]) == 32                 # T x k
    assert report["results"]["snn"]["scan_angles"] is not None
    json.dumps(report, allow_nan=False)


def test_controller_flags_and_field_scan():
    from drone6dof.benchmark import field_weights_info

    names = ["ds_guidance", "ann"] + (["snn"] if _HAS_WEIGHTS else [])
    if field_weights_info().get("loaded"):
        names += ["field_ann", "field_snn"]
    res = run_benchmark(names, steps=20)["results"]
    assert res["ds_guidance"]["sensor"] is False and res["ds_guidance"]["scan"] is None
    assert res["ann"]["sensor"] is True and res["ann"]["spiking"] is False
    if _HAS_WEIGHTS:
        assert res["snn"]["spiking"] is True
    if "field_snn" in res:
        assert res["field_snn"]["spiking"] is True
        assert res["field_snn"]["sensor"] is True
        assert res["field_snn"]["scan"] is not None
        assert res["field_ann"]["spiking"] is False and res["field_ann"]["scan"] is not None


def test_available_controllers_reports_reasons():
    avail = available_controllers()
    assert avail["ds_guidance"]["available"] is True
    assert avail["pid"]["available"] is True
    assert avail["snn"]["available"] == _HAS_WEIGHTS
