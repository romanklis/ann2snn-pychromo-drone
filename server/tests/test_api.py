"""Flask API contract (skipped unless Flask is installed, e.g. the dashboard image)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

flask = pytest.importorskip("flask")

from server.app import create_app  # noqa: E402

WEB_ROOT = Path(__file__).resolve().parents[1] / "static"


@pytest.fixture()
def client():
    app = create_app()
    app.testing = True
    return app.test_client()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["example"] == "quad6dof"
    assert "trained" in body and "weights" in body


def test_controllers_catalogue(client):
    r = client.get("/api/controllers")
    assert r.status_code == 200
    body = r.get_json()
    names = [c["name"] for c in body["catalogue"]]
    assert names == ["ds_guidance", "pid", "ann", "snn", "field_ann", "field_snn"]
    meta = {c["name"]: c for c in body["catalogue"]}
    assert meta["pid"]["display"] is False                 # hidden from the dashboard
    assert meta["ds_guidance"]["sensor"] is False
    assert meta["ann"]["sensor"] is True and meta["ann"]["spiking"] is False
    assert meta["snn"]["spiking"] is True
    assert meta["field_ann"]["sensor"] is True and meta["field_ann"]["spiking"] is False
    assert meta["field_snn"]["spiking"] is True and meta["field_snn"]["sensor"] is True
    assert body["defaults"]["steps"] >= 1
    assert body["scene"]["goal"] == [0.0, 0.0, 2.5]
    assert body["map_sources"] == ["truth", "slam"]


def test_benchmark_rejects_bad_map_source(client):
    r = client.post(
        "/api/benchmark",
        json={"controllers": ["ds_guidance"], "steps": 10, "map_source": "nope"},
    )
    assert r.status_code == 400


def test_benchmark_window_is_applied(client):
    r = client.post(
        "/api/benchmark",
        json={"controllers": ["ds_guidance"], "steps": 200, "from": 1.0, "to": 2.0},
    )
    assert r.status_code == 200
    body = r.get_json()
    res = body["results"]["ds_guidance"]
    assert body["window"] == {"from": 1.0, "to": 2.0}
    assert abs(res["t"][0] - 1.0) < 0.05
    assert abs(res["t"][-1] - 2.0) < 0.05


def test_benchmark_bad_window_is_400(client):
    assert client.post(
        "/api/benchmark",
        json={"controllers": ["ds_guidance"], "steps": 50, "from": 2.0, "to": 1.0},
    ).status_code == 400
    assert client.post(
        "/api/benchmark",
        json={"controllers": ["ds_guidance"], "from": "abc"},
    ).status_code == 400


def test_benchmark_slam_includes_map(client):
    r = client.post(
        "/api/benchmark",
        json={"controllers": ["field_ann"], "steps": 60, "map_source": "slam"},
    )
    if r.status_code != 200:
        pytest.skip("field bundle not available")
    body = r.get_json()
    res = body["results"].get("field_ann")
    if res is None:
        pytest.skip("field bundle not available")
    assert body["map_source"] == "slam"
    assert res["map"] is not None and res["slam"] is not None
    assert len(res["map"]["shape"]) == 2
    assert 0.0 <= res["slam"]["surface_coverage"] <= 1.0
    steps = res["slam"]["replan_steps"]
    assert steps == sorted(steps)
    assert all(isinstance(v, int) and 0 <= v <= 60 for v in steps)


def test_benchmark_runs_selected_controllers(client):
    r = client.post("/api/benchmark", json={"controllers": ["ds_guidance", "pid"], "steps": 20})
    assert r.status_code == 200
    body = r.get_json()
    assert set(body["results"]) == {"ds_guidance", "pid"}
    assert body["results"]["ds_guidance"]["metrics"]["steps"] == 20


def test_simulate_single(client):
    r = client.post("/api/simulate", json={"controller": "ds_guidance", "steps": 10})
    assert r.status_code == 200
    assert "ds_guidance" in r.get_json()["results"]


def test_benchmark_goal_is_used(client):
    r = client.post(
        "/api/benchmark",
        json={"controllers": ["ds_guidance"], "steps": 20, "goal": [1.0, 1.0, 1.5]},
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["goal"] == [1.0, 1.0, 1.5]
    assert body["scene"]["goal"] == [1.0, 1.0, 1.5]


def test_benchmark_scene_is_used(client):
    r = client.post(
        "/api/benchmark",
        json={"controllers": ["ds_guidance"], "steps": 20, "scene": "boxes"},
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["scene_name"] == "boxes"
    assert len(body["scene"]["obstacles"]) == 2


def test_unknown_scene_is_400(client):
    r = client.post(
        "/api/benchmark",
        json={"controllers": ["ds_guidance"], "steps": 20, "scene": "nope"},
    )
    assert r.status_code == 400


def test_out_of_bounds_goal_is_400(client):
    r = client.post(
        "/api/benchmark",
        json={"controllers": ["ds_guidance"], "steps": 20, "goal": [0.0, 0.0, 99.0]},
    )
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_benchmark_body_has_no_nan_tokens(client):
    """Browsers reject `NaN`/`Infinity`; the payload must be strict JSON."""
    r = client.post("/api/benchmark", json={"controllers": ["ds_guidance"], "steps": 20})
    text = r.get_data(as_text=True)
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


def test_unknown_controller_is_400(client):
    r = client.post("/api/benchmark", json={"controllers": ["nope"], "steps": 10})
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_steps_are_clamped(client):
    r = client.post("/api/benchmark", json={"controllers": ["ds_guidance"], "steps": 99999})
    assert r.status_code == 200
    assert r.get_json()["steps"] == 2000


def test_api_404_is_json(client):
    r = client.get("/api/does-not-exist")
    assert r.status_code == 404
    assert r.is_json


@pytest.mark.skipif(not (WEB_ROOT / "index.html").exists(), reason="frontend not built")
def test_static_pages(client):
    assert client.get("/").status_code == 200
    assert client.get("/extended").status_code == 200
