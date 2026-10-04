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
    assert names == ["ds_guidance", "pid", "ann", "snn"]
    assert body["defaults"]["steps"] >= 1
    assert body["scene"]["goal"] == [0.0, 0.0, 2.5]


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
