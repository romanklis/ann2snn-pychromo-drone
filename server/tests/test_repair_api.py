"""Contract tests for the read-only repair inspector API."""

from __future__ import annotations

import json

import numpy as np
import pytest

flask = pytest.importorskip("flask")

from server.app import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    root = tmp_path / "repair"
    run = root / "run-1"
    (run / "baseline").mkdir(parents=True)
    (run / "cycles").mkdir()
    (run / "final").mkdir()
    (run / "manifest.json").write_text(json.dumps({
        "run_id": "run-1",
        "scenario": {"name": "slalom"},
        "proposer": "template",
        "cycles": [],
    }))
    (run / "baseline" / "metrics.json").write_text(json.dumps({
        "practice": {"success_rate": 0.0},
        "regression": {"success_rate": 1.0},
    }))
    (run / "baseline" / "trace.json").write_text(json.dumps({"t": [0.0], "trajectory": [[0, 0, 0]]}))
    (run / "cycles" / "0001.json").write_text(json.dumps({
        "cycle": 1, "accepted": True, "score": 0.5,
        "practice": {"metrics": {"success_rate": 0.5}},
        "regression": {"metrics": {"success_rate": 1.0}},
    }))
    (run / "cycles" / "0001.trace.json").write_text(json.dumps({
        "t": [0.0, 1.0],
        "trajectory": [[0, 0, 0], [1, 1, 1]],
        "goal_dist": [2.0, 1.0],
        "clearance": [1.0, 0.5],
    }))
    (run / "cycles" / "0001.diff.json").write_text(json.dumps({
        "top_units": [{"unit": 3, "w_in_norm": 1.0, "w_out_norm": 0.2, "bias": 0.0}],
        "changed_units": [3],
        "n_added_edges": 1,
    }))
    np.savez(
        run / "baseline" / "activity.npz",
        silent=np.array([1, 0, 1], dtype=np.uint8),
        rare=np.array([0, 1, 0], dtype=np.uint8),
        spare=np.array([1, 1, 1], dtype=np.uint8),
        critical=np.array([0, 0, 0], dtype=np.uint8),
        spike_counts=np.array([0, 10, 0], dtype=np.int64),
        spike_frames=np.array([0, 4, 0], dtype=np.int64),
        mean_rate=np.array([0.0, 0.1, 0.0]),
        contribution=np.array([0.0, 0.2, 0.0]),
        w_in_norm=np.array([1.0, 2.0, 1.0]),
        w_out_norm=np.array([0.0, 1.0, 0.0]),
        bias=np.zeros(3),
    )
    (run / "events.jsonl").write_text(json.dumps({"type": "start"}) + "\n")

    no_shift = root / "no-shift"
    no_shift.mkdir()
    (no_shift / "manifest.json").write_text(json.dumps({
        "run_id": "no-shift",
        "status": "no_shift",
        "suggested_scenario": "goal_behind_wall",
        "probe": {
            "status": "no_shift",
            "suggested_scenario": "goal_behind_wall",
            "scenarios": [{"scenario": "goal_behind_wall", "gap": 0.4}],
        },
    }))
    monkeypatch.setenv("DRONE6DOF_REPAIR_ROOT", str(root))
    app = create_app()
    app.testing = True
    return app.test_client()


def test_list_runs(client):
    r = client.get("/api/repair/runs")
    assert r.status_code == 200
    assert r.get_json()["runs"][0]["run_id"] == "run-1"


def test_get_run_and_cycle(client):
    r = client.get("/api/repair/runs/run-1")
    assert r.status_code == 200
    body = r.get_json()
    assert body["run_id"] == "run-1"
    assert body["baseline_trace"]["trajectory"] == [[0, 0, 0]]
    assert len(body["cycles"]) == 1

    r = client.get("/api/repair/runs/run-1/cycles/1")
    assert r.status_code == 200
    assert r.get_json()["accepted"] is True

    r = client.get("/api/repair/runs/run-1/weights/1")
    assert r.status_code == 200
    assert r.get_json()["n_added_edges"] == 1


def test_no_shift_run_is_visible(client):
    r = client.get("/api/repair/runs/no-shift")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "no_shift"
    assert body["probe"]["suggested_scenario"] == "goal_behind_wall"
    assert client.get("/api/repair/runs/no-shift/traces").status_code == 200


def test_traces(client):
    r = client.get("/api/repair/runs/run-1/traces")
    assert r.status_code == 200
    body = r.get_json()
    assert body["baseline"]["trajectory"] == [[0, 0, 0]]
    assert body["cycles"]["1"]["trajectory"][-1] == [1, 1, 1]


def test_activity_and_events(client):
    r = client.get("/api/repair/runs/run-1/activity")
    assert r.status_code == 200
    assert r.get_json()["silent"] == [1, 0, 1]

    r = client.get("/api/repair/runs/run-1/events?since=0")
    assert r.status_code == 200
    assert r.get_json()["events"][0]["type"] == "start"


def test_bad_ids_and_missing(client):
    assert client.get("/api/repair/runs/..%2Fetc").status_code in (400, 404)
    assert client.get("/api/repair/runs/nope").status_code == 404
    assert client.get("/api/repair/runs/run-1/cycles/99").status_code == 404
