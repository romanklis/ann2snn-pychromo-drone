"""Flask API over the numpy benchmark plus the built Vite frontend.

Data-only: the 3-D visualization stays in PyChrono (`make demo`).  Run with
``python -m server.app`` (dev) or ``gunicorn server.wsgi:application``.
"""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from drone6dof.benchmark import (
    available_controllers,
    controller_catalogue,
    run_benchmark,
    weights_info,
)
from drone6dof.config import (
    CONTROL_LIMIT,
    INIT_STATE,
    MAP_SOURCE_DEFAULT,
    PRESET_SCENES,
    QUAD_SCENE,
    SLAM,
    STEPS,
)
from drone6dof.params import DT

_REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = Path(os.environ.get("DRONE6DOF_WEB_ROOT", _REPO_ROOT / "server" / "static"))

MAX_STEPS = 2000


def _clamp_steps(value) -> int:
    try:
        steps = int(value)
    except (TypeError, ValueError):
        steps = STEPS
    return max(1, min(MAX_STEPS, steps))


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)

    @app.get("/api/health")
    def health():
        weights = weights_info()
        return jsonify(
            {
                "status": "ok",
                "example": "quad6dof",
                "trained": bool(weights.get("loaded")),
                "weights": weights,
                "steps_default": STEPS,
                "dt": DT,
            }
        )

    @app.get("/api/controllers")
    def controllers():
        return jsonify(
            {
                "catalogue": controller_catalogue(),
                "available": available_controllers(),
                "weights": weights_info(),
                "scene": QUAD_SCENE.to_dict(),
                "scenes": sorted(PRESET_SCENES),
                "map_sources": ["truth", "slam"],
                "slam": dict(SLAM),
                "defaults": {
                    "steps": STEPS,
                    "dt": DT,
                    "seed": 0,
                    "control_limit": CONTROL_LIMIT,
                    "init_state": list(INIT_STATE),
                    "map_source": MAP_SOURCE_DEFAULT,
                },
            }
        )

    def _run(payload: dict) -> dict:
        names = payload.get("controllers")
        if names is None and payload.get("controller"):
            names = [payload["controller"]]
        steps = _clamp_steps(payload.get("steps", STEPS))
        seed = int(payload.get("seed", 0) or 0)
        goal = payload.get("goal")
        scene_name = payload.get("scene")
        map_source = payload.get("map_source", MAP_SOURCE_DEFAULT)
        if map_source not in ("truth", "slam"):
            raise ValueError("map_source must be 'truth' or 'slam'")
        return run_benchmark(
            names, steps=steps, seed=seed, goal=goal, scene_name=scene_name,
            map_source=map_source,
        )

    @app.post("/api/simulate")
    def simulate():
        payload = request.get_json(silent=True) or {}
        if not payload.get("controller") and not payload.get("controllers"):
            return jsonify({"error": "provide 'controller' or 'controllers'"}), 400
        try:
            return jsonify(_run(payload))
        except (KeyError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/benchmark")
    def benchmark():
        payload = request.get_json(silent=True) or {}
        try:
            return jsonify(_run(payload))
        except (KeyError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    # ---- static frontend -------------------------------------------------- #
    @app.get("/")
    def index():
        return send_from_directory(WEB_ROOT, "index.html")

    @app.get("/extended")
    @app.get("/extended.html")
    def extended():
        return send_from_directory(WEB_ROOT, "extended.html")

    @app.get("/assets/<path:filename>")
    def assets(filename):
        return send_from_directory(WEB_ROOT / "assets", filename)

    @app.get("/favicon.ico")
    def favicon():
        return ("", 204)

    @app.errorhandler(404)
    def not_found(_err):
        if request.path.startswith("/api/"):
            return jsonify({"error": "not found", "path": request.path}), 404
        return jsonify({"error": "not found"}), 404

    @app.after_request
    def _cache_headers(response):
        # HTML must never be cached: a rebuilt image otherwise keeps serving an old
        # bundle and the UI changes (e.g. the goal command) silently do nothing.
        if request.path.startswith("/assets/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            response.headers["Cache-Control"] = "no-store"
        return response

    return app


app = create_app()


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="drone6dof dashboard server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
