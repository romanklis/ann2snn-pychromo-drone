"""Compare controllers on one scene: performance, activity and parameter counts.

    python tools/compare_controllers.py --scene pillar
    python tools/compare_controllers.py --scene boxes --steps 800 --json
"""

from __future__ import annotations

import argparse
import json

from drone6dof.benchmark import available_controllers, run_benchmark
from drone6dof.config import STEPS

_ORDER = ["ds_guidance", "pid", "ann", "snn", "field_ann", "field_snn"]


def parameter_counts() -> dict:
    from drone6dof.weights import (
        DEFAULT_FIELD_PATH,
        DEFAULT_WEIGHTS_PATH,
        load_field_weights,
        load_weights,
    )

    out = {}
    try:
        b = load_weights(DEFAULT_WEIGHTS_PATH)
        out["ann"] = out["snn"] = int(b["w_in"].size + b["w_out"].size + b["w_mag"].size)
    except Exception:
        pass
    try:
        f = load_field_weights(DEFAULT_FIELD_PATH)
        out["field_ann"] = out["field_snn"] = int(
            f["w_in"].size + f["w_out"].size + f["w_mag"].size
        )
    except Exception:
        pass
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare drone controllers")
    ap.add_argument("--scene", default="pillar")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--goal", nargs=3, type=float, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    avail = available_controllers()
    names = [n for n in _ORDER if avail.get(n, {}).get("available")]
    report = run_benchmark(names, steps=args.steps, scene_name=args.scene, goal=args.goal)
    counts = parameter_counts()

    rows = []
    def rnd(v, n=3):
        if v is None or v != v:
            return None
        return round(float(v), n)

    for name in report["controllers"]:
        m = report["results"][name]["metrics"]
        hz = m.get("spike_rate_hz")
        rows.append({
            "controller": name,
            "collisions": m["collisions"],
            "clearance_min_m": rnd(m["clearance_min_m"], 4),
            "final_goal_dist_m": rnd(m["final_goal_dist_m"], 4),
            "reached_goal": m["reached_goal"],
            "trajectory_length_m": rnd(m["trajectory_length_m"]),
            "command_smoothness": rnd(m["command_smoothness"], 4),
            "latency_ms": rnd(m["latency_ms"]),
            "spike_rate_hz": rnd(hz, 1) if report["results"][name]["spiking"] else None,
            "params": counts.get(name),
        })
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"scene={report['scene_name']} goal={report['goal']} steps={report['steps']}")
    print(f"{'controller':<11}{'coll':>5}{'clr_min':>9}{'final':>8}{'reach':>7}"
          f"{'len':>8}{'smooth':>8}{'lat_ms':>8}{'Hz':>7}{'params':>9}")
    def fmt(v, n=3):
        return "-" if v is None else f"{v:.{n}f}"

    for r in rows:
        par = "-" if r["params"] is None else str(r["params"])
        hz = "-" if r["spike_rate_hz"] is None else f"{r['spike_rate_hz']:.0f}"
        print(f"{r['controller']:<11}{r['collisions']:>5}{fmt(r['clearance_min_m'], 3):>9}"
              f"{fmt(r['final_goal_dist_m'], 3):>8}{str(r['reached_goal']):>7}"
              f"{fmt(r['trajectory_length_m'], 2):>8}{fmt(r['command_smoothness']):>8}"
              f"{fmt(r['latency_ms']):>8}{hz:>7}{par:>9}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
