"""Quantify the rate-coded ANN -> SNN transfer (no torch needed).

Reports how well the integrate-and-fire net approximates the trained connectome
ANN over a range of inference horizons (micro-steps per frame).  This is the
evidence behind *not* calling the conversion "lossless": with recurrent weights
and a finite window the SNN is an approximation whose error shrinks with the
window.

    python tools/snn_transfer_eval.py --horizons 1 2 4 8 16 32 64 128
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from drone6dof.connectome import ConnectomeANN, RateCodedConnectomeSNN, SparseRecurrence
from drone6dof.weights import (
    DEFAULT_FIELD_PATH,
    DEFAULT_REF_IO_PATH,
    DEFAULT_WEIGHTS_PATH,
    load_field_weights,
    load_weights,
)

__all__ = ["transfer_metrics", "evaluate", "main"]

_LIMIT = 12.0


def _axis_metrics(a: np.ndarray, s: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    err = s - a
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    spread = float(np.ptp(a))
    nrmse = rmse / (spread + 1e-9)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    if a.std() > 0.0 and s.std() > 0.0:
        pearson = float(np.corrcoef(a, s)[0, 1])
    else:
        pearson = float("nan")
    denom = float(np.linalg.norm(a) * np.linalg.norm(s))
    cosine = float(a @ s / denom) if denom > 0.0 else float("nan")
    gain = float(a @ s / (a @ a)) if (a @ a) > 0.0 else float("nan")
    return {"mae": mae, "rmse": rmse, "nrmse": nrmse, "r2": r2,
            "pearson": pearson, "cosine": cosine, "gain": gain}


def transfer_metrics(ann_out: np.ndarray, snn_out: np.ndarray) -> Dict:
    """Per-dimension and averaged ANN-vs-SNN error metrics for ``(T, D)`` arrays."""
    a = np.asarray(ann_out, dtype=np.float64)
    s = np.asarray(snn_out, dtype=np.float64)
    per_dim = [_axis_metrics(a[:, d], s[:, d]) for d in range(a.shape[1])]
    keys = per_dim[0].keys()
    aggregate = {
        k: float(np.nanmean([p[k] for p in per_dim])) for k in keys
    }
    return {"per_dim": per_dim, "aggregate": aggregate}


def _recurrence(bundle: dict) -> SparseRecurrence:
    edges = np.asarray(bundle["edges"], dtype=np.int64)
    signed = np.abs(np.asarray(bundle["w_mag"], dtype=np.float64)) * np.asarray(bundle["polarity"])[edges[1]]
    return SparseRecurrence(edges, signed, int(bundle["n_neurons"]))


def evaluate(
    horizons: List[int],
    *,
    frames: Optional[int] = None,
    kind: str = "connectome",
    weights_path=None,
    ref_io_path=None,
) -> Dict[str, Dict]:
    """Return ``{horizon: metrics}`` for the ANN vs the SNN at that micro-step count."""
    if kind == "field":
        bundle = load_field_weights(weights_path or DEFAULT_FIELD_PATH)
        ref_io_path = ref_io_path or Path(DEFAULT_FIELD_PATH).with_name("quad6dof_field_ref.npz")
    else:
        bundle = load_weights(weights_path or DEFAULT_WEIGHTS_PATH)
        ref_io_path = ref_io_path or DEFAULT_REF_IO_PATH
    rec = _recurrence(bundle)
    ann = ConnectomeANN(
        bundle["w_in"], bundle["w_out"], rec,
        steps_per_frame=int(bundle.get("connectome_steps", 3)), limit=_LIMIT,
    )
    io = np.load(ref_io_path or DEFAULT_REF_IO_PATH)
    x = np.asarray(io["inputs"], dtype=np.float64)
    if frames:
        x = x[: int(frames)]
    ann_out = np.asarray([ann.forward_input(row) for row in x], dtype=np.float64)

    out: Dict[str, Dict] = {}
    for horizon in horizons:
        snn = RateCodedConnectomeSNN(
            bundle["w_in"], bundle["w_out"], rec,
            micro_steps=int(horizon), v_th=float(bundle.get("v_th", 1.0)), limit=_LIMIT,
            readout_gain=float(bundle.get("readout_gain", 1.0)),
        )
        snn_out = np.asarray([snn.forward_input(row) for row in x], dtype=np.float64)
        metrics = transfer_metrics(ann_out, snn_out)
        # activity statistics over the run
        fired = np.zeros(snn.n_neurons, dtype=bool)
        rates = []
        for _ in range(len(x)):
            fired |= snn.last_spikes.astype(bool)
            rates.append(float(snn.last_telemetry(dt=0.02)["active_frac"]))
        metrics["silent_frac"] = float(1.0 - fired.mean())
        metrics["mean_active_frac"] = float(np.mean(rates))
        out[str(int(horizon))] = metrics
    return out


def _print_table(results: Dict[str, Dict]) -> None:
    print(f"{'T':>5}{'MAE':>9}{'RMSE':>9}{'NRMSE':>8}{'R2':>8}"
          f"{'pearson':>9}{'cosine':>9}{'gain':>8}{'silent':>8}")
    for horizon, m in results.items():
        a = m["aggregate"]
        print(f"{horizon:>5}{a['mae']:>9.4f}{a['rmse']:>9.4f}{a['nrmse']:>8.3f}"
              f"{a['r2']:>8.3f}{a['pearson']:>9.3f}{a['cosine']:>9.3f}"
              f"{a['gain']:>8.3f}{m['silent_frac']:>8.3f}")


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="ANN vs rate-coded SNN transfer metrics")
    ap.add_argument("--kind", choices=["connectome", "field"], default="connectome")
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default=None, help="write the metrics JSON here")
    args = ap.parse_args(argv)

    results = evaluate(args.horizons, frames=args.frames, kind=args.kind)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")
    if args.json:
        print(json.dumps(results, indent=2))
        return 0
    _print_table(results)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
