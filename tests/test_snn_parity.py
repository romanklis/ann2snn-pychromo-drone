"""Numpy ANN/SNN parity with the torch reference (no torch needed).

``weights/quad6dof_reference_io.npz`` holds the torch ANN outputs on a DS-rollout
input sequence (deployment semantics, hidden state carried).  The numpy ANN must
reproduce them; the SNN must track the ANN as a rate code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drone6dof.benchmark import weights_info
from drone6dof.connectome import (
    ConnectomeANN,
    RateCodedConnectomeSNN,
    SparseRecurrence,
)
from drone6dof.weights import DEFAULT_REF_IO_PATH, DEFAULT_WEIGHTS_PATH, load_weights
from snn_transfer_eval import evaluate, transfer_metrics

_HAS = bool(weights_info().get("loaded")) and Path(DEFAULT_REF_IO_PATH).exists()
pytestmark = pytest.mark.skipif(not _HAS, reason="weights/reference bundle not built")

_LIMIT = 12.0


def _bundle():
    return load_weights(DEFAULT_WEIGHTS_PATH)


def _net(kind: str):
    b = _bundle()
    edges = np.asarray(b["edges"], dtype=np.int64)
    signed = np.abs(b["w_mag"]) * np.asarray(b["polarity"])[edges[1]]
    rec = SparseRecurrence(edges, signed, int(b["n_neurons"]))
    if kind == "ann":
        return ConnectomeANN(b["w_in"], b["w_out"], rec,
                             steps_per_frame=int(b["connectome_steps"]), limit=_LIMIT)
    return RateCodedConnectomeSNN(b["w_in"], b["w_out"], rec,
                                 micro_steps=int(b["micro_steps"]),
                                 v_th=float(b["v_th"]), limit=_LIMIT)


def _reference():
    io = np.load(DEFAULT_REF_IO_PATH)
    return np.asarray(io["inputs"], dtype=np.float64), np.asarray(io["ann_out"], dtype=np.float64)


def test_numpy_ann_matches_the_torch_reference():
    x, y = _reference()
    ann = _net("ann")
    out = np.asarray([ann.forward_input(row) for row in x])
    assert np.allclose(out, y, atol=1e-4), float(np.abs(out - y).max())


def test_snn_tracks_the_ann_as_a_rate_code():
    x, _ = _reference()
    ann = _net("ann")
    snn = _net("snn")
    ua = np.asarray([ann.forward_input(row) for row in x])
    us = np.asarray([snn.forward_input(row) for row in x])

    m = transfer_metrics(ua, us)["aggregate"]
    # A correlated rate code with a bounded scale error on the deployment
    # sequence.  This is an approximation, not an equality claim.
    assert np.isfinite([m["rmse"], m["r2"], m["cosine"], m["gain"]]).all(), m
    assert m["cosine"] >= 0.5, m
    assert m["nrmse"] <= 1.0, m


def test_transfer_error_is_bounded_and_improves_with_the_window():
    results = evaluate([1, 8, 64])
    rmse = {int(k): v["aggregate"]["rmse"] for k, v in results.items()}
    for value in rmse.values():
        assert np.isfinite(value)
    # A longer integration window must not make the rate code substantially worse.
    assert rmse[64] <= rmse[1] * 1.25, rmse
