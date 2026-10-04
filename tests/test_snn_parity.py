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
    LosslessConnectomeSNN,
    SparseRecurrence,
)
from drone6dof.weights import DEFAULT_REF_IO_PATH, DEFAULT_WEIGHTS_PATH, load_weights

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
    return LosslessConnectomeSNN(b["w_in"], b["w_out"], rec,
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

    checkable = 0
    for axis in range(ua.shape[1]):
        a = ua[:, axis]
        s = us[:, axis]
        if np.ptp(a) < 0.10 * _LIMIT or np.ptp(s) < 0.10 * _LIMIT:
            continue
        checkable += 1
        k = float(np.dot(a, s) / (np.dot(a, a) + 1e-12))
        pred = k * a
        ss_res = float(np.sum((s - pred) ** 2))
        ss_tot = float(np.sum((s - s.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        assert 0.5 <= k <= 2.0, (axis, k)
        assert r2 >= 0.5, (axis, r2)
    if checkable == 0:
        pytest.skip("policy near-silent on every axis")
