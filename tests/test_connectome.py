"""Connectome topology, ANN and IF-SNN invariants (numpy only)."""

from __future__ import annotations

import numpy as np

from drone6dof.connectome import (
    ConnectomeANN,
    ConnectomeTopology,
    LosslessConnectomeSNN,
    SparseRecurrence,
    DEFAULTS,
)


def test_topology_is_deterministic():
    a = ConnectomeTopology(n_neurons=64, k=8, seed=7)
    b = ConnectomeTopology(n_neurons=64, k=8, seed=7)
    assert np.array_equal(a.edges, b.edges)
    assert np.array_equal(a.polarity, b.polarity)
    assert np.array_equal(a.init_mag, b.init_mag)
    c = ConnectomeTopology(n_neurons=64, k=8, seed=8)
    assert not np.array_equal(a.edges, c.edges)


def test_topology_dales_law_and_counts():
    topo = ConnectomeTopology(n_neurons=1000, k=40, seed=42)
    assert np.unique(topo.polarity).tolist() == [-4.0, 1.0]
    frac = float(np.mean(topo.polarity < 0))
    assert 0.15 <= frac <= 0.25, frac
    assert topo.edges.shape == (2, 1000 * 40)
    assert topo.init_mag.shape == (1000 * 40,)
    assert (topo.init_mag >= 0).all()
    # random fan-in: mean degree is K
    counts = np.bincount(topo.edges[0], minlength=1000)
    assert abs(float(counts.mean()) - 40.0) < 1.0


def test_signed_weights_follow_polarity():
    topo = ConnectomeTopology(n_neurons=32, k=4, seed=3)
    mags = np.full(topo.total, 2.0)
    signed = topo.signed_weights(mags)
    expected = mags * topo.polarity[topo.edges[1]]
    assert np.allclose(signed, expected)
    assert np.allclose(np.sign(signed), np.sign(topo.polarity[topo.edges[1]]))


def test_sparse_matvec_matches_dense():
    topo = ConnectomeTopology(n_neurons=48, k=6, seed=5)
    signed = topo.signed_weights(topo.init_mag)
    rec = SparseRecurrence(topo.edges, signed, topo.n_neurons)
    dense = np.zeros((topo.n_neurons, topo.n_neurons))
    np.add.at(dense, (topo.edges[0], topo.edges[1]), signed)
    x = np.random.default_rng(0).standard_normal(topo.n_neurons)
    assert np.allclose(rec.matvec(x), dense @ x)


def _nets(kind: str, n_neurons: int = 64, k: int = 8):
    topo = ConnectomeTopology(n_neurons=n_neurons, k=k, seed=1)
    rec = SparseRecurrence(topo.edges, topo.signed_weights(topo.init_mag), n_neurons)
    rng = np.random.default_rng(0)
    w_in = rng.standard_normal((n_neurons, 13)) * 0.1
    w_out = rng.standard_normal((3, n_neurons)) * 0.1
    if kind == "ann":
        return ConnectomeANN(w_in, w_out, rec, steps_per_frame=3, limit=12.0)
    return LosslessConnectomeSNN(w_in, w_out, rec, micro_steps=10, v_th=1.0, limit=12.0)


def test_ann_forward_shape_and_finiteness():
    net = _nets("ann")
    u = net.forward_input(np.ones(13))
    assert u.shape == (3,)
    assert np.isfinite(u).all()
    assert np.all(np.abs(u) <= 12.0 + 1e-9)


def test_snn_forward_invariants():
    net = _nets("snn")
    u = net.forward_input(np.ones(13))
    assert u.shape == (3,)
    assert (net.v >= -1e-9).all()               # membrane stays non-negative
    assert set(np.unique(net.s)).issubset({0.0, 1.0})
    assert net.last_spikes.dtype == np.uint8
    tel = net.last_telemetry(dt=0.02)
    for key in ("spike_rate_hz", "active_frac", "spikes_this_frame", "mean_v"):
        assert key in tel
    assert np.isfinite(u).all()


def test_snn_is_deterministic():
    a = _nets("snn")
    b = _nets("snn")
    x = np.linspace(-1, 1, 13)
    ua = np.array([a.forward_input(x) for _ in range(5)])
    ub = np.array([b.forward_input(x) for _ in range(5)])
    assert np.array_equal(ua, ub)


def test_defaults_match_upstream():
    assert DEFAULTS["n_neurons"] == 1000
    assert DEFAULTS["k"] == 40
    assert DEFAULTS["inhibitory_fraction"] == 0.20
    assert DEFAULTS["connectome_steps"] == 3
    assert DEFAULTS["micro_steps"] == 10
    assert DEFAULTS["v_th"] == 1.0
