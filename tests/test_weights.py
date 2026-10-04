"""Weight-bundle loading and fingerprint guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from drone6dof.benchmark import weights_info
from drone6dof.config import SENSOR
from drone6dof.policy import policy_input_dim
from drone6dof.weights import (
    DEFAULT_WEIGHTS_PATH,
    WeightsMismatch,
    WeightsMissing,
    default_fields,
    load_weights,
)

_HAS_BUNDLE = bool(weights_info().get("loaded"))
pytestmark = pytest.mark.skipif(not _HAS_BUNDLE, reason="weights bundle not built")


def test_bundle_loads_with_expected_shapes():
    bundle = load_weights(DEFAULT_WEIGHTS_PATH, expect_fields=default_fields())
    n_in = policy_input_dim(SENSOR)
    assert bundle["w_in"].shape == (1000, n_in)
    assert bundle["w_out"].shape == (3, 1000)
    assert bundle["w_mag"].shape == (40000,)
    assert bundle["edges"].shape == (2, 40000)
    assert bundle["polarity"].shape == (1000,)
    assert bundle["n_neurons"] == 1000
    assert bundle["k"] == 40
    assert bundle["n_in"] == n_in
    assert bundle["n_out"] == 3
    assert bundle["connectome_steps"] == 3
    assert bundle["micro_steps"] == 10
    assert bundle["v_th"] == 1.0


def test_fingerprint_mismatch_is_rejected(tmp_path):
    fields = default_fields()
    fields = {**fields, "control_limit": 99.0}
    with pytest.raises(WeightsMismatch):
        load_weights(DEFAULT_WEIGHTS_PATH, expect_fields=fields)


def test_missing_bundle_raises(tmp_path):
    with pytest.raises(WeightsMissing):
        load_weights(tmp_path / "absent.npz")
