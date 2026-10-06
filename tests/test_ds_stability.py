"""Nominal DS stability: eigenvalues and a quadratic Lyapunov argument.

This establishes the stability of the *nominal* (obstacle-free) linear DS only —
not global stability of the saturated nonlinear drone closed loop.
"""

from __future__ import annotations

import numpy as np
import pytest

from drone6dof.field import DEFAULT_GAINS, FieldConfig, nominal_ds


def _linearisation():
    g = DEFAULT_GAINS
    return np.array([
        [-g["radial"], -g["curl"], 0.0],
        [g["curl"], -g["radial"], 0.0],
        [0.0, 0.0, -g["climb"]],
    ])


def test_nominal_ds_linearisation_eigenvalues_are_stable():
    g = DEFAULT_GAINS
    eig = np.linalg.eigvals(_linearisation())
    assert np.all(eig.real < 0.0), eig
    # horizontal pair -g_r ± i g_c, vertical pole -climb
    expected = np.array([-g["radial"] + 1j * g["curl"],
                         -g["radial"] - 1j * g["curl"],
                         -g["climb"]])
    assert np.allclose(np.sort_complex(eig), np.sort_complex(expected))


def test_lyapunov_derivative_is_negative_inside_the_speed_cap():
    g = DEFAULT_GAINS
    for e in ([0.5, 0.3, 0.4], [0.05, -0.02, 0.1], [0.8, 0.0, 0.2]):
        e = np.asarray(e, dtype=float)
        v = nominal_ds(e)
        vdot = float(e @ v)                     # V = ½‖e‖², V̇ = eᵀAe
        expected = -g["radial"] * (e[0] ** 2 + e[1] ** 2) - g["climb"] * e[2] ** 2
        assert vdot < 0.0
        assert vdot == pytest.approx(expected, rel=1e-6)   # cap does not bind here


def test_lyapunov_derivative_stays_negative_when_the_cap_binds():
    g = DEFAULT_GAINS
    e = np.array([4.0, 3.0, 1.0])               # large error: cap binds
    v = nominal_ds(e)
    uncapped = np.array([
        -g["radial"] * e[0] - g["curl"] * e[1],
        -g["radial"] * e[1] + g["curl"] * e[0],
        -g["climb"] * e[2],
    ])
    scale = g["cap"] / np.linalg.norm(uncapped)
    assert scale < 1.0
    assert np.allclose(v, scale * uncapped)
    assert float(e @ v) == pytest.approx(scale * float(e @ uncapped))
    assert float(e @ v) < 0.0                    # still a descent direction


def test_speed_cap_preserves_direction():
    e = np.array([4.0, 3.0, 1.0])               # large error: cap binds
    v = nominal_ds(e)
    assert np.linalg.norm(v) == pytest.approx(DEFAULT_GAINS["cap"], rel=1e-6)
    uncapped = np.array([
        -DEFAULT_GAINS["radial"] * e[0] - DEFAULT_GAINS["curl"] * e[1],
        -DEFAULT_GAINS["radial"] * e[1] + DEFAULT_GAINS["curl"] * e[0],
        -DEFAULT_GAINS["climb"] * e[2],
    ])
    assert np.allclose(v / np.linalg.norm(v), uncapped / np.linalg.norm(uncapped))
