"""S2: cross-model uncertainty. Golden values hand-computed from the RMS
cross-model force deviation; properties check monotonicity and fail-closed."""

from __future__ import annotations

import numpy as np
import pytest

from mlipgauge.core.uq import UQConfig, aggregate_uncertainty
from mlipgauge.types import EnsembleForces


def test_zero_disagreement_gives_zero_u():
    f = np.ones((3, 2, 4, 3))  # 3 identical models
    res = aggregate_uncertainty(EnsembleForces(f))
    assert res.u_raw == pytest.approx(0.0)
    assert res.u == pytest.approx(0.0)


def test_golden_two_model_single_atom():
    # model0 force 0, model1 force [d,0,0] on one atom, one frame.
    # RMS dev = sqrt( ((d/2)^2 + (d/2)^2)/2 ) = d/2
    d = 0.2
    f = np.zeros((2, 1, 1, 3))
    f[1, 0, 0, 0] = d
    res = aggregate_uncertainty(EnsembleForces(f), UQConfig(scale_ev_per_ang=0.1))
    assert res.u_raw == pytest.approx(d / 2)  # 0.1
    assert res.u == pytest.approx(1.0 - np.exp(-1.0))  # ~0.6321


def test_nonfinite_fails_closed():
    f = np.zeros((2, 2, 1, 3))
    f[0, 0, 0, 0] = np.inf
    res = aggregate_uncertainty(EnsembleForces(f))
    assert res.method == "abstain_nonfinite"
    assert res.u == 1.0
    assert np.isnan(res.per_frame).all()


def test_u_monotone_in_disagreement():
    prev = -1.0
    for d in [0.0, 0.05, 0.1, 0.3, 1.0]:
        f = np.zeros((2, 1, 1, 3))
        f[1, 0, 0, 0] = d
        u = aggregate_uncertainty(EnsembleForces(f)).u
        assert u >= prev
        prev = u


def test_atom_reduce_max_ge_mean():
    rng = np.random.default_rng(0)
    f = rng.standard_normal((3, 2, 5, 3))
    ef = EnsembleForces(f)
    u_mean = aggregate_uncertainty(ef, UQConfig(atom_reduce="mean")).u_raw
    u_max = aggregate_uncertainty(ef, UQConfig(atom_reduce="max")).u_raw
    assert u_max >= u_mean
