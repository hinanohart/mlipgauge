"""S1: the IR contract. Construction must validate shapes and reject malformed
windows (fail-closed begins at the type boundary)."""

from __future__ import annotations

import numpy as np
import pytest

from mlipgauge.types import (
    EnsembleForces,
    GateResult,
    GaugeDecision,
    TrajectoryWindow,
    UQResult,
    Verdict,
)


def make_window(f=4, n=3, **kw):
    rng = np.random.default_rng(0)
    return TrajectoryWindow(
        positions=rng.standard_normal((f, n, 3)),
        forces=rng.standard_normal((f, n, 3)),
        potential_energy=rng.standard_normal(f),
        atomic_numbers=np.arange(1, n + 1),
        timestep_fs=1.0,
        **kw,
    )


def test_window_basic_shapes():
    w = make_window(f=5, n=2)
    assert w.n_frames == 5
    assert w.n_atoms == 2
    assert w.finite


def test_window_optional_fields():
    w = make_window(
        f=3,
        n=2,
        cell=np.eye(3)[None].repeat(3, 0),
        stress=np.zeros((3, 3, 3)),
        kinetic_energy=np.ones(3),
    )
    assert w.cell.shape == (3, 3, 3)
    assert w.stress.shape == (3, 3, 3)
    assert w.kinetic_energy.shape == (3,)


@pytest.mark.parametrize(
    "bad",
    [
        {"positions": np.zeros((4, 3))},  # wrong ndim
        {"forces": np.zeros((4, 2, 3))},  # mismatched N
        {"potential_energy": np.zeros(3)},  # wrong F
        {"timestep_fs": 0.0},  # non-positive dt
        {"timestep_fs": -1.0},
    ],
)
def test_window_rejects_malformed(bad):
    kw = dict(
        positions=np.zeros((4, 3, 3)),
        forces=np.zeros((4, 3, 3)),
        potential_energy=np.zeros(4),
        atomic_numbers=np.arange(1, 4),
        timestep_fs=1.0,
    )
    kw.update(bad)
    with pytest.raises(ValueError):
        TrajectoryWindow(**kw)


def test_window_finite_probe_detects_nan():
    rng = np.random.default_rng(1)
    f = rng.standard_normal((2, 2, 3))
    f[0, 0, 0] = np.nan
    w = TrajectoryWindow(
        positions=np.zeros((2, 2, 3)),
        forces=f,
        potential_energy=np.zeros(2),
        atomic_numbers=np.array([1, 2]),
        timestep_fs=1.0,
    )
    assert not w.finite


def test_ensemble_needs_two_models():
    with pytest.raises(ValueError):
        EnsembleForces(np.zeros((1, 4, 3, 3)))
    ef = EnsembleForces(np.zeros((3, 4, 3, 3)))
    assert ef.n_models == 3


def test_uq_range_enforced():
    with pytest.raises(ValueError):
        UQResult(u_raw=0.1, u=1.5, per_frame=np.zeros(2), method="x")
    UQResult(u_raw=0.1, u=0.5, per_frame=np.zeros(2), method="x")


def test_gate_hard_valid_binary():
    with pytest.raises(ValueError):
        GateResult(hard_valid=2, hard_checks={}, soft_scores={})
    GateResult(hard_valid=1, hard_checks={"a": True}, soft_scores={"b": 0.5})


def test_gauge_decision_q_range():
    uq = UQResult(u_raw=0.0, u=0.0, per_frame=np.zeros(1), method="x")
    gr = GateResult(hard_valid=1, hard_checks={}, soft_scores={})
    with pytest.raises(ValueError):
        GaugeDecision(Q=2.0, verdict=Verdict.TRUST, uq=uq, gate=gr, active_learning=False)
    d = GaugeDecision(Q=0.9, verdict=Verdict.TRUST, uq=uq, gate=gr, active_learning=False)
    assert d.verdict is Verdict.TRUST
