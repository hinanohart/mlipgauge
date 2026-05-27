"""S5: high-level gauge API, trajectory streaming, and the active-learning queue."""

from __future__ import annotations

import numpy as np

from mlipgauge.backends.mock import MockBackend
from mlipgauge.core.al_trigger import ActiveLearningQueue
from mlipgauge.gauge_api import gauge_trajectory_from_backends, gauge_window
from mlipgauge.types import (
    EnsembleForces,
    GateResult,
    GaugeDecision,
    TrajectoryWindow,
    UQResult,
    Verdict,
)


def _clean_traj(n_frames=6, n=3, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, 3)) * 0.2
    step = rng.standard_normal((n, 3)) * 1e-3
    return np.stack([base + t * step for t in range(n_frames)]), np.array([1, 6, 8])


def test_gauge_window_trusts_clean_low_disagreement():
    traj, z = _clean_traj()
    primary = MockBackend(k=1.5)
    from mlipgauge.backends.runner import evaluate_ensemble_forces, evaluate_window

    w = evaluate_window(primary, traj, z)
    ef = evaluate_ensemble_forces([MockBackend(1.5), MockBackend(1.51), MockBackend(1.49)], traj, z)
    d = gauge_window(w, ef)
    assert d.verdict is Verdict.TRUST and d.Q > 0.7


def test_gauge_window_halts_on_hard_violation():
    # build a window with an energy-force inconsistency
    pos = np.zeros((3, 2, 3))
    pos[1, 0, 0] = 0.01
    pos[2, 0, 0] = 0.02
    forces = np.zeros((3, 2, 3))  # zero forces but...
    pe = np.array([0.0, -1.0, -2.0])  # ...energy changes a lot -> inconsistent
    w = TrajectoryWindow(
        positions=pos,
        forces=forces,
        potential_energy=pe,
        atomic_numbers=np.array([1, 1]),
        timestep_fs=1.0,
    )
    ef = EnsembleForces(np.zeros((2, 3, 2, 3)))
    d = gauge_window(w, ef)
    assert d.verdict is Verdict.HALT and d.Q == 0.0
    assert d.active_learning is True


def test_trajectory_clean_run_trusts_all_and_empty_queue():
    traj, z = _clean_traj(n_frames=8)
    primary = MockBackend(k=1.5)
    ens = [MockBackend(1.5), MockBackend(1.501), MockBackend(1.499)]
    decisions, queue = gauge_trajectory_from_backends(primary, ens, traj, z, window_size=4)
    assert len(decisions) == 5  # 8 - 4 + 1
    assert all(d.verdict is Verdict.TRUST for d in decisions)
    assert len(queue) == 0


def test_trajectory_high_disagreement_flags_and_queues():
    traj, z = _clean_traj(n_frames=6)
    primary = MockBackend(k=1.5)
    ens = [MockBackend(0.2), MockBackend(5.0)]  # wildly different -> high uncertainty
    decisions, queue = gauge_trajectory_from_backends(primary, ens, traj, z, window_size=4)
    assert any(d.verdict is Verdict.FLAG for d in decisions)
    assert len(queue) >= 1
    rec = queue.export()[0]
    assert {"window_id", "frame_index", "Q", "verdict", "positions"} <= set(rec)


def test_al_queue_dedups_and_respects_max():
    uq = UQResult(u_raw=1.0, u=0.9, per_frame=np.array([0.1, 0.9]), method="mean")
    gate = GateResult(hard_valid=1, hard_checks={"x": True}, soft_scores={})
    d = GaugeDecision(Q=0.1, verdict=Verdict.FLAG, uq=uq, gate=gate, active_learning=True)
    w = TrajectoryWindow(
        positions=np.zeros((2, 2, 3)),
        forces=np.zeros((2, 2, 3)),
        potential_energy=np.zeros(2),
        atomic_numbers=np.array([1, 1]),
        timestep_fs=1.0,
    )
    q = ActiveLearningQueue(max_size=1)
    assert q.consider(d, w, "a") is True
    assert q.consider(d, w, "a") is False  # dedup
    assert q.consider(d, w, "b") is False  # max_size reached
    assert len(q) == 1
    # worst frame should be the high-uncertainty frame (index 1)
    assert q.export()[0]["frame_index"] == 1


def test_al_queue_ignores_non_flagged():
    uq = UQResult(u_raw=0.0, u=0.0, per_frame=np.zeros(2), method="mean")
    gate = GateResult(hard_valid=1, hard_checks={"x": True}, soft_scores={})
    d = GaugeDecision(Q=0.95, verdict=Verdict.TRUST, uq=uq, gate=gate, active_learning=False)
    w = TrajectoryWindow(
        positions=np.zeros((2, 2, 3)),
        forces=np.zeros((2, 2, 3)),
        potential_energy=np.zeros(2),
        atomic_numbers=np.array([1, 1]),
        timestep_fs=1.0,
    )
    q = ActiveLearningQueue()
    assert q.consider(d, w, "a") is False
    assert len(q) == 0
