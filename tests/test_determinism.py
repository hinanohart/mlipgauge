"""S2: the decision pipeline must be bit-for-bit deterministic for a fixed input
(no hidden RNG, no dict-ordering leakage into Q)."""

from __future__ import annotations

import numpy as np

from mlipgauge.core.gauge import decide
from mlipgauge.core.physics_gate import run_physics_gate
from mlipgauge.core.uq import aggregate_uncertainty
from mlipgauge.types import EnsembleForces, TrajectoryWindow


def _fixture():
    rng = np.random.default_rng(123)
    n_frames, n = 4, 3
    base_force = rng.standard_normal((n, 3)) * 0.1
    forces = np.broadcast_to(base_force, (n_frames, n, 3)).copy()
    dx = rng.standard_normal((n, 3)) * 1e-3
    positions = np.stack([t * dx for t in range(n_frames)])
    de = -float(np.sum(base_force * dx))
    energies = np.array([t * de for t in range(n_frames)])
    window = TrajectoryWindow(
        positions=positions,
        forces=forces,
        potential_energy=energies,
        atomic_numbers=np.array([1, 6, 8]),
        timestep_fs=1.0,
    )
    ensemble = EnsembleForces(np.stack([forces, forces + 0.02, forces - 0.01]))
    return window, ensemble


def test_pipeline_deterministic_100x():
    window, ensemble = _fixture()
    results = []
    for _ in range(100):
        uq = aggregate_uncertainty(ensemble)
        gate = run_physics_gate(window)
        d = decide(uq, gate)
        results.append((d.Q, d.verdict, d.active_learning))
    first = results[0]
    assert all(r == first for r in results)
    assert 0.0 <= first[0] <= 1.0
