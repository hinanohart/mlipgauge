"""S3: the mock backend and the end-to-end pipeline (backend -> IR -> gate -> uq
-> decision) on a GPU-free, deterministic path. Also exercises the fail-closed
license allow-list."""

from __future__ import annotations

import numpy as np
import pytest

from mlipgauge.backends.base import LicenseError, load_backend
from mlipgauge.backends.mock import MockBackend
from mlipgauge.backends.runner import evaluate_ensemble_forces, evaluate_window
from mlipgauge.core.gauge import decide
from mlipgauge.core.physics_gate import run_physics_gate
from mlipgauge.core.uq import aggregate_uncertainty
from mlipgauge.types import Verdict


def test_mock_is_conservative():
    mb = MockBackend(k=2.0)
    x = np.array([[0.3, 0.0, 0.0], [0.0, -0.4, 0.0]])
    pred = mb.predict(x, np.array([1, 1]))
    np.testing.assert_allclose(pred.forces, -2.0 * x)  # F = −k x
    assert pred.energy == pytest.approx(0.5 * 2.0 * np.sum(x * x))
    np.testing.assert_allclose(pred.stress, pred.stress.T)  # symmetric


def _trajectory(n_frames=4, n=3, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, 3)) * 0.2
    steps = rng.standard_normal((n, 3)) * 1e-3
    return np.stack([base + t * steps for t in range(n_frames)])


def test_harmonic_window_passes_gate_exactly():
    # trapezoidal work is exact for a linear (harmonic) force => residual 0
    traj = _trajectory()
    w = evaluate_window(MockBackend(k=1.5), traj, np.array([1, 6, 8]))
    r = run_physics_gate(w)
    assert r.hard_checks["energy_force_consistency"] is True
    assert r.hard_checks["stress_symmetry"] is True
    assert r.hard_valid == 1


def test_full_pipeline_trusts_clean_low_disagreement_run():
    traj = _trajectory()
    z = np.array([1, 6, 8])
    ensemble = [MockBackend(k=1.50), MockBackend(k=1.51), MockBackend(k=1.49)]
    w = evaluate_window(ensemble[0], traj, z)
    ef = evaluate_ensemble_forces(ensemble, traj, z)
    decision = decide(aggregate_uncertainty(ef), run_physics_gate(w))
    assert decision.gate.hard_valid == 1
    assert decision.verdict is Verdict.TRUST
    assert decision.Q > 0.7


def test_load_backend_mock_ok():
    b = load_backend("mock")
    assert b.name == "mock" and b.license == "Apache-2.0"


def test_license_refuses_asl_by_default():
    with pytest.raises(LicenseError):
        load_backend("mace-omat-0")  # ASL non-commercial -> refused fail-closed


def test_license_unknown_backend_refused():
    with pytest.raises(LicenseError):
        load_backend("totally-unknown-net")


def test_license_explicit_optin_passes_check_then_lacks_loader():
    # Opting ASL in clears the *license* gate; mace-omat-0 then has no loader,
    # so it still raises (we ship no ASL loader) — but with a different reason.
    with pytest.raises(LicenseError, match="no loader"):
        load_backend("mace-omat-0", allow_extra=frozenset({"ASL"}))
