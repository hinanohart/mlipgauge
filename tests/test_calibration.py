"""S2: calibration core. Golden values are hand-computed; properties are
cross-checks independent of the implementation path."""

from __future__ import annotations

import numpy as np
import pytest

from mlipgauge.core.calibration import (
    IsotonicCalibrator,
    _pava,
    bootstrap_ci,
    expected_calibration_error,
)


def test_pava_golden_pool_two():
    # [1,0,1] -> pool first two to 0.5 -> [0.5, 0.5, 1.0]
    out = _pava(np.array([1.0, 0.0, 1.0]), np.ones(3))
    np.testing.assert_allclose(out, [0.5, 0.5, 1.0])


def test_pava_golden_anti_monotone():
    # fully decreasing -> single block = overall mean
    out = _pava(np.array([3.0, 2.0, 1.0]), np.ones(3))
    np.testing.assert_allclose(out, [2.0, 2.0, 2.0])


def test_pava_already_monotone_is_identity():
    y = np.array([0.0, 0.1, 0.4, 0.9])
    np.testing.assert_allclose(_pava(y, np.ones(4)), y)


def test_pava_invariants_monotone_and_sum_preserving():
    rng = np.random.default_rng(3)
    for _ in range(50):
        y = rng.standard_normal(20)
        out = _pava(y, np.ones(20))
        assert np.all(np.diff(out) >= -1e-12)  # non-decreasing
        assert abs(out.sum() - y.sum()) < 1e-9  # PAVA preserves total


def test_isotonic_predict_monotone_and_bounded():
    rng = np.random.default_rng(4)
    scores = rng.uniform(0, 1, 200)
    labels = (rng.uniform(0, 1, 200) < scores).astype(float)  # roughly calibrated
    cal = IsotonicCalibrator.fit(scores, labels, synthetic=True)
    grid = np.linspace(0, 1, 50)
    p = cal.predict(grid)
    assert np.all((p >= 0) & (p <= 1))
    assert np.all(np.diff(p) >= -1e-12)
    assert cal.synthetic is True


def test_isotonic_rejects_non_binary_labels():
    with pytest.raises(ValueError):
        IsotonicCalibrator.fit(np.array([0.1, 0.2]), np.array([0.5, 1.0]))


def test_ece_golden_perfect_and_worst():
    # all conf 0.0 but all correct -> |0-1| = 1.0
    assert expected_calibration_error(np.zeros(4), np.ones(4), n_bins=10) == pytest.approx(1.0)
    # conf 0.5 with half-and-half labels in that bin -> 0
    probs = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.0)


def test_bootstrap_ci_constant_is_point():
    lo, hi = bootstrap_ci(np.full(50, 5.0), seed=1)
    assert lo == pytest.approx(5.0) and hi == pytest.approx(5.0)


def test_bootstrap_ci_deterministic_with_seed():
    rng = np.random.default_rng(7)
    data = rng.standard_normal(100)
    a = bootstrap_ci(data, seed=42)
    b = bootstrap_ci(data, seed=42)
    assert a == b
