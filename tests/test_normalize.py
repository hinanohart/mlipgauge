"""S3: unit/sign normalization. Wrong conversions silently corrupt the physics
checks, so each conversion has a golden value."""

from __future__ import annotations

import numpy as np
import pytest

from mlipgauge.backends.base import Prediction
from mlipgauge.backends.normalize import (
    energy_to_ev,
    force_to_ev_per_ang,
    normalize_prediction,
    stress_to_ev_per_ang3,
)


def test_energy_units_golden():
    assert energy_to_ev(1.0, "eV") == 1.0
    assert energy_to_ev(1.0, "Hartree") == pytest.approx(27.211386245988)
    assert energy_to_ev(2.0, "Ry") == pytest.approx(27.211386245988, rel=1e-6)


def test_force_gradient_sign_flip():
    g = np.array([[1.0, 0.0, 0.0]])  # +dE/dx
    f = force_to_ev_per_ang(g, is_gradient=True)
    np.testing.assert_allclose(f, [[-1.0, 0.0, 0.0]])  # physical force = −gradient
    f2 = force_to_ev_per_ang(g, is_gradient=False)
    np.testing.assert_allclose(f2, g)


def test_force_unit_conversion_hartree_per_bohr():
    f = force_to_ev_per_ang(np.array([[1.0, 0, 0]]), "Hartree", "Bohr")
    expected = 27.211386245988 / 0.529177210903
    np.testing.assert_allclose(f, [[expected, 0, 0]])


def test_stress_gpa_golden():
    s = stress_to_ev_per_ang3(np.eye(3), "GPa")
    np.testing.assert_allclose(np.diag(s), [0.006241509074460763] * 3)


def test_unknown_units_raise():
    with pytest.raises(ValueError):
        energy_to_ev(1.0, "joule")
    with pytest.raises(ValueError):
        stress_to_ev_per_ang3(np.eye(3), "psi")


def test_normalize_prediction_roundtrip():
    pred = normalize_prediction(
        energy=1.0,
        forces=np.array([[1.0, 0, 0]]),
        stress=np.eye(3),
        energy_unit="Hartree",
        stress_unit="GPa",
    )
    assert isinstance(pred, Prediction)
    assert pred.energy == pytest.approx(27.211386245988)


def test_prediction_rejects_bad_shapes():
    with pytest.raises(ValueError):
        Prediction(energy=0.0, forces=np.zeros((3,)))
    with pytest.raises(ValueError):
        Prediction(energy=0.0, forces=np.zeros((2, 3)), stress=np.zeros((2, 2)))
