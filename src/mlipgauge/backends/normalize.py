"""Unit and sign normalization to canonical mlipgauge units.

Canonical units: energy eV, force eV/Å, stress eV/Å³, with the physics
convention force = −dE/dx and a symmetric stress tensor. Different MLIP codes
report different units/sign conventions; getting this wrong silently corrupts
every downstream physics check, so normalization is explicit and unit-tested.
"""

from __future__ import annotations

import numpy as np

from mlipgauge.backends.base import Prediction

# energy unit -> eV
_ENERGY_TO_EV: dict[str, float] = {
    "eV": 1.0,
    "Hartree": 27.211386245988,
    "Ry": 13.605693122994,
    "kcal/mol": 0.04336410390059,
    "kJ/mol": 0.01036410,
}
# length unit -> Å
_LENGTH_TO_ANG: dict[str, float] = {
    "Angstrom": 1.0,
    "Ang": 1.0,
    "Bohr": 0.529177210903,
    "nm": 10.0,
}
# stress unit -> eV/Å³
_STRESS_TO_EV_A3: dict[str, float] = {
    "eV/Ang^3": 1.0,
    "eV/A^3": 1.0,
    "GPa": 0.006241509074460763,
    "kbar": 0.0006241509074460763,
}


def energy_to_ev(value: float, unit: str = "eV") -> float:
    if unit not in _ENERGY_TO_EV:
        raise ValueError(f"unknown energy unit {unit!r}; known {sorted(_ENERGY_TO_EV)}")
    return float(value) * _ENERGY_TO_EV[unit]


def force_to_ev_per_ang(
    forces: np.ndarray,
    energy_unit: str = "eV",
    length_unit: str = "Angstrom",
    *,
    is_gradient: bool = False,
) -> np.ndarray:
    """Convert forces to eV/Å. If ``is_gradient`` the input is +dE/dx and is
    negated to obtain the physical force −dE/dx."""
    if energy_unit not in _ENERGY_TO_EV:
        raise ValueError(f"unknown energy unit {energy_unit!r}")
    if length_unit not in _LENGTH_TO_ANG:
        raise ValueError(f"unknown length unit {length_unit!r}")
    factor = _ENERGY_TO_EV[energy_unit] / _LENGTH_TO_ANG[length_unit]
    f = np.asarray(forces, dtype=np.float64) * factor
    return -f if is_gradient else f


def stress_to_ev_per_ang3(
    stress: np.ndarray,
    unit: str = "eV/Ang^3",
    *,
    sign: float = 1.0,
) -> np.ndarray:
    if unit not in _STRESS_TO_EV_A3:
        raise ValueError(f"unknown stress unit {unit!r}; known {sorted(_STRESS_TO_EV_A3)}")
    return np.asarray(stress, dtype=np.float64) * (_STRESS_TO_EV_A3[unit] * sign)


def normalize_prediction(
    energy: float,
    forces: np.ndarray,
    stress: np.ndarray | None = None,
    *,
    energy_unit: str = "eV",
    length_unit: str = "Angstrom",
    force_is_gradient: bool = False,
    stress_unit: str = "eV/Ang^3",
    stress_sign: float = 1.0,
) -> Prediction:
    """Build a canonical-units :class:`Prediction` from a backend's raw output."""
    e = energy_to_ev(energy, energy_unit)
    f = force_to_ev_per_ang(forces, energy_unit, length_unit, is_gradient=force_is_gradient)
    s = None if stress is None else stress_to_ev_per_ang3(stress, stress_unit, sign=stress_sign)
    return Prediction(energy=e, forces=f, stress=s)
