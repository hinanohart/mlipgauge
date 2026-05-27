"""Shared ASE-calculator glue for the real backends.

NOTE (honesty): live inference with real MLIP backends is NOT exercised in CI
(no GPU, no weights) and is therefore NOT a tested claim of 0.1.0a1 — it is
deferred/validated in a later release. This code is real (it calls the model
through ASE, not a fake), but treat it as unverified-in-CI until then.

ASE already reports energy in eV, forces in eV/Å (physical, = −dE/dx), and
stress in eV/Å³, so no unit conversion is needed here.
"""

from __future__ import annotations

import numpy as np

from mlipgauge.backends.base import Prediction


def ase_predict(calculator, positions, atomic_numbers, cell=None) -> Prediction:
    try:
        from ase import Atoms
    except ImportError as e:  # pragma: no cover - exercised only without ASE
        raise ImportError("ASE is required for real backends: pip install 'mlipgauge[ase]'") from e
    pbc = cell is not None
    atoms = Atoms(
        numbers=np.asarray(atomic_numbers),
        positions=np.asarray(positions, dtype=np.float64),
        cell=None if cell is None else np.asarray(cell, dtype=np.float64),
        pbc=pbc,
    )
    atoms.calc = calculator
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    stress = None
    if pbc:
        try:
            stress = np.asarray(atoms.get_stress(voigt=False), dtype=np.float64)
        except Exception:  # pragma: no cover - calculator may not provide stress
            stress = None
    return Prediction(energy=energy, forces=forces, stress=stress)
