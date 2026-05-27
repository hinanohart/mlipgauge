"""Deterministic, dependency-free backend used for tests and CI (GPU-free).

It is an analytic harmonic potential E(x) = ½k·Σ‖x_i − x0‖² with the exact
gradient force F_i = −k(x_i − x0) and a symmetric virial stress. Because energy
and forces come from the same closed form, a trajectory evaluated with this
backend is conservative by construction and passes the physics gate — which is
exactly what we want for a clean end-to-end pipeline test. Varying ``k`` across
instances yields a controllable ensemble for uncertainty tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mlipgauge.backends.base import Prediction


@dataclass
class MockBackend:
    k: float = 1.0
    reference: np.ndarray | None = None  # (N,3) equilibrium positions; default 0
    name: str = field(default="mock", init=False)
    license: str = field(default="Apache-2.0", init=False)

    def predict(
        self,
        positions: np.ndarray,
        atomic_numbers: np.ndarray,
        cell: np.ndarray | None = None,
    ) -> Prediction:
        x = np.asarray(positions, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError(f"positions must be (N,3), got {x.shape}")
        ref = np.zeros_like(x) if self.reference is None else np.asarray(self.reference, float)
        disp = x - ref
        energy = float(0.5 * self.k * np.sum(disp * disp))
        forces = -self.k * disp  # F = −dE/dx
        virial = -(disp[:, :, None] * forces[:, None, :]).sum(axis=0)  # (3,3)
        virial = 0.5 * (virial + virial.T)  # symmetric by construction
        if cell is not None:
            vol = abs(float(np.linalg.det(np.asarray(cell, dtype=np.float64))))
            vol = vol if vol > 1e-12 else 1.0
        else:
            vol = 1.0
        stress = virial / vol
        return Prediction(energy=energy, forces=forces, stress=stress)
