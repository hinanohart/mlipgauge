"""Evaluate a position trajectory with one or more backends to build the IR
objects the decision core consumes (TrajectoryWindow, EnsembleForces)."""

from __future__ import annotations

import numpy as np

from mlipgauge.backends.base import Backend
from mlipgauge.types import EnsembleForces, TrajectoryWindow


def evaluate_window(
    backend: Backend,
    positions_traj: np.ndarray,  # (F, N, 3) Å
    atomic_numbers: np.ndarray,  # (N,)
    *,
    cell_traj: np.ndarray | None = None,  # (F, 3, 3) Å
    kinetic_energy: np.ndarray | None = None,
    timestep_fs: float = 1.0,
) -> TrajectoryWindow:
    pos = np.asarray(positions_traj, dtype=np.float64)
    if pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError(f"positions_traj must be (F,N,3), got {pos.shape}")
    n_frames = pos.shape[0]
    energies, forces, stresses = [], [], []
    for t in range(n_frames):
        cell = None if cell_traj is None else cell_traj[t]
        pred = backend.predict(pos[t], atomic_numbers, cell)
        energies.append(pred.energy)
        forces.append(pred.forces)
        stresses.append(pred.stress)
    stress = None if stresses[0] is None else np.stack(stresses)
    return TrajectoryWindow(
        positions=pos,
        forces=np.stack(forces),
        potential_energy=np.asarray(energies, dtype=np.float64),
        atomic_numbers=atomic_numbers,
        timestep_fs=timestep_fs,
        cell=cell_traj,
        stress=stress,
        kinetic_energy=kinetic_energy,
    )


def evaluate_ensemble_forces(
    backends: list[Backend],
    positions_traj: np.ndarray,
    atomic_numbers: np.ndarray,
    *,
    cell_traj: np.ndarray | None = None,
) -> EnsembleForces:
    if len(backends) < 2:
        raise ValueError("ensemble UQ needs at least 2 backends")
    pos = np.asarray(positions_traj, dtype=np.float64)
    n_frames = pos.shape[0]
    per_model = []
    for b in backends:
        frames = []
        for t in range(n_frames):
            cell = None if cell_traj is None else cell_traj[t]
            frames.append(b.predict(pos[t], atomic_numbers, cell).forces)
        per_model.append(np.stack(frames))
    return EnsembleForces(np.stack(per_model))  # (K, F, N, 3)
