"""Intermediate representation (IR) for mlipgauge.

These dataclasses are the normative contract between the data-producing layer
(backends / MD driver) and the decision core (uq, calibration, physics_gate,
gauge). Every core function consumes and returns these types 1:1.

Design rules:
- Frozen dataclasses (decisions are immutable records).
- Validation happens at construction (`__post_init__`): wrong shapes or
  non-finite physics inputs raise immediately, so a malformed window can never
  silently produce a "trust" verdict. (Fail-closed begins at the type boundary.)
- Units are fixed and explicit: positions/cell in Å, forces in eV/Å, energies
  in eV, stress in eV/Å³, time in fs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np


class Verdict(StrEnum):
    """Outcome of gauging one trajectory window."""

    TRUST = "trust"  # Q high, no hard violation, low uncertainty
    FLAG = "flag"  # no hard violation but elevated (uncalibrated) uncertainty
    HALT = "halt"  # a hard physics gate fired -> Q collapses to 0 (fail-closed)
    ABSTAIN = "abstain"  # inputs unusable / UQ uncomputable -> treated fail-closed


def _as_float_array(x, name: str, ndim: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != ndim:
        raise ValueError(f"{name}: expected {ndim}D array, got shape {arr.shape}")
    return arr


@dataclass(frozen=True)
class TrajectoryWindow:
    """A contiguous window of ``F`` MD frames for a system of ``N`` atoms.

    ``stress``, ``cell`` and ``kinetic_energy`` are optional: gates that need
    them are skipped (and reported as skipped) when absent, rather than guessed.
    """

    positions: np.ndarray  # (F, N, 3) Å
    forces: np.ndarray  # (F, N, 3) eV/Å
    potential_energy: np.ndarray  # (F,) eV
    atomic_numbers: np.ndarray  # (N,) int
    timestep_fs: float
    cell: np.ndarray | None = None  # (F, 3, 3) Å
    stress: np.ndarray | None = None  # (F, 3, 3) eV/Å³
    kinetic_energy: np.ndarray | None = None  # (F,) eV

    def __post_init__(self) -> None:
        pos = _as_float_array(self.positions, "positions", 3)
        frc = _as_float_array(self.forces, "forces", 3)
        pe = _as_float_array(self.potential_energy, "potential_energy", 1)
        z = np.asarray(self.atomic_numbers, dtype=np.int64)
        object.__setattr__(self, "positions", pos)
        object.__setattr__(self, "forces", frc)
        object.__setattr__(self, "potential_energy", pe)
        object.__setattr__(self, "atomic_numbers", z)

        f, n = pos.shape[0], pos.shape[1]
        if f < 1:
            raise ValueError("window must have at least 1 frame")
        if pos.shape != (f, n, 3):
            raise ValueError(f"positions must be (F,N,3), got {pos.shape}")
        if frc.shape != (f, n, 3):
            raise ValueError(f"forces must match positions (F,N,3), got {frc.shape}")
        if pe.shape != (f,):
            raise ValueError(f"potential_energy must be (F,), got {pe.shape}")
        if z.shape != (n,):
            raise ValueError(f"atomic_numbers must be (N,), got {z.shape}")
        if not np.isfinite(self.timestep_fs) or self.timestep_fs <= 0:
            raise ValueError("timestep_fs must be a positive finite number")

        if self.cell is not None:
            cell = _as_float_array(self.cell, "cell", 3)
            if cell.shape != (f, 3, 3):
                raise ValueError(f"cell must be (F,3,3), got {cell.shape}")
            object.__setattr__(self, "cell", cell)
        if self.stress is not None:
            st = _as_float_array(self.stress, "stress", 3)
            if st.shape != (f, 3, 3):
                raise ValueError(f"stress must be (F,3,3), got {st.shape}")
            object.__setattr__(self, "stress", st)
        if self.kinetic_energy is not None:
            ke = _as_float_array(self.kinetic_energy, "kinetic_energy", 1)
            if ke.shape != (f,):
                raise ValueError(f"kinetic_energy must be (F,), got {ke.shape}")
            object.__setattr__(self, "kinetic_energy", ke)

    @property
    def n_frames(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self.positions.shape[1])

    @property
    def finite(self) -> bool:
        """True iff all *required* physics inputs are finite (fail-closed probe)."""
        ok = bool(
            np.isfinite(self.positions).all()
            and np.isfinite(self.forces).all()
            and np.isfinite(self.potential_energy).all()
        )
        return ok


@dataclass(frozen=True)
class EnsembleForces:
    """Force predictions from ``K`` independent models over a window's frames.

    This is the sole uncertainty source in 0.1.0a2: cross-model force
    disagreement. ``forces`` has shape (K, F, N, 3) in eV/Å with K >= 2.
    """

    forces: np.ndarray

    def __post_init__(self) -> None:
        arr = _as_float_array(self.forces, "ensemble forces", 4)
        if arr.shape[0] < 2:
            raise ValueError("ensemble UQ needs at least 2 models (K>=2)")
        object.__setattr__(self, "forces", arr)

    @property
    def n_models(self) -> int:
        return int(self.forces.shape[0])


@dataclass(frozen=True)
class UQResult:
    """Aggregated uncertainty for a window."""

    u_raw: float  # raw scale (eV/Å): mean over frames of per-atom force std
    u: float  # mapped to [0,1] (calibrated if a calibrator was applied)
    per_frame: np.ndarray  # (F,) raw per-frame uncertainty
    method: str
    calibrated: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.u <= 1.0):
            raise ValueError(f"u must be in [0,1], got {self.u}")


@dataclass(frozen=True)
class GateResult:
    """Result of the physics-validity gate over a window."""

    hard_valid: int  # ∏ hard checks ∈ {0,1}
    hard_checks: dict[str, bool]  # name -> passed (True=ok)
    soft_scores: dict[str, float]  # name -> [0,1], 1=healthy
    skipped: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.hard_valid not in (0, 1):
            raise ValueError("hard_valid must be 0 or 1")
        for k, v in self.soft_scores.items():
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"soft score {k} out of [0,1]: {v}")


@dataclass(frozen=True)
class GaugeDecision:
    """Final per-window decision: Q = hard_valid * calib(u-derived health)."""

    Q: float  # gauge value in [0,1]; 0 == hard fail-closed
    verdict: Verdict
    uq: UQResult
    gate: GateResult
    active_learning: bool  # push offending config to AL queue?

    def __post_init__(self) -> None:
        if not (0.0 <= self.Q <= 1.0):
            raise ValueError(f"Q must be in [0,1], got {self.Q}")
