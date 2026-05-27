"""Physics-validity gate — the intellectual core of mlipgauge (the CLAIM).

A universal MLIP can be confidently *wrong*: it can emit forces that are not the
gradient of its energy (non-conservative / discontinuous PES), let the conserved
energy of an NVE run drift, place a structure at a saddle with imaginary phonons,
or return an asymmetric stress tensor. None of these are caught by an uncertainty
estimator — they are violations of physics that hold regardless of how "confident"
the model is. This module turns four such invariants into *hard* checks evaluated
over a molecular-dynamics trajectory *window*, and combines them multiplicatively:

    hard_valid = ∏_k  1[check_k passed]          (any single violation -> 0)

so the gate is fail-closed by construction. What mlipgauge adds over the
underlying physics primitives (which one could compute with ASE/phonopy/numpy)
is the *decision layer*: trajectory-window evaluation, per-atom normalization,
an imaginary-phonon test that projects out the rigid translational (acoustic)
modes, skip-not-guess handling of absent inputs, and the multiplicative
hard/soft separation feeding the runtime gauge.

Hard checks (each: applicable? + passed?):
  1. energy_force_consistency  -- ΔE vs −∮F·dx (catches non-conservative /
                                  discontinuous PES); needs >=2 frames.
  2. nve_energy_conservation   -- drift of total (pot+kin) energy; needs KE.
  3. imaginary_phonon          -- min mass-weighted Hessian eigenvalue < −tol
                                  (ω²<0), after the three rigid translational
                                  (acoustic) modes are projected out; needs a
                                  Hessian + masses.
  4. stress_symmetry           -- ‖σ−σᵀ‖/‖σ‖; needs a stress above a tiny
                                  absolute floor (a ~zero stress is skipped).

Absent inputs => the dependent check is *skipped and recorded*, never guessed.
Residuals/drifts are normalized per atom; this targets system-wide (extensive)
violations and can dilute a single-atom localized discontinuity in a very large
cell — a known limitation of the synthetic-scope ``0.1.0a*`` release (see README).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlipgauge.types import GateResult, TrajectoryWindow

# standard atomic weights (amu) for Z=1..92, covering the periodic table through
# uranium (materials-science systems routinely include heavy transition metals,
# lanthanides and actinides). Index 0 unused. Pass ``masses`` explicitly for any
# Z outside this range or for isotopically-resolved studies.
_ATOMIC_MASSES = np.array(
    [
        0.0,
        1.008,
        4.0026,
        6.94,
        9.0122,
        10.81,
        12.011,
        14.007,
        15.999,
        18.998,
        20.180,
        22.990,
        24.305,
        26.982,
        28.085,
        30.974,
        32.06,
        35.45,
        39.948,
        39.098,
        40.078,
        44.956,
        47.867,
        50.942,
        51.996,
        54.938,
        55.845,
        58.933,
        58.693,
        63.546,
        65.38,
        69.723,
        72.630,
        74.922,
        78.971,
        79.904,
        83.798,
        85.468,
        87.62,
        88.906,
        91.224,
        92.906,
        95.95,
        98.0,
        101.07,
        102.91,
        106.42,
        107.868,
        112.414,
        114.818,
        118.710,
        121.760,
        127.60,
        126.904,
        131.293,
        132.905,
        137.327,
        138.905,
        140.116,
        140.908,
        144.242,
        145.0,
        150.36,
        151.964,
        157.25,
        158.925,
        162.500,
        164.930,
        167.259,
        168.934,
        173.045,
        174.967,
        178.49,
        180.948,
        183.84,
        186.207,
        190.23,
        192.217,
        195.084,
        196.967,
        200.592,
        204.38,
        207.2,
        208.980,
        209.0,
        210.0,
        222.0,
        223.0,
        226.0,
        227.0,
        232.038,
        231.036,
        238.029,
    ]
)


@dataclass(frozen=True)
class PhysicsGateConfig:
    # max |ΔE − (−∮F·dx)| per atom (eV/atom) tolerated between consecutive frames
    energy_force_tol_ev_per_atom: float = 1.0e-2
    # max NVE total-energy drift (max−min over window) per atom (eV/atom)
    nve_drift_tol_ev_per_atom: float = 1.0e-2
    # most-negative mass-weighted Hessian eigenvalue tolerated (eV/Å²/amu)
    phonon_neg_tol: float = 1.0e-4
    # max relative stress asymmetry ‖σ−σᵀ‖_F / (‖σ‖_F+eps)
    stress_asym_tol: float = 1.0e-3
    # below this Frobenius norm a stress tensor is treated as ~zero: its symmetry
    # is un-assessable, so the check is skipped (not silently passed or failed)
    stress_min_norm: float = 1.0e-8
    # project the three rigid translational (acoustic) modes out of the
    # mass-weighted Hessian before the imaginary-phonon test, so a finite-
    # difference acoustic artefact is not mistaken for a real instability
    project_acoustic: bool = True
    # soft-score decay scale: soft = exp(−(violation_size / threshold) * this)
    soft_decay: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "energy_force_tol_ev_per_atom",
            "nve_drift_tol_ev_per_atom",
            "phonon_neg_tol",
            "stress_asym_tol",
            "stress_min_norm",
        ):
            v = getattr(self, name)
            if not (v > 0 and np.isfinite(v)):
                raise ValueError(f"{name} must be positive and finite")


def _masses_for(atomic_numbers: np.ndarray) -> np.ndarray:
    z = np.asarray(atomic_numbers, dtype=np.int64)
    if z.min() < 1 or z.max() >= _ATOMIC_MASSES.size:
        raise ValueError(
            f"atomic numbers must be in [1,{_ATOMIC_MASSES.size - 1}] for the "
            "built-in mass table; pass masses explicitly otherwise"
        )
    return _ATOMIC_MASSES[z]


def _project_out_translations(dyn: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Remove the three rigid-body translational (acoustic) modes from a
    mass-weighted Hessian.

    Translational invariance makes these modes *exactly* zero for an ideal
    Hessian, but a finite-difference one carries small (and possibly negative)
    residuals on them; without this projection a perfectly stable structure can
    be misread as having an imaginary phonon. In mass-weighted coordinates the
    translation along axis ``a`` has component √mᵢ on coordinate ``3i+a``; the
    three such (mutually orthogonal) vectors are normalized into ``V`` and
    removed with ``P = I − VVᵀ``, returning the symmetric ``P·dyn·P``.

    Only translations are projected: for a periodic crystal the rotational modes
    are not zero-modes, and for an isolated cluster they are intentionally left
    in (a true internal soft mode, being orthogonal to rigid translation, is
    unaffected and still detected).
    """
    n = masses.shape[0]
    sqrt_m = np.sqrt(masses)
    v = np.zeros((3 * n, 3), dtype=np.float64)
    for a in range(3):
        v[a::3, a] = sqrt_m  # coordinate 3i+a carries √mᵢ
    v /= np.linalg.norm(v, axis=0, keepdims=True)
    proj = np.eye(3 * n) - v @ v.T
    return proj @ dyn @ proj


def _soft(violation: float, threshold: float, decay: float) -> float:
    """Map a non-negative violation magnitude to a health score in [0,1]
    (1 at zero violation, decaying past the threshold)."""
    return float(np.exp(-(max(violation, 0.0) / threshold) * decay))


def run_physics_gate(
    window: TrajectoryWindow,
    config: PhysicsGateConfig | None = None,
    *,
    hessian: np.ndarray | None = None,
    masses: np.ndarray | None = None,
) -> GateResult:
    """Evaluate the physics-validity gate over ``window``.

    ``hessian`` (3N×3N, eV/Å²) and ``masses`` (N, amu) enable the imaginary-phonon
    check; if ``masses`` is None it is filled from the built-in table. ``hessian``
    is symmetrized before diagonalization.
    """
    cfg = config or PhysicsGateConfig()
    n = window.n_atoms
    hard: dict[str, bool] = {}
    soft: dict[str, float] = {}
    skipped: list[str] = []
    reasons: list[str] = []

    # Fail-closed on non-finite required inputs: the gate cannot certify validity.
    if not window.finite:
        return GateResult(
            hard_valid=0,
            hard_checks={"finite_inputs": False},
            soft_scores={},
            skipped=[],
            reasons=["non-finite positions/forces/energy: cannot certify validity"],
        )

    # 1) energy–force consistency (ΔE vs −∮F·dx, trapezoidal work)
    if window.n_frames >= 2:
        dx = window.positions[1:] - window.positions[:-1]  # (F-1,N,3)
        fbar = 0.5 * (window.forces[1:] + window.forces[:-1])  # (F-1,N,3)
        pred_dE = -np.einsum("fnd,fnd->f", fbar, dx)  # (F-1,)
        actual_dE = window.potential_energy[1:] - window.potential_energy[:-1]
        resid = np.abs(actual_dE - pred_dE)  # (F-1,)
        # worst single-step residual, normalized per atom (not a per-frame mean):
        # a localized discontinuity at any one step is enough to reject the window
        max_resid_per_atom = float(resid.max() / n)
        ok = max_resid_per_atom <= cfg.energy_force_tol_ev_per_atom
        hard["energy_force_consistency"] = ok
        soft["energy_force_smoothness"] = _soft(
            max_resid_per_atom, cfg.energy_force_tol_ev_per_atom, cfg.soft_decay
        )
        if not ok:
            reasons.append(
                f"energy–force inconsistency {max_resid_per_atom:.3e} eV/atom "
                f"> tol {cfg.energy_force_tol_ev_per_atom:.3e} "
                "(non-conservative or discontinuous PES)"
            )
    else:
        skipped.append("energy_force_consistency")

    # 2) NVE total-energy conservation
    if window.kinetic_energy is not None and window.n_frames >= 2:
        total = window.potential_energy + window.kinetic_energy
        drift_per_atom = float((total.max() - total.min()) / n)
        ok = drift_per_atom <= cfg.nve_drift_tol_ev_per_atom
        hard["nve_energy_conservation"] = ok
        soft["nve_conservation"] = _soft(
            drift_per_atom, cfg.nve_drift_tol_ev_per_atom, cfg.soft_decay
        )
        if not ok:
            reasons.append(
                f"NVE energy drift {drift_per_atom:.3e} eV/atom "
                f"> tol {cfg.nve_drift_tol_ev_per_atom:.3e}"
            )
    else:
        skipped.append("nve_energy_conservation")

    # 3) imaginary phonon (mass-weighted Hessian eigenvalues)
    if hessian is not None:
        h = np.asarray(hessian, dtype=np.float64)
        if h.shape != (3 * n, 3 * n):
            raise ValueError(f"hessian must be (3N,3N)=({3 * n},{3 * n}), got {h.shape}")
        m = _masses_for(window.atomic_numbers) if masses is None else np.asarray(masses, float)
        if m.shape != (n,):
            raise ValueError(f"masses must be (N,)=({n},), got {m.shape}")
        inv_sqrt_m = 1.0 / np.sqrt(np.repeat(m, 3))  # (3N,)
        h_sym = 0.5 * (h + h.T)
        dyn = h_sym * inv_sqrt_m[:, None] * inv_sqrt_m[None, :]  # mass-weighted
        if cfg.project_acoustic:
            dyn = _project_out_translations(dyn, m)  # drop rigid acoustic modes
        eig = np.linalg.eigvalsh(dyn)  # ascending, = ω²
        min_eig = float(eig[0])
        ok = min_eig >= -cfg.phonon_neg_tol
        hard["imaginary_phonon"] = ok
        soft["dynamical_stability"] = _soft(max(-min_eig, 0.0), cfg.phonon_neg_tol, cfg.soft_decay)
        if not ok:
            reasons.append(
                f"imaginary phonon mode ω²={min_eig:.3e} < −tol "
                f"{-cfg.phonon_neg_tol:.3e} (dynamically unstable)"
            )
    else:
        skipped.append("imaginary_phonon")

    # 4) stress-tensor symmetry
    if window.stress is not None:
        s = window.stress  # (F,3,3)
        s_norm = np.linalg.norm(s, axis=(1, 2))  # (F,)
        if float(s_norm.max()) < cfg.stress_min_norm:
            # a ~zero stress carries no symmetry information: skip, don't guess
            # (avoids both a vacuous pass on all-zeros and a spurious fail on
            #  antisymmetric numerical noise of negligible magnitude)
            skipped.append("stress_symmetry")
        else:
            asym = np.linalg.norm(s - np.transpose(s, (0, 2, 1)), axis=(1, 2))
            denom = s_norm + 1e-12
            rel = float((asym / denom).max())
            ok = rel <= cfg.stress_asym_tol
            hard["stress_symmetry"] = ok
            soft["stress_symmetry_margin"] = _soft(rel, cfg.stress_asym_tol, cfg.soft_decay)
            if not ok:
                reasons.append(
                    f"stress asymmetry {rel:.3e} > tol {cfg.stress_asym_tol:.3e} "
                    "(σ≠σᵀ: spurious torque)"
                )
    else:
        skipped.append("stress_symmetry")

    hard_valid = 1 if all(hard.values()) else 0  # all() over {} is True (no check fired)
    return GateResult(
        hard_valid=hard_valid,
        hard_checks=hard,
        soft_scores=soft,
        skipped=skipped,
        reasons=reasons,
    )
