"""Measure the physics gate's detection behaviour on ground-truth-labelled
trajectories (the CLAIM).

Honesty: the trajectories here are *constructed* (a conservative harmonic mock
backend for the valid class; deliberately injected violations for the positive
class), not real MLIP-MD runs on real materials. That is a synthetic but
physically-grounded measurement of whether the gate fires on known violations
and stays quiet on known-valid windows. Real-backend / real-DFT validation is
deferred. mode = "synthetic" is recorded accordingly.

Metrics (each with an n-trial bootstrap CI over Bernoulli outcomes):
  - sensitivity[type]  : P(check fires | that violation injected)   -> want 1.0
  - specificity        : P(hard_valid==1 | clean window)            -> want 1.0
  - threshold sweep    : detection vs violation magnitude (graded, not always-on)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from _measure_common import make_meta  # noqa: E402

from mlipgauge import __version__  # noqa: E402
from mlipgauge.backends.mock import MockBackend  # noqa: E402
from mlipgauge.backends.runner import evaluate_window  # noqa: E402
from mlipgauge.core.calibration import bootstrap_ci  # noqa: E402
from mlipgauge.core.physics_gate import PhysicsGateConfig, run_physics_gate  # noqa: E402
from mlipgauge.types import TrajectoryWindow  # noqa: E402


def _clean_window(rng, n=4, n_frames=4, k=1.5):
    base = rng.standard_normal((n, 3)) * 0.2
    step = rng.standard_normal((n, 3)) * 1e-3
    traj = np.stack([base + t * step for t in range(n_frames)])
    z = np.array([1, 6, 8, 7])[:n]
    return evaluate_window(MockBackend(k=k), traj, z)


def _rate(outcomes: list[bool], seed: int) -> dict:
    arr = np.array([1.0 if o else 0.0 for o in outcomes])
    lo, hi = bootstrap_ci(arr, seed=seed) if arr.size else (None, None)
    return {"rate": float(arr.mean()), "n": int(arr.size), "ci95": [lo, hi]}


def measure(n: int = 200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    cfg = PhysicsGateConfig()

    specificity_outcomes = []
    sens = {
        k: []
        for k in (
            "energy_force_consistency",
            "nve_energy_conservation",
            "imaginary_phonon",
            "stress_symmetry",
        )
    }

    for _ in range(n):
        clean = _clean_window(rng)
        nat = clean.n_atoms

        # clean specificity: provide all optional inputs in a valid form
        kin_ok = 100.0 - clean.potential_energy
        sym_stress = np.broadcast_to(np.eye(3) * 0.1, (clean.n_frames, 3, 3)).copy()
        h_pd = np.eye(3 * nat) * 2.0
        w_clean = TrajectoryWindow(
            positions=clean.positions,
            forces=clean.forces,
            potential_energy=clean.potential_energy,
            atomic_numbers=clean.atomic_numbers,
            timestep_fs=1.0,
            kinetic_energy=kin_ok,
            stress=sym_stress,
        )
        specificity_outcomes.append(
            run_physics_gate(w_clean, cfg, hessian=h_pd, masses=np.ones(nat)).hard_valid == 1
        )

        # energy-force violation
        e_bad = clean.potential_energy.copy()
        e_bad[-1] += cfg.energy_force_tol_ev_per_atom * nat * 5.0
        w = TrajectoryWindow(
            positions=clean.positions,
            forces=clean.forces,
            potential_energy=e_bad,
            atomic_numbers=clean.atomic_numbers,
            timestep_fs=1.0,
        )
        sens["energy_force_consistency"].append(
            run_physics_gate(w, cfg).hard_checks["energy_force_consistency"] is False
        )

        # NVE drift violation
        kin_bad = kin_ok.copy()
        kin_bad[-1] += cfg.nve_drift_tol_ev_per_atom * nat * 5.0
        w = TrajectoryWindow(
            positions=clean.positions,
            forces=clean.forces,
            potential_energy=clean.potential_energy,
            atomic_numbers=clean.atomic_numbers,
            timestep_fs=1.0,
            kinetic_energy=kin_bad,
        )
        sens["nve_energy_conservation"].append(
            run_physics_gate(w, cfg).hard_checks["nve_energy_conservation"] is False
        )

        # imaginary phonon: a genuine soft optical mode (eigenvector orthogonal
        # to rigid translation, so it survives acoustic-mode projection)
        u = np.zeros(3 * nat)
        u[0], u[3] = 1.0, -1.0
        u /= np.linalg.norm(u)
        h_neg = np.eye(3 * nat) * 2.0 + (-1.0 - 2.0) * np.outer(u, u)
        sens["imaginary_phonon"].append(
            run_physics_gate(clean, cfg, hessian=h_neg, masses=np.ones(nat)).hard_checks[
                "imaginary_phonon"
            ]
            is False
        )

        # stress asymmetry
        asym = sym_stress.copy()
        asym[:, 0, 1] = 0.5
        asym[:, 1, 0] = -0.5
        w = TrajectoryWindow(
            positions=clean.positions,
            forces=clean.forces,
            potential_energy=clean.potential_energy,
            atomic_numbers=clean.atomic_numbers,
            timestep_fs=1.0,
            stress=asym,
        )
        sens["stress_symmetry"].append(
            run_physics_gate(w, cfg).hard_checks["stress_symmetry"] is False
        )

    # threshold sweep on energy-force: detection vs multiple of tolerance
    sweep = {}
    for mult in (0.5, 0.9, 1.1, 2.0, 5.0):
        outs = []
        for _ in range(n):
            clean = _clean_window(rng)
            e = clean.potential_energy.copy()
            e[-1] += cfg.energy_force_tol_ev_per_atom * clean.n_atoms * mult
            w = TrajectoryWindow(
                positions=clean.positions,
                forces=clean.forces,
                potential_energy=e,
                atomic_numbers=clean.atomic_numbers,
                timestep_fs=1.0,
            )
            outs_fired = run_physics_gate(w, cfg).hard_checks["energy_force_consistency"] is False
            outs.append(outs_fired)
        sweep[f"{mult}x_tol"] = float(np.mean(outs))

    return {
        "meta": make_meta(n=n, mode="synthetic", seed=seed),
        "claim": "physics gate detects injected hard violations and passes valid windows",
        "disclaimer": "constructed trajectories (harmonic mock + injected violations), "
        "not real MLIP-MD on real materials; live validation deferred",
        "sensitivity": {k: _rate(v, seed) for k, v in sens.items()},
        "specificity_clean": _rate(specificity_outcomes, seed),
        "energy_force_threshold_sweep": sweep,
    }


def main() -> int:
    out = measure()
    path = Path(__file__).resolve().parent.parent / "results" / f"{__version__}_gate.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
    print(
        json.dumps(
            {
                k: out[k]
                for k in ("sensitivity", "specificity_clean", "energy_force_threshold_sweep")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
