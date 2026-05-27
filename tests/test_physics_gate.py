"""S2/S4: the physics gate (the CLAIM). Each hard check is exercised with a
physically-consistent window (must pass) and a deliberately-violating one (must
fail), so the gate is shown to be falsifiable rather than always-on."""

from __future__ import annotations

import numpy as np

from mlipgauge.core.physics_gate import PhysicsGateConfig, run_physics_gate
from mlipgauge.types import TrajectoryWindow


def _consistent_window(n_frames=3, force_per_atom=None, dx_pattern=None):
    """Build a window whose energies are exactly −∮F·dx consistent (conservative,
    constant force field), so energy_force_consistency must pass with residual 0."""
    n = 2
    if force_per_atom is None:
        force_per_atom = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    if dx_pattern is None:
        dx_pattern = np.array([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])
    forces = np.broadcast_to(force_per_atom, (n_frames, n, 3)).copy()
    positions = np.stack([t * dx_pattern for t in range(n_frames)])
    # ΔE per step = −Σ F·Δx (Δx = dx_pattern, constant)
    de = -float(np.sum(force_per_atom * dx_pattern))
    energies = np.array([t * de for t in range(n_frames)])
    return TrajectoryWindow(
        positions=positions,
        forces=forces,
        potential_energy=energies,
        atomic_numbers=np.array([1, 8]),
        timestep_fs=1.0,
    ), de


def test_energy_force_consistency_passes_when_conservative():
    w, _ = _consistent_window()
    r = run_physics_gate(w)
    assert r.hard_checks["energy_force_consistency"] is True
    assert r.hard_valid == 1


def test_energy_force_consistency_fails_on_discontinuity():
    w, _ = _consistent_window()
    bad_e = w.potential_energy.copy()
    bad_e[-1] += 0.5  # inject a 0.5 eV jump inconsistent with the forces
    w_bad = TrajectoryWindow(
        positions=w.positions,
        forces=w.forces,
        potential_energy=bad_e,
        atomic_numbers=w.atomic_numbers,
        timestep_fs=1.0,
    )
    r = run_physics_gate(w_bad)
    assert r.hard_checks["energy_force_consistency"] is False
    assert r.hard_valid == 0
    assert any("energy" in s for s in r.reasons)


def test_nve_conservation_pass_and_fail():
    w, _ = _consistent_window()
    # make total energy (pot+kin) constant -> kin = const - pot
    kin_ok = 10.0 - w.potential_energy
    w_ok = TrajectoryWindow(
        positions=w.positions,
        forces=w.forces,
        potential_energy=w.potential_energy,
        atomic_numbers=w.atomic_numbers,
        timestep_fs=1.0,
        kinetic_energy=kin_ok,
    )
    assert run_physics_gate(w_ok).hard_checks["nve_energy_conservation"] is True

    kin_bad = kin_ok.copy()
    kin_bad[-1] += 1.0  # 1 eV / 2 atoms = 0.5 eV/atom drift > tol
    w_bad = TrajectoryWindow(
        positions=w.positions,
        forces=w.forces,
        potential_energy=w.potential_energy,
        atomic_numbers=w.atomic_numbers,
        timestep_fs=1.0,
        kinetic_energy=kin_bad,
    )
    r = run_physics_gate(w_bad)
    assert r.hard_checks["nve_energy_conservation"] is False
    assert r.hard_valid == 0


def test_stress_symmetry_pass_and_fail():
    w, _ = _consistent_window()
    sym = np.broadcast_to(np.eye(3) * 0.3, (w.n_frames, 3, 3)).copy()
    w_ok = TrajectoryWindow(
        positions=w.positions,
        forces=w.forces,
        potential_energy=w.potential_energy,
        atomic_numbers=w.atomic_numbers,
        timestep_fs=1.0,
        stress=sym,
    )
    assert run_physics_gate(w_ok).hard_checks["stress_symmetry"] is True

    asym = sym.copy()
    asym[:, 0, 1] = 0.2
    asym[:, 1, 0] = -0.2  # strongly non-symmetric
    w_bad = TrajectoryWindow(
        positions=w.positions,
        forces=w.forces,
        potential_energy=w.potential_energy,
        atomic_numbers=w.atomic_numbers,
        timestep_fs=1.0,
        stress=asym,
    )
    assert run_physics_gate(w_bad).hard_checks["stress_symmetry"] is False


def test_near_zero_stress_is_skipped_not_guessed():
    """A ~zero stress tensor carries no symmetry information; the check is
    skipped rather than spuriously passed (all-zeros) or failed (tiny noise)."""
    w, _ = _consistent_window()
    tiny = np.zeros((w.n_frames, 3, 3))
    tiny[:, 0, 1] = 1e-12
    tiny[:, 1, 0] = -1e-12  # antisymmetric, but far below stress_min_norm
    w_tiny = TrajectoryWindow(
        positions=w.positions,
        forces=w.forces,
        potential_energy=w.potential_energy,
        atomic_numbers=w.atomic_numbers,
        timestep_fs=1.0,
        stress=tiny,
    )
    r = run_physics_gate(w_tiny)
    assert "stress_symmetry" in r.skipped
    assert "stress_symmetry" not in r.hard_checks


def _optical_mode(n_atoms):
    """A unit eigenvector orthogonal to rigid translation (atoms 0 and 1 move
    oppositely along x): a genuine optical mode, not a translational artefact."""
    u = np.zeros(3 * n_atoms)
    u[0] = 1.0
    u[3] = -1.0
    return u / np.linalg.norm(u)


def test_imaginary_phonon_pass_and_fail():
    w, _ = _consistent_window()
    n = w.n_atoms
    masses = np.array([1.0, 1.0])
    # positive-definite Hessian -> no imaginary modes
    h_pd = np.eye(3 * n) * 2.0
    assert run_physics_gate(w, hessian=h_pd, masses=masses).hard_checks["imaginary_phonon"] is True
    # a genuine soft optical mode (eigenvalue −1 along a translation-orthogonal
    # eigenvector) survives acoustic projection and must be flagged
    u = _optical_mode(n)
    h_neg = np.eye(3 * n) * 2.0 + (-1.0 - 2.0) * np.outer(u, u)
    r = run_physics_gate(w, hessian=h_neg, masses=masses)
    assert r.hard_checks["imaginary_phonon"] is False
    assert r.hard_valid == 0
    assert any("imaginary" in s for s in r.reasons)


def test_acoustic_translation_not_flagged_as_imaginary():
    """A stable structure whose finite-difference Hessian has a slightly-negative
    rigid translational mode must pass once acoustic modes are projected out, and
    be (correctly) flagged when projection is disabled — proving it is the
    projection, not a loosened tolerance, that removes the artefact."""
    w, _ = _consistent_window()
    n = w.n_atoms
    masses = np.array([1.0, 1.0])
    t = np.zeros(3 * n)
    t[0] = 1.0
    t[3] = 1.0  # rigid x-translation (mass-weighted; equal masses)
    t /= np.linalg.norm(t)
    # translational eigenvalue = −1e-2, well below −phonon_neg_tol (1e-4)
    h = np.eye(3 * n) * 2.0 + (-1.0e-2 - 2.0) * np.outer(t, t)
    raw = run_physics_gate(w, PhysicsGateConfig(project_acoustic=False), hessian=h, masses=masses)
    assert raw.hard_checks["imaginary_phonon"] is False  # artefact flagged w/o projection
    proj = run_physics_gate(w, PhysicsGateConfig(project_acoustic=True), hessian=h, masses=masses)
    assert proj.hard_checks["imaginary_phonon"] is True  # artefact removed by projection


def test_multiplicative_one_violation_zeroes_hard_valid():
    w, _ = _consistent_window()
    sym = np.broadcast_to(np.eye(3), (w.n_frames, 3, 3)).copy()
    sym[:, 0, 1] = 0.5
    sym[:, 1, 0] = -0.5  # only stress is broken; energy-force still consistent
    w_mixed = TrajectoryWindow(
        positions=w.positions,
        forces=w.forces,
        potential_energy=w.potential_energy,
        atomic_numbers=w.atomic_numbers,
        timestep_fs=1.0,
        stress=sym,
    )
    r = run_physics_gate(w_mixed)
    assert r.hard_checks["energy_force_consistency"] is True
    assert r.hard_checks["stress_symmetry"] is False
    assert r.hard_valid == 0  # product collapses


def test_absent_inputs_are_skipped_not_guessed():
    # single frame, no kinetic/stress/hessian -> every check skipped
    w = TrajectoryWindow(
        positions=np.zeros((1, 2, 3)),
        forces=np.zeros((1, 2, 3)),
        potential_energy=np.zeros(1),
        atomic_numbers=np.array([1, 1]),
        timestep_fs=1.0,
    )
    r = run_physics_gate(w)
    assert set(r.skipped) == {
        "energy_force_consistency",
        "nve_energy_conservation",
        "imaginary_phonon",
        "stress_symmetry",
    }
    assert r.hard_checks == {}
    assert r.hard_valid == 1  # vacuously valid; gauge layer treats this as ABSTAIN


def test_nonfinite_inputs_fail_closed():
    bad_f = np.zeros((3, 2, 3))
    bad_f[0, 0, 0] = np.nan
    w = TrajectoryWindow(
        positions=np.zeros((3, 2, 3)),
        forces=bad_f,
        potential_energy=np.zeros(3),
        atomic_numbers=np.array([1, 1]),
        timestep_fs=1.0,
    )
    r = run_physics_gate(w)
    assert r.hard_valid == 0
    assert r.hard_checks == {"finite_inputs": False}


def test_tolerance_is_respected_at_boundary():
    w, _ = _consistent_window()
    # inject a residual just above per-atom tol*N so it fails, just below so it passes
    cfg = PhysicsGateConfig(energy_force_tol_ev_per_atom=0.01)
    n = w.n_atoms
    e = w.potential_energy.copy()
    e[-1] += 0.01 * n * 0.9  # 90% of tolerance budget -> pass
    w_pass = TrajectoryWindow(
        positions=w.positions,
        forces=w.forces,
        potential_energy=e,
        atomic_numbers=w.atomic_numbers,
        timestep_fs=1.0,
    )
    assert run_physics_gate(w_pass, cfg).hard_checks["energy_force_consistency"] is True
    e2 = w.potential_energy.copy()
    e2[-1] += 0.01 * n * 1.1  # 110% -> fail
    w_fail = TrajectoryWindow(
        positions=w.positions,
        forces=w.forces,
        potential_energy=e2,
        atomic_numbers=w.atomic_numbers,
        timestep_fs=1.0,
    )
    assert run_physics_gate(w_fail, cfg).hard_checks["energy_force_consistency"] is False
