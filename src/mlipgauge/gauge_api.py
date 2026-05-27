"""High-level entry point: bundle the three sub-configs and run the full
decision for a window, or stream a trajectory through backends into decisions
plus an active-learning queue."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mlipgauge.backends.base import Backend
from mlipgauge.backends.runner import evaluate_ensemble_forces, evaluate_window
from mlipgauge.core.al_trigger import ActiveLearningQueue
from mlipgauge.core.calibration import IsotonicCalibrator
from mlipgauge.core.gauge import DecisionConfig, decide
from mlipgauge.core.physics_gate import PhysicsGateConfig, run_physics_gate
from mlipgauge.core.uq import UQConfig, aggregate_uncertainty
from mlipgauge.types import EnsembleForces, GaugeDecision, TrajectoryWindow


@dataclass(frozen=True)
class GaugeConfig:
    uq: UQConfig = field(default_factory=UQConfig)
    gate: PhysicsGateConfig = field(default_factory=PhysicsGateConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)


def gauge_window(
    window: TrajectoryWindow,
    ensemble: EnsembleForces,
    *,
    config: GaugeConfig | None = None,
    hessian: np.ndarray | None = None,
    masses: np.ndarray | None = None,
    calibrator: IsotonicCalibrator | None = None,
) -> GaugeDecision:
    """Gauge a single trajectory window: calibrated uncertainty × physics gate."""
    cfg = config or GaugeConfig()
    uq = aggregate_uncertainty(ensemble, cfg.uq, calibrator)
    gate = run_physics_gate(window, cfg.gate, hessian=hessian, masses=masses)
    return decide(uq, gate, cfg.decision)


def gauge_trajectory_from_backends(
    primary: Backend,
    ensemble: list[Backend],
    positions_traj: np.ndarray,
    atomic_numbers: np.ndarray,
    *,
    window_size: int = 4,
    cell_traj: np.ndarray | None = None,
    config: GaugeConfig | None = None,
    calibrator: IsotonicCalibrator | None = None,
    window_id_prefix: str = "win",
) -> tuple[list[GaugeDecision], ActiveLearningQueue]:
    """Slide a window over a trajectory, gauge each window, and collect
    active-learning candidates. ``primary`` supplies the canonical trajectory;
    ``ensemble`` (>=2 backends) supplies cross-model disagreement."""
    pos = np.asarray(positions_traj, dtype=np.float64)
    if pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError(f"positions_traj must be (F,N,3), got {pos.shape}")
    n_frames = pos.shape[0]
    if window_size < 2:
        raise ValueError("window_size must be >= 2 for the gate to evaluate")
    if n_frames < window_size:
        # Refuse rather than emit a degenerate short window: a sub-window-size
        # trajectory cannot run the multi-frame hard checks, so certifying it
        # would let a single static check (e.g. stress symmetry) grant TRUST.
        raise ValueError(
            f"trajectory has {n_frames} frames < window_size {window_size}; "
            "supply at least window_size frames"
        )

    decisions: list[GaugeDecision] = []
    queue = ActiveLearningQueue()
    for start in range(0, n_frames - window_size + 1):
        sl = slice(start, start + window_size)
        ptraj = pos[sl]
        ctraj = None if cell_traj is None else cell_traj[sl]
        window = evaluate_window(primary, ptraj, atomic_numbers, cell_traj=ctraj)
        ens = evaluate_ensemble_forces(ensemble, ptraj, atomic_numbers, cell_traj=ctraj)
        decision = gauge_window(window, ens, config=config, calibrator=calibrator)
        decisions.append(decision)
        queue.consider(decision, window, f"{window_id_prefix}_{start}")
    return decisions, queue
