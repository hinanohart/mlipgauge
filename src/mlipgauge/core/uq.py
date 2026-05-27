"""Uncertainty quantification (0.1.0a2: single source = cross-model force
disagreement).

Given force predictions from K>=2 independent MLIPs over the same frames, the
epistemic uncertainty of an atom is estimated as the root-mean-square deviation
of its predicted force vector across models. This is aggregated per frame and
then over the window to a raw scale ``u_raw`` (eV/Å), and squashed into [0,1].

The metric itself is standard (cross-model disagreement; cf. heterogeneous
ensembles for atomistic foundation models). mlipgauge does not claim a novel UQ
estimator in 0.1.0a2 — heterogeneous-UQ calibration is deferred to a later
release. The contribution here is wiring this signal, fail-closed, into the
runtime gauge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlipgauge.core.calibration import IsotonicCalibrator
from mlipgauge.types import EnsembleForces, UQResult


@dataclass(frozen=True)
class UQConfig:
    # scale (eV/Å) at which squashed uncertainty reaches 1 - 1/e ~ 0.63.
    scale_ev_per_ang: float = 0.1
    # aggregation across atoms within a frame: "mean" (typical) or "max" (worst-atom).
    atom_reduce: str = "mean"

    def __post_init__(self) -> None:
        if not (self.scale_ev_per_ang > 0 and np.isfinite(self.scale_ev_per_ang)):
            raise ValueError("scale_ev_per_ang must be positive and finite")
        if self.atom_reduce not in ("mean", "max"):
            raise ValueError("atom_reduce must be 'mean' or 'max'")


def _per_atom_disagreement(forces: np.ndarray) -> np.ndarray:
    """forces: (K, F, N, 3) -> (F, N) RMS deviation of the force vector across
    the K models (eV/Å)."""
    mean = forces.mean(axis=0)  # (F, N, 3)
    dev = forces - mean[None]  # (K, F, N, 3)
    sq = np.einsum("kfnd,kfnd->kfn", dev, dev)  # (K, F, N) squared vector norm
    return np.sqrt(sq.mean(axis=0))  # (F, N)


def aggregate_uncertainty(
    ensemble: EnsembleForces,
    config: UQConfig | None = None,
    calibrator: IsotonicCalibrator | None = None,
) -> UQResult:
    """Reduce ensemble force disagreement to a single calibrated/squashed scalar.

    Fail-closed: if any prediction is non-finite, returns maximal uncertainty
    (u=1.0) flagged ``abstain_nonfinite`` rather than a falsely-confident value.
    """
    cfg = config or UQConfig()
    f = ensemble.forces
    if not np.isfinite(f).all():
        n_frames = f.shape[1]
        return UQResult(
            u_raw=1e9,  # sentinel "very large" raw scale for a non-finite prediction
            u=1.0,
            per_frame=np.full(n_frames, np.nan),
            method="abstain_nonfinite",
            calibrated=False,
        )

    per_atom = _per_atom_disagreement(f)  # (F, N)
    if cfg.atom_reduce == "mean":
        per_frame = per_atom.mean(axis=1)  # (F,)
    else:
        per_frame = per_atom.max(axis=1)
    u_raw = float(per_frame.mean())

    if calibrator is not None:
        u = calibrator.predict_one(u_raw)
        calibrated = True
    else:
        # monotone saturating squash: 0 at no disagreement, ->1 as disagreement grows
        u = float(1.0 - np.exp(-u_raw / cfg.scale_ev_per_ang))
        calibrated = False
    u = float(min(max(u, 0.0), 1.0))
    return UQResult(
        u_raw=u_raw, u=u, per_frame=per_frame, method=cfg.atom_reduce, calibrated=calibrated
    )
