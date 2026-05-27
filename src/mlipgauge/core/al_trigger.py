"""Active-learning trigger / queue.

When the gauge does not cleanly trust a window (hard violation, abstain, or
elevated uncertainty), the offending configuration is the most valuable thing to
label with DFT next. This module collects such configurations, de-duplicates by
caller-supplied id, picks the single worst frame as the representative to label,
and exports a serialisable queue. It does not run DFT — it decides *what* to
label, which is the active-learning contribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mlipgauge.types import GaugeDecision, TrajectoryWindow, Verdict


@dataclass(frozen=True)
class ALCandidate:
    window_id: str
    frame_index: int
    Q: float
    verdict: Verdict
    u: float
    reasons: list[str]
    positions: np.ndarray  # (N,3) the representative (worst) frame
    atomic_numbers: np.ndarray  # (N,)

    def to_record(self) -> dict:
        return {
            "window_id": self.window_id,
            "frame_index": self.frame_index,
            "Q": self.Q,
            "verdict": str(self.verdict),
            "u": self.u,
            "reasons": list(self.reasons),
            "positions": self.positions.tolist(),
            "atomic_numbers": self.atomic_numbers.tolist(),
        }


def _worst_frame(decision: GaugeDecision) -> int:
    pf = decision.uq.per_frame
    if pf is None or pf.size == 0 or not np.isfinite(pf).any():
        return 0
    return int(np.nanargmax(pf))


@dataclass
class ActiveLearningQueue:
    """Accumulates configurations worth labelling. De-dups by ``window_id``."""

    max_size: int | None = None
    _items: list[ALCandidate] = field(default_factory=list, init=False)
    _seen: set[str] = field(default_factory=set, init=False)

    def consider(self, decision: GaugeDecision, window: TrajectoryWindow, window_id: str) -> bool:
        """Queue the window's worst frame iff the decision flagged it for AL.
        Returns True if a new candidate was added."""
        if not decision.active_learning:
            return False
        if window_id in self._seen:
            return False
        if self.max_size is not None and len(self._items) >= self.max_size:
            return False
        idx = _worst_frame(decision)
        cand = ALCandidate(
            window_id=window_id,
            frame_index=idx,
            Q=decision.Q,
            verdict=decision.verdict,
            u=decision.uq.u,
            reasons=list(decision.gate.reasons),
            positions=np.asarray(window.positions[idx], dtype=np.float64),
            atomic_numbers=np.asarray(window.atomic_numbers),
        )
        self._items.append(cand)
        self._seen.add(window_id)
        return True

    def export(self) -> list[dict]:
        return [c.to_record() for c in self._items]

    def __len__(self) -> int:
        return len(self._items)
