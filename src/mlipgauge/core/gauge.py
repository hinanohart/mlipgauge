"""Combine the physics gate and the calibrated uncertainty into one decision.

    Q = hard_valid · (1 − u) · ∏ soft_scores

- ``hard_valid ∈ {0,1}`` from the physics gate makes Q collapse to 0 on any hard
  violation (fail-closed).
- ``(1 − u)`` turns calibrated uncertainty into trust.
- the soft scores (∈[0,1]) gently down-weight near-violations.

Verdict logic is deliberately conservative: any hard violation -> HALT; an
uncomputable uncertainty or a window where *no* hard check could even run ->
ABSTAIN (also fail-closed); otherwise TRUST iff Q clears the threshold, else FLAG.
Anything that is not a clean TRUST is queued for active learning.
"""

from __future__ import annotations

from dataclasses import dataclass

from mlipgauge.types import GateResult, GaugeDecision, UQResult, Verdict


@dataclass(frozen=True)
class DecisionConfig:
    trust_threshold: float = 0.7  # Q >= this (and hard ok) -> TRUST
    al_uncertainty_threshold: float = 0.5  # u >= this -> active-learning queue
    treat_all_skipped_as_abstain: bool = True  # no hard check ran -> cannot certify

    def __post_init__(self) -> None:
        for name in ("trust_threshold", "al_uncertainty_threshold"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0,1]")


def _soft_health(soft_scores: dict[str, float]) -> float:
    health = 1.0
    for v in soft_scores.values():
        health *= v
    return health


def decide(uq: UQResult, gate: GateResult, config: DecisionConfig | None = None) -> GaugeDecision:
    cfg = config or DecisionConfig()

    # 1) hard physics violation -> fail-closed HALT
    if gate.hard_valid == 0:
        return GaugeDecision(Q=0.0, verdict=Verdict.HALT, uq=uq, gate=gate, active_learning=True)

    # 2) uncertainty uncomputable -> fail-closed ABSTAIN
    if uq.method == "abstain_nonfinite":
        return GaugeDecision(Q=0.0, verdict=Verdict.ABSTAIN, uq=uq, gate=gate, active_learning=True)

    # 3) no hard check could run (all skipped) -> cannot certify -> ABSTAIN
    if cfg.treat_all_skipped_as_abstain and len(gate.hard_checks) == 0:
        return GaugeDecision(Q=0.0, verdict=Verdict.ABSTAIN, uq=uq, gate=gate, active_learning=True)

    trust = 1.0 - uq.u
    health = _soft_health(gate.soft_scores)
    q = float(max(0.0, min(1.0, gate.hard_valid * trust * health)))

    active_learning = uq.u >= cfg.al_uncertainty_threshold
    if q >= cfg.trust_threshold:
        verdict = Verdict.TRUST
    else:
        verdict = Verdict.FLAG
        active_learning = True  # anything not cleanly trusted is worth labelling

    return GaugeDecision(Q=q, verdict=verdict, uq=uq, gate=gate, active_learning=active_learning)
