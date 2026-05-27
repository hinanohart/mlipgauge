"""mlipgauge — uncertainty-driven runtime guardrail for MLIP molecular dynamics.

Public surface (lazily resolved, so importing a submodule never drags in heavy
or not-yet-built siblings):

    from mlipgauge import gauge_window, GaugeConfig, GaugeDecision
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0a1"

# name -> ("module-tag", attr). Module tag is dispatched by *literal* imports
# below (no dynamic import_module of a variable: keeps the import surface a
# fixed, auditable whitelist).
_TYPES_NAMES = frozenset({"TrajectoryWindow", "UQResult", "GateResult", "GaugeDecision", "Verdict"})
_API_NAMES = frozenset({"gauge_window", "GaugeConfig"})

__all__ = [
    "__version__",
    "gauge_window",
    "GaugeConfig",
    "TrajectoryWindow",
    "UQResult",
    "GateResult",
    "GaugeDecision",
    "Verdict",
]

if TYPE_CHECKING:  # pragma: no cover
    from mlipgauge.gauge_api import GaugeConfig, gauge_window
    from mlipgauge.types import (
        GateResult,
        GaugeDecision,
        TrajectoryWindow,
        UQResult,
        Verdict,
    )


def __getattr__(name: str):
    if name in _TYPES_NAMES:
        from mlipgauge import types as mod

        return getattr(mod, name)
    if name in _API_NAMES:
        from mlipgauge import gauge_api as mod

        return getattr(mod, name)
    raise AttributeError(f"module 'mlipgauge' has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
