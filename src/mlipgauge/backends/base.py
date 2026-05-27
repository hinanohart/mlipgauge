"""Backend protocol and a fail-closed license allow-list.

A backend turns an atomic configuration into (energy, forces, stress). The only
dependency-free backend is the deterministic ``mock``; real backends (MACE /
CHGNet / MatterSim) are optional extras imported lazily.

License enforcement: mlipgauge keeps an explicit allow-list of commercially
usable weight licenses. ``load_backend`` refuses, fail-closed, to instantiate a
backend whose declared weight license is not allowed (e.g. MACE-OMAT-0's ASL),
unless the caller *explicitly* opts that license in — making the acceptance a
deliberate, auditable act rather than a silent default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

# Weight licenses considered commercially usable by default.
ALLOWED_LICENSES: frozenset[str] = frozenset(
    {"MIT", "BSD-3-Clause", "BSD-2-Clause", "Apache-2.0", "CC0-1.0"}
)

# Declared weight license per known backend name. Names not present here cannot
# be loaded at all (unknown provenance -> refuse).
BACKEND_LICENSES: dict[str, str] = {
    "mock": "Apache-2.0",  # deterministic, ships no weights
    "mace-mp-0": "MIT",
    "chgnet": "BSD-3-Clause",
    "mattersim": "MIT",
    "mace-omat-0": "ASL",  # non-commercial: deliberately NOT allowed
}


class LicenseError(RuntimeError):
    """Raised when a backend's weight license is not on the allow-list."""


@dataclass(frozen=True)
class Prediction:
    """One backend evaluation of a single configuration (canonical units:
    energy eV, forces eV/Å of shape (N,3), stress eV/Å³ of shape (3,3) or None)."""

    energy: float
    forces: np.ndarray
    stress: np.ndarray | None = None

    def __post_init__(self) -> None:
        f = np.asarray(self.forces, dtype=np.float64)
        if f.ndim != 2 or f.shape[1] != 3:
            raise ValueError(f"forces must be (N,3), got {f.shape}")
        object.__setattr__(self, "forces", f)
        if self.stress is not None:
            s = np.asarray(self.stress, dtype=np.float64)
            if s.shape != (3, 3):
                raise ValueError(f"stress must be (3,3), got {s.shape}")
            object.__setattr__(self, "stress", s)


@runtime_checkable
class Backend(Protocol):
    """Minimal contract every backend implements."""

    name: str
    license: str

    def predict(
        self,
        positions: np.ndarray,
        atomic_numbers: np.ndarray,
        cell: np.ndarray | None = None,
    ) -> Prediction:
        """positions (N,3) Å, atomic_numbers (N,), optional cell (3,3) Å."""
        ...


def assert_license_allowed(name: str, *, allow_extra: frozenset[str] = frozenset()) -> str:
    """Return the backend's license if allowed, else raise. ``allow_extra`` lets a
    caller explicitly accept additional licenses (e.g. an academic user enabling
    an ASL checkpoint)."""
    if name not in BACKEND_LICENSES:
        raise LicenseError(f"unknown backend {name!r}: no declared weight license")
    lic = BACKEND_LICENSES[name]
    if lic not in ALLOWED_LICENSES and lic not in allow_extra:
        raise LicenseError(
            f"backend {name!r} weight license {lic!r} is not on the allow-list "
            f"{sorted(ALLOWED_LICENSES)}; pass allow_extra={{{lic!r}}} to accept it explicitly"
        )
    return lic


def load_backend(name: str, *, allow_extra: frozenset[str] = frozenset(), **kwargs) -> Backend:
    """Instantiate a backend by name after the license check (fail-closed)."""
    assert_license_allowed(name, allow_extra=allow_extra)
    if name == "mock":
        from mlipgauge.backends.mock import MockBackend

        return MockBackend(**kwargs)
    if name == "mace-mp-0":
        from mlipgauge.backends.mace import MaceBackend

        return MaceBackend(**kwargs)
    if name == "chgnet":
        from mlipgauge.backends.chgnet import ChgnetBackend

        return ChgnetBackend(**kwargs)
    if name == "mattersim":
        from mlipgauge.backends.mattersim import MatterSimBackend

        return MatterSimBackend(**kwargs)
    raise LicenseError(f"backend {name!r} has a license entry but no loader")
