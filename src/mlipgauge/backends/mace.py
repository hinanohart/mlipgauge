"""MACE-MP-0 backend (weights: MIT). Optional extra ``mlipgauge[mace]``.

Live inference is deferred / not CI-tested (see _ase_common). Install with
``pip install 'mlipgauge[mace]'``. MACE-OMAT-0 (ASL, non-commercial) is NOT
loadable via this backend by design.
"""

from __future__ import annotations

import numpy as np

from mlipgauge.backends._ase_common import ase_predict
from mlipgauge.backends.base import Prediction


class MaceBackend:
    name = "mace-mp-0"
    license = "MIT"

    def __init__(self, model: str = "medium", device: str = "cpu", **kwargs):
        try:
            from mace.calculators import mace_mp
        except ImportError as e:
            raise ImportError(
                "MACE backend requires the extra: pip install 'mlipgauge[mace]'"
            ) from e
        self._calc = mace_mp(model=model, device=device, default_dtype="float64", **kwargs)

    def predict(
        self, positions: np.ndarray, atomic_numbers: np.ndarray, cell: np.ndarray | None = None
    ) -> Prediction:
        return ase_predict(self._calc, positions, atomic_numbers, cell)
