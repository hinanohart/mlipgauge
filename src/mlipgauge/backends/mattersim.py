"""MatterSim backend (weights: MIT). Optional extra ``mlipgauge[mattersim]``.

Live inference is deferred / not CI-tested (see _ase_common).
"""

from __future__ import annotations

import numpy as np

from mlipgauge.backends._ase_common import ase_predict
from mlipgauge.backends.base import Prediction


class MatterSimBackend:
    name = "mattersim"
    license = "MIT"

    def __init__(self, model: str = "MatterSim-v1.0.0-1M", device: str = "cpu", **kwargs):
        try:
            from mattersim.forcefield import MatterSimCalculator
        except ImportError as e:
            raise ImportError(
                "MatterSim backend requires the extra: pip install 'mlipgauge[mattersim]'"
            ) from e
        self._calc = MatterSimCalculator(load_path=model, device=device, **kwargs)

    def predict(
        self, positions: np.ndarray, atomic_numbers: np.ndarray, cell: np.ndarray | None = None
    ) -> Prediction:
        return ase_predict(self._calc, positions, atomic_numbers, cell)
