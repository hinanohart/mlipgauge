"""CHGNet backend (weights: BSD-3-Clause). Optional extra ``mlipgauge[chgnet]``.

Live inference is deferred / not CI-tested (see _ase_common).
"""

from __future__ import annotations

import numpy as np

from mlipgauge.backends._ase_common import ase_predict
from mlipgauge.backends.base import Prediction


class ChgnetBackend:
    name = "chgnet"
    license = "BSD-3-Clause"

    def __init__(self, device: str = "cpu", **kwargs):
        try:
            from chgnet.model.dynamics import CHGNetCalculator
        except ImportError as e:
            raise ImportError(
                "CHGNet backend requires the extra: pip install 'mlipgauge[chgnet]'"
            ) from e
        self._calc = CHGNetCalculator(use_device=device, **kwargs)

    def predict(
        self, positions: np.ndarray, atomic_numbers: np.ndarray, cell: np.ndarray | None = None
    ) -> Prediction:
        return ase_predict(self._calc, positions, atomic_numbers, cell)
