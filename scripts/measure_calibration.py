"""Measure the isotonic calibrator on SYNTHETIC data (NON-CLAIM).

This only demonstrates that the calibration code reduces ECE on a constructed
miscalibrated score distribution with a train/holdout split. It is explicitly
NOT a materials/MLIP claim — no real uncertainty/error pairs are used. mode is
recorded as "synthetic" and a disclaimer is embedded so the README cannot
present these numbers as a measured product capability.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from _measure_common import make_meta  # noqa: E402

from mlipgauge import __version__  # noqa: E402
from mlipgauge.core.calibration import (  # noqa: E402
    IsotonicCalibrator,
    bootstrap_ci,
    expected_calibration_error,
)


def measure(n: int = 4000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    # latent "true" probability, observed binary outcome, and a miscalibrated score
    p_true = rng.uniform(0, 1, n)
    labels = (rng.uniform(0, 1, n) < p_true).astype(float)
    # overconfident, monotone-distorted score (needs calibration)
    scores = np.clip(p_true**1.7 + rng.normal(0, 0.05, n), 0, 1)

    half = n // 2
    cal = IsotonicCalibrator.fit(scores[:half], labels[:half], synthetic=True)

    raw_hold = scores[half:]
    cal_hold = cal.predict(raw_hold)
    lab_hold = labels[half:]

    ece_raw = expected_calibration_error(raw_hold, lab_hold)
    ece_cal = expected_calibration_error(cal_hold, lab_hold)

    # bootstrap CI of ECE: resample holdout indices (passed as float "values",
    # cast back to int inside the statistic), so the CI reflects holdout sampling.
    idx = np.arange(lab_hold.size).astype(float)

    def _ece_raw(sample_idx):
        i = sample_idx.astype(int)
        return expected_calibration_error(raw_hold[i], lab_hold[i])

    def _ece_cal(sample_idx):
        i = sample_idx.astype(int)
        return expected_calibration_error(cal_hold[i], lab_hold[i])

    raw_ci = bootstrap_ci(idx, statistic=_ece_raw, n_boot=200, seed=seed + 2)
    cal_ci = bootstrap_ci(idx, statistic=_ece_cal, n_boot=200, seed=seed + 3)

    return {
        "meta": make_meta(n=n, mode="synthetic", seed=seed),
        "non_claim": True,
        "disclaimer": "SYNTHETIC calibration demo only; not a measured MLIP/materials "
        "capability. Calibration is a non-claim layer of mlipgauge.",
        "ece_raw": {"value": ece_raw, "ci95": list(raw_ci)},
        "ece_calibrated": {"value": ece_cal, "ci95": list(cal_ci)},
        "ece_reduction": ece_raw - ece_cal,
    }


def main() -> int:
    out = measure()
    path = Path(__file__).resolve().parent.parent / "results" / f"{__version__}_calibration.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
    print(json.dumps({k: out[k] for k in ("ece_raw", "ece_calibrated", "ece_reduction")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
