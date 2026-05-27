"""S6: the measurement scripts must be deterministic and produce honest,
reproducible numbers (so results/*.json can be regenerated in a clean clone)."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

import measure_calibration  # noqa: E402
import measure_gate  # noqa: E402


def test_gate_measurement_deterministic_and_detects():
    a = measure_gate.measure(n=20, seed=0)
    b = measure_gate.measure(n=20, seed=0)
    assert a["sensitivity"] == b["sensitivity"]  # deterministic
    for stats in a["sensitivity"].values():
        assert stats["rate"] == 1.0  # catches injected violations
    assert a["specificity_clean"]["rate"] == 1.0  # no false positives on clean
    assert a["meta"]["mode"] == "synthetic"


def test_gate_threshold_sweep_is_graded_not_always_on():
    sweep = measure_gate.measure(n=20, seed=1)["energy_force_threshold_sweep"]
    assert sweep["0.5x_tol"] == 0.0  # below tol: silent
    assert sweep["0.9x_tol"] == 0.0
    assert sweep["1.1x_tol"] == 1.0  # above tol: fires
    assert sweep["5.0x_tol"] == 1.0


def test_calibration_reduces_ece_and_is_marked_synthetic():
    r = measure_calibration.measure(n=1000, seed=0)
    assert r["ece_calibrated"]["value"] < r["ece_raw"]["value"]
    assert r["ece_reduction"] > 0
    assert r["meta"]["mode"] == "synthetic"
    assert r["non_claim"] is True
