"""Calibration utilities (NON-CLAIM layer).

This module implements a standard isotonic-regression (PAVA) calibrator plus
ECE and a bootstrap CI helper, in pure NumPy. It is deliberately *not* a novel
contribution: the same PAVA + holdout-ECE recipe appears across the literature
and in sibling hinanohart calibration repos (foldconsensus / foldgauge). It is
re-implemented self-contained here rather than imported from those (unpublished,
pre-alpha) packages so that mlipgauge stays pip-installable with no cross-repo
dependency and no schema drift. The intellectual core of mlipgauge is
``physics_gate``; calibration only shapes the soft uncertainty signal.

Honesty contract: any calibrated number reported by mlipgauge that was produced
with synthetic labels MUST be marked synthetic. See ``IsotonicCalibrator.fit``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators: the non-decreasing fit g minimizing
    ``sum_i w_i (y_i - g_i)^2``. Inputs are assumed already ordered by the
    covariate. O(n) via a block stack."""
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if y.shape != w.shape or y.ndim != 1:
        raise ValueError("y and w must be 1-D arrays of equal length")
    vals: list[float] = []
    wts: list[float] = []
    cnts: list[int] = []
    for yi, wi in zip(y, w, strict=True):
        vals.append(float(yi))
        wts.append(float(wi))
        cnts.append(1)
        # pool while the monotonicity (non-decreasing) constraint is violated
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnts.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnts.pop()
            wt = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / wt)
            wts.append(wt)
            cnts.append(c1 + c2)
    out = np.empty(y.size, dtype=np.float64)
    idx = 0
    for v, c in zip(vals, cnts, strict=True):
        out[idx : idx + c] = v
        idx += c
    return out


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Monotone score -> probability map fit by PAVA.

    ``synthetic`` records whether the fitting labels were synthetic, so that
    downstream reporting can refuse to present the output as a measured claim.
    """

    x: np.ndarray  # sorted breakpoint scores
    y: np.ndarray  # monotone non-decreasing calibrated values in [0,1]
    synthetic: bool = False

    @classmethod
    def fit(cls, scores, labels, *, synthetic: bool = False) -> IsotonicCalibrator:
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        if scores.shape != labels.shape or scores.ndim != 1:
            raise ValueError("scores and labels must be 1-D arrays of equal length")
        if scores.size == 0:
            raise ValueError("cannot fit on empty data")
        if not np.all(np.isin(np.unique(labels), (0.0, 1.0))):
            raise ValueError("labels must be binary {0,1}")
        order = np.argsort(scores, kind="mergesort")
        s = scores[order]
        g = _pava(labels[order], np.ones(labels.size))
        g = np.clip(g, 0.0, 1.0)
        return cls(x=s, y=g, synthetic=bool(synthetic))

    def predict(self, scores) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        if self.x.size == 1:  # degenerate single-block fit
            return np.full(scores.shape, self.y[0], dtype=np.float64)
        return np.interp(scores, self.x, self.y, left=self.y[0], right=self.y[-1])

    def predict_one(self, score: float) -> float:
        return float(self.predict(np.array([score]))[0])


def expected_calibration_error(probs, labels, n_bins: int = 10) -> float:
    """Equal-width-bin ECE. ``probs`` and ``labels`` in [0,1]/{0,1}."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if probs.shape != labels.shape or probs.ndim != 1:
        raise ValueError("probs and labels must be 1-D arrays of equal length")
    if probs.size == 0:
        raise ValueError("empty input")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
    n = probs.size
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            conf = probs[mask].mean()
            acc = labels[mask].mean()
            ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def bootstrap_ci(
    values, statistic=np.mean, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``statistic`` over a 1-D sample."""
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim != 1 or vals.size == 0:
        raise ValueError("values must be a non-empty 1-D array")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boot = np.array([statistic(vals[row]) for row in idx], dtype=np.float64)
    lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)
