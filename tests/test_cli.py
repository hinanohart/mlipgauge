"""S5: CLI smoke tests via Typer's CliRunner."""

from __future__ import annotations

import json

import numpy as np
from typer.testing import CliRunner

from mlipgauge import __version__
from mlipgauge.cli import app

runner = CliRunner()


def test_info_lists_version_and_refused_asl():
    res = runner.invoke(app, ["info"])
    assert res.exit_code == 0
    assert __version__ in res.stdout
    assert "mace-omat-0" in res.stdout
    assert "REFUSED" in res.stdout


def test_demo_emits_valid_json():
    res = runner.invoke(app, ["demo", "--frames", "6", "--atoms", "4"])
    assert res.exit_code == 0
    payload = json.loads(res.stdout)
    assert payload["n_windows"] >= 1
    assert "verdicts" in payload
    assert "not a benchmark" in payload["note"]


def test_gauge_reads_npz(tmp_path):
    rng = np.random.default_rng(0)
    n = 3
    base = rng.standard_normal((n, 3)) * 0.2
    traj = np.stack([base + t * 1e-3 for t in range(6)])
    p = tmp_path / "traj.npz"
    np.savez(p, positions=traj, atomic_numbers=np.array([1, 6, 8]))
    res = runner.invoke(app, ["gauge", str(p), "--window-size", "4"])
    assert res.exit_code == 0
    payload = json.loads(res.stdout)
    assert payload["n_windows"] == 3
