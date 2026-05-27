"""Command-line interface for mlipgauge.

    mlipgauge info                 # version + backend license matrix
    mlipgauge demo                 # deterministic mock-backend pipeline (synthetic)
    mlipgauge gauge traj.npz       # gauge a real trajectory file

Honesty: ``demo`` runs the dependency-free *mock* backend on *synthetic*
positions; its output demonstrates the pipeline, it is NOT a benchmark and
prints no calibration metric as if measured.
"""

from __future__ import annotations

import json

import numpy as np
import typer

from mlipgauge import __version__
from mlipgauge.backends.base import ALLOWED_LICENSES, BACKEND_LICENSES
from mlipgauge.backends.mock import MockBackend
from mlipgauge.gauge_api import gauge_trajectory_from_backends

app = typer.Typer(add_completion=False, help="Runtime guardrail for MLIP molecular dynamics.")


def _summarize(decisions) -> dict:
    from collections import Counter

    counts = Counter(str(d.verdict) for d in decisions)
    return {
        "n_windows": len(decisions),
        "verdicts": dict(counts),
        "mean_Q": float(np.mean([d.Q for d in decisions])) if decisions else None,
        "n_active_learning": sum(1 for d in decisions if d.active_learning),
    }


@app.command()
def info() -> None:
    """Print version and the backend weight-license allow-list."""
    typer.echo(f"mlipgauge {__version__}")
    typer.echo(f"allowed weight licenses: {sorted(ALLOWED_LICENSES)}")
    typer.echo("backends:")
    for name, lic in BACKEND_LICENSES.items():
        ok = "allowed" if lic in ALLOWED_LICENSES else "REFUSED (not on allow-list)"
        typer.echo(f"  {name:14s} {lic:14s} {ok}")


@app.command()
def demo(frames: int = 6, atoms: int = 4, seed: int = 0, window_size: int = 4) -> None:
    """Run the deterministic mock-backend pipeline on a synthetic trajectory."""
    rng = np.random.default_rng(seed)
    z = np.array([1, 6, 8, 7][:atoms] + [1] * max(0, atoms - 4))[:atoms]
    base = rng.standard_normal((atoms, 3)) * 0.2
    step = rng.standard_normal((atoms, 3)) * 1e-3
    traj = np.stack([base + t * step for t in range(frames)])
    primary = MockBackend(k=1.5)
    ensemble = [MockBackend(k=1.5), MockBackend(k=1.51), MockBackend(k=1.49)]
    decisions, queue = gauge_trajectory_from_backends(
        primary, ensemble, traj, z, window_size=window_size
    )
    out = _summarize(decisions)
    out["active_learning_queue"] = len(queue)
    out["note"] = "mock backend, synthetic trajectory — demonstration, not a benchmark"
    typer.echo(json.dumps(out, indent=2))


@app.command()
def gauge(
    path: str,
    window_size: int = 4,
    out: str = typer.Option("", help="optional path to write per-window decisions JSON"),
) -> None:
    """Gauge a trajectory from a .npz with keys: positions (F,N,3), atomic_numbers
    (N,), optional cell (F,3,3). Uses a deterministic mock ensemble (real-backend
    inference is deferred to a later release)."""
    data = np.load(path)
    positions = np.asarray(data["positions"], dtype=np.float64)
    z = np.asarray(data["atomic_numbers"])
    cell = np.asarray(data["cell"], dtype=np.float64) if "cell" in data else None
    primary = MockBackend(k=1.5)
    ensemble = [MockBackend(k=1.5), MockBackend(k=1.51), MockBackend(k=1.49)]
    decisions, queue = gauge_trajectory_from_backends(
        primary, ensemble, positions, z, window_size=window_size, cell_traj=cell
    )
    summary = _summarize(decisions)
    summary["active_learning_queue"] = queue.export()
    text = json.dumps(summary, indent=2)
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text)
        typer.echo(f"wrote {out}")
    else:
        typer.echo(text)


if __name__ == "__main__":  # pragma: no cover
    app()
