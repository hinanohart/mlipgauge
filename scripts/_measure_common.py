"""Shared helpers for measurement scripts: provenance metadata so every number
in results/ is traceable to where/how it was produced."""

from __future__ import annotations

import datetime
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mlipgauge import __version__  # noqa: E402


def make_meta(n: int, mode: str, seed: int) -> dict:
    """mode is 'synthetic' (constructed data) or 'live' (real backend/DFT)."""
    assert mode in ("synthetic", "live")
    return {
        "n": n,
        "mode": mode,
        "hw": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": seed,
        "version": __version__,
    }
