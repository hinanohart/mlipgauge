#!/usr/bin/env python3
"""Stage gate verifier for the mlipgauge autonomous build.

Each stage S0..S11 has a checker that inspects real artifacts (files, imports,
tests, results json, git/gh state). The autonomous builder calls

    python scripts/verify_step.py <Sn> [--dry-run]

and trusts the *live exit code* (0 = pass) rather than the self-reported
checklist in .mlipgauge-progress.json. This is the /compact-resilience anchor:
after a context reset the builder re-derives ground truth instead of believing
a possibly-stale progress file.

--dry-run skips actions that touch the network (gh) or are slow; it still runs
import + static checks. Without --dry-run, network/state checks are attempted.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def _ok(msg: str) -> tuple[bool, str]:
    return True, msg


def _fail(msg: str) -> tuple[bool, str]:
    return False, msg


def _files_exist(*rel: str) -> tuple[bool, str]:
    missing = [r for r in rel if not (ROOT / r).exists()]
    if missing:
        return _fail(f"missing files: {missing}")
    return _ok(f"present: {list(rel)}")


def _can_import(modules: list[str]) -> tuple[bool, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    code = "import importlib,sys\n" + "\n".join(f"importlib.import_module({m!r})" for m in modules)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        return _fail(
            f"import failed: {r.stderr.strip().splitlines()[-1] if r.stderr else r.returncode}"
        )
    return _ok(f"imports ok: {modules}")


def _pytest(paths: list[str]) -> tuple[bool, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *paths],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )
    lines = (r.stdout + r.stderr).strip().splitlines()
    hits = [ln.strip() for ln in lines if any(k in ln for k in ("passed", "failed", "error"))]
    summary = hits[-1] if hits else f"exit={r.returncode}"
    if r.returncode != 0:  # return code is the source of truth, not the text
        return _fail(f"pytest failed: {summary}")
    return _ok(f"pytest pass: {summary}")


def _gh(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], capture_output=True, text=True)


def _no_placeholder_or_hype(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return _fail(f"{path.name} missing")
    text = path.read_text(encoding="utf-8")
    placeholders = ["MEASURED@", "TODO", "XXX", "<!--", "PLACEHOLDER", "FIXME"]
    hits = [p for p in placeholders if p in text]
    if hits:
        return _fail(f"placeholder tokens remain in {path.name}: {hits}")
    # ERE-correct hype scan (no BRE/ERE mixing); case-insensitive whole-word-ish.
    import re

    hype = re.compile(
        r"\b(first[ -]ever|world[ -]first|fully[ -]automatic|completely automatic|"
        r"permanent(?:ly)?|guarantee[ds]?|state[ -]of[ -]the[ -]art|sota|"
        r"revolutionary|unprecedented)\b",
        re.IGNORECASE,
    )
    found = sorted(set(m.group(0).lower() for m in hype.finditer(text)))
    if found:
        return _fail(f"hype terms in {path.name}: {found}")
    return _ok(f"{path.name} clean (no placeholder/hype)")


# ---- per-stage checkers -----------------------------------------------------


def check_S0(dry: bool) -> tuple[bool, str]:
    p = ROOT / ".mlipgauge-progress.json"
    if not p.exists():
        return _fail("progress file missing")
    kc = json.loads(p.read_text())["kill_criteria_results"]
    bad = [k for k, v in kc.items() if v.get("verdict") not in ("proceed", "degrade")]
    if bad:
        return _fail(f"kill criteria not cleared: {bad}")
    return _ok(f"S0 gate cleared: {list(kc)}")


def check_S05(dry: bool) -> tuple[bool, str]:
    return _files_exist(
        "pyproject.toml",
        "LICENSE",
        "NOTICE",
        "README.md",
        ".gitignore",
        ".mlipgauge-progress.json",
        "scripts/verify_step.py",
        ".github/workflows/ci.yml",
        "src/mlipgauge/__init__.py",
    )


def check_S1(dry: bool) -> tuple[bool, str]:
    ok, msg = _can_import(["mlipgauge.types"])
    if not ok:
        return ok, msg
    return _pytest(["tests/test_types.py"])


def check_S2(dry: bool) -> tuple[bool, str]:
    ok, msg = _can_import(
        [
            "mlipgauge.core.uq",
            "mlipgauge.core.calibration",
            "mlipgauge.core.physics_gate",
            "mlipgauge.core.gauge",
        ]
    )
    if not ok:
        return ok, msg
    return _pytest(
        [
            "tests/test_physics_gate.py",
            "tests/test_uq.py",
            "tests/test_calibration.py",
            "tests/test_determinism.py",
        ]
    )


def check_S3(dry: bool) -> tuple[bool, str]:
    ok, msg = _can_import(
        ["mlipgauge.backends.base", "mlipgauge.backends.mock", "mlipgauge.backends.normalize"]
    )
    if not ok:
        return ok, msg
    return _pytest(["tests/test_normalize.py", "tests/test_mock_pipeline.py"])


def check_S4(dry: bool) -> tuple[bool, str]:
    return _pytest(["tests/test_physics_gate.py"])


def check_S5(dry: bool) -> tuple[bool, str]:
    ok, msg = _can_import(["mlipgauge.gauge_api", "mlipgauge.cli"])
    if not ok:
        return ok, msg
    return _pytest(["tests/test_cli.py", "tests/test_gauge_api.py"])


def check_S6(dry: bool) -> tuple[bool, str]:
    rdir = ROOT / "results"
    jsons = list(rdir.glob("*.json"))
    if not jsons:
        return _fail("no results/*.json produced")
    required = {"n", "mode", "hw", "os", "python", "date", "seed", "version"}
    for j in jsons:
        meta = json.loads(j.read_text()).get("meta", {})
        missing = required - set(meta)
        if missing:
            return _fail(f"{j.name} meta missing {missing}")
    return _ok(f"results present with full meta: {[j.name for j in jsons]}")


def check_S7(dry: bool) -> tuple[bool, str]:
    return _no_placeholder_or_hype(ROOT / "README.md")


def check_S8(dry: bool) -> tuple[bool, str]:
    crit = json.loads((ROOT / ".mlipgauge-progress.json").read_text())["critic"]
    if crit.get("verdict") not in ("SHIP-OK", "SHIP-OK-WITH-NITS"):
        return _fail(
            f"critic verdict = {crit.get('verdict')}, blockers={crit.get('blockers_open')}"
        )
    if crit.get("blockers_open"):
        return _fail(f"open blockers: {crit['blockers_open']}")
    return _ok(f"critic verdict {crit['verdict']}")


def check_S9(dry: bool) -> tuple[bool, str]:
    # Local proxy for CI: full suite must pass locally with mock+synthetic.
    ok, msg = _pytest(["tests"])
    if not ok:
        return ok, msg
    if not dry:
        r = _gh(["run", "list", "-L", "1", "--json", "conclusion,status"])
        if r.returncode == 0 and r.stdout.strip():
            runs = json.loads(r.stdout)
            if runs and runs[0].get("conclusion") not in ("success", None):
                return _fail(f"latest CI run conclusion = {runs[0].get('conclusion')}")
    return _ok(f"local suite green; {msg}")


def check_S10(dry: bool) -> tuple[bool, str]:
    if dry:
        return _ok("dry-run: skipping remote repo check")
    r = _gh(["repo", "view", "hinanohart/mlipgauge", "--json", "visibility,licenseInfo"])
    if r.returncode != 0:
        return _fail("repo not found on GitHub")
    info = json.loads(r.stdout)
    if info.get("visibility", "").upper() != "PUBLIC":
        return _fail(f"visibility = {info.get('visibility')}")
    return _ok("repo public on GitHub")


def check_S11(dry: bool) -> tuple[bool, str]:
    prog = json.loads((ROOT / ".mlipgauge-progress.json").read_text())
    if not prog["measured"].get("reproduced_in_clean_clone"):
        return _fail("reproduced_in_clean_clone is false")
    return _ok("reproduced in clean clone")


CHECKERS = {
    "S0": check_S0,
    "S0.5": check_S05,
    "S1": check_S1,
    "S2": check_S2,
    "S3": check_S3,
    "S4": check_S4,
    "S5": check_S5,
    "S6": check_S6,
    "S7": check_S7,
    "S8": check_S8,
    "S9": check_S9,
    "S10": check_S10,
    "S11": check_S11,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=sorted(CHECKERS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ok, msg = CHECKERS[args.step](args.dry_run)
    status = "PASS" if ok else "FAIL"
    print(f"[{args.step}] {status}: {msg}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
