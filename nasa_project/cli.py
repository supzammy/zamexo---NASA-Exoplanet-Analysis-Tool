from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES: Dict[str, List[str]] = {
    "evaluate_models": [
        "scripts/evaluate_models.py",
        "evaluate_models.py",
        "archive/legacy/scripts/evaluate_models.py",
        "archive/legacy/evaluate_models.py",
        "kepler-koi-ml/scripts/evaluate_models.py",
    ],
    "train_baseline": [
        "scripts/train_baseline.py",
        "train_baseline.py",
        "archive/legacy/scripts/train_baseline.py",
        "archive/legacy/train_baseline.py",
        "kepler-koi-ml/scripts/train_baseline.py",
    ],
    "train_multisource": [
        "scripts/train_multisource.py",
        "train_multisource.py",
        "archive/legacy/scripts/train_multisource.py",
        "archive/legacy/train_multisource.py",
        "kepler-koi-ml/scripts/train_multisource.py",
    ],
    "verify_bls": [
        "scripts/verify_bls.py",
        "verify_bls.py",
        "archive/legacy/scripts/verify_bls.py",
        "archive/legacy/verify_bls.py",
        "kepler-koi-ml/scripts/verify_bls.py",
    ],
    "verify_data_sources": [
        "scripts/verify_data_sources.py",
        "verify_data_sources.py",
        "archive/legacy/scripts/verify_data_sources.py",
        "archive/legacy/verify_data_sources.py",
        "kepler-koi-ml/scripts/verify_data_sources.py",
    ],
    "features": [
        "nasa_project/features.py",
        "nasa_project/utils.py",
        "utils/features.py",
        "features.py",
        "archive/legacy/utils/features.py",
        "archive/legacy/features.py",
        "kepler-koi-ml/utils/features.py",
    ],
    "app": [
        "app/streamlit_app.py",
        "01.py",
        "app.py",
        "archive/legacy/01.py",
        "archive/legacy/app.py",
    ],
}


def _env():
    e = os.environ.copy()
    e["PYTHONPATH"] = f"{ROOT}:{e.get('PYTHONPATH', '')}"
    return e


def find_first(paths: List[str]):
    for p in paths:
        cp = ROOT / p
        if cp.exists():
            return cp
    return None


def run_script(script, script_args: List[str]) -> int:
    cmd = [sys.executable, str(script), *script_args]
    return subprocess.call(cmd, env=_env(), cwd=ROOT)


def run_streamlit(script, extra: List[str]) -> int:
    cmd = ["streamlit", "run", str(script), *extra]
    return subprocess.call(cmd, env=_env(), cwd=ROOT)


def cmd_list() -> int:
    print("Detectable commands:")
    for name, paths in CANDIDATES.items():
        p = find_first(paths)
        status = f"OK -> {p.relative_to(ROOT)}" if p else "missing"
        print(f"  - {name:20s} {status}")
    return 0


def cmd_run(name: str, script_args: List[str]) -> int:
    p = find_first(CANDIDATES[name])
    if not p:
        print(
            f"[error] No script found for '{name}'. Run: python -m nasa_project list",
            file=sys.stderr,
        )
        return 2
    if name == "app":
        return run_streamlit(p, script_args)
    return run_script(p, script_args)


def cmd_app(extra: List[str]) -> int:
    p = find_first(CANDIDATES["app"])
    if not p:
        print(
            "[error] No app entrypoint found (expected app/streamlit_app.py, 01.py, or app.py)",
            file=sys.stderr,
        )
        return 2
    return run_streamlit(p, extra)


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="nasa", description="NASA project CLI (legacy runner)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List detectable legacy scripts").set_defaults(
        func=lambda a: cmd_list()
    )

    pr = sub.add_parser("run", help="Run a legacy script by name")
    pr.add_argument("name", choices=sorted(CANDIDATES.keys()))
    pr.add_argument(
        "script_args", nargs=argparse.REMAINDER, help="Args for the underlying script"
    )
    pr.set_defaults(func=lambda a: cmd_run(a.name, a.script_args))

    pa = sub.add_parser("app", help="Run the Streamlit app (auto-detect entrypoint)")
    pa.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args for Streamlit")
    pa.set_defaults(func=lambda a: cmd_app(a.extra))

    args = p.parse_args(argv)
    return int(args.func(args))
