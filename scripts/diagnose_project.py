from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES: Dict[str, List[str]] = {
    "app": [
        "app/streamlit_app.py",
        "01.py",
        "app.py",
        "archive/legacy/01.py",
        "archive/legacy/app.py",
    ],
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
        "nasa_project/utils.py",
        "utils/features.py",
        "features.py",
        "archive/legacy/utils/features.py",
        "archive/legacy/features.py",
        "kepler-koi-ml/utils/features.py",
    ],
    "nasa_utils": [
        "utils/nasa.py",
        "nasa.py",
        "archive/legacy/utils/nasa.py",
        "archive/legacy/nasa.py",
        "kepler-koi-ml/utils/nasa.py",
    ],
}


def find_first(paths: List[str]) -> Path | None:
    for p in paths:
        cp = ROOT / p
        if cp.exists():
            return cp
    return None


def check_make_targets() -> Tuple[List[str], List[str]]:
    mk = ROOT / "Makefile"
    have, missing = [], []
    want = [
        "setup",
        "dev-setup",
        "fmt",
        "lint",
        "test",
        "run",
        "clean",
        "all",
        "app",
        "list-legacy",
    ]
    if not mk.exists():
        return [], want
    text = mk.read_text()
    targets = set()
    for line in text.splitlines():
        if line.startswith("\t"):
            continue
        m = re.match(r"^([A-Za-z0-9_.-]+):", line)
        if m:
            targets.add(m.group(1))
    for t in want:
        (have if t in targets else missing).append(t)
    return have, missing


def try_import(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def run_collect_pytest() -> Tuple[int, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    return r.returncode, (r.stdout + "\n" + r.stderr).strip()


def main() -> int:
    print(f"Project root: {ROOT}")
    print("\n[Makefile targets]")
    have, missing = check_make_targets()
    print("  present:", ", ".join(sorted(have)) if have else "none")
    print("  missing:", ", ".join(sorted(missing)) if missing else "none")

    print("\n[Packages importable]")
    for m in [
        "nasa_project",
        "nasa_project.utils",
        "nasa_project.cli",
        "streamlit",
        "lightkurve",
        "pytest",
        "ruff",
        "black",
    ]:
        print(f"  {m:22s}: {'OK' if try_import(m) else 'missing'}")

    print("\n[Entrypoints and legacy scripts]")
    for name, paths in CANDIDATES.items():
        p = find_first(paths)
        status = f"OK -> {p.relative_to(ROOT)}" if p else "missing"
        print(f"  {name:20s}: {status}")

    print("\n[Tests discovery]")
    code, out = run_collect_pytest()
    first = "\n".join(out.splitlines()[-10:])
    print(f"  pytest collect rc={code}")
    print(first or "  (no output)")

    print("\n[Summary]")
    ok_app = find_first(CANDIDATES["app"]) is not None and try_import("streamlit")
    ok_tests = code == 0
    ok_make = "fmt" in have and "lint" in have and "test" in have
    print(f"  app runnable: {'YES' if ok_app else 'NO'}")
    print(f"  tests ready:  {'YES' if ok_tests else 'NO'}")
    print(f"  make ok:      {'YES' if ok_make else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
