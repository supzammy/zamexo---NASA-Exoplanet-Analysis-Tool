import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    return subprocess.run(
        [sys.executable, "-m", "nasa_project", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )


def test_cli_list_runs():
    r = run_cli("list")
    assert r.returncode == 0
    assert "Detectable commands:" in r.stdout
