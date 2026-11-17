import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_two_separate_processes_form_a_grid():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "hello_grid.py"), "--workers", "2"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "worker-0" in result.stdout and "worker-1" in result.stdout
    assert "2 workers in 2 distinct processes" in result.stdout
