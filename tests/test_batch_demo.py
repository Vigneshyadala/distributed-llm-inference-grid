import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_demo(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "batch_demo.py"), *args],
                          capture_output=True, text=True, timeout=120)


def test_demo_shows_continuous_batching_winning():
    result = run_demo("--check")
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert "STATIC batching" in out and "CONTINUOUS batching" in out
    assert "BATCH CHECK OK: continuous finished in 10 steps against 13" in out
    assert "blocks free at the end 40" in out


def test_demo_check_fails_clearly_when_the_cache_is_too_small():
    result = run_demo("--check", "--blocks", "3")
    assert result.returncode == 1
    assert "BATCH CHECK FAILED" in result.stdout and "limited the batch" in result.stdout
