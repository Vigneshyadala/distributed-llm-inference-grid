import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SMALL = ["--hidden", "48", "--heads", "6", "--kv-heads", "2", "--head-dim", "8",
         "--layers", "2", "--tokens", "10", "--block-size", "3"]


def run_demo(*args):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "kv_demo.py"), *args],
        capture_output=True, text=True, timeout=300,
    )


def test_demo_shows_growth_reuse_and_a_refusal():
    result = run_demo()
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert "B finishes, 3 blocks freed" in out
    assert "[AAADDCCCCad.....]" in out                      # D took the blocks B gave back
    assert "E needs 30 tokens: refused" in out and "only 5 are free" in out
    assert "12.0 KiB" in out and "6.0 KiB" in out          # Qwen2.5-0.5B memory per token, 1 and 2 workers


@pytest.mark.parametrize("workers", [1, 2])
def test_cached_decoding_check_with_threads(workers):
    result = run_demo("--check", "--backend", "threads", "--workers", str(workers), *SMALL)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "KV CACHE CHECK OK" in result.stdout


def test_check_says_so_when_the_heads_cannot_be_split():
    result = run_demo("--check", "--backend", "threads", "--workers", "4", *SMALL)
    assert result.returncode == 2
    assert "COULD NOT RUN" in result.stdout and "Traceback" not in result.stderr


def test_cached_decoding_check_with_real_processes_and_gloo():
    pytest.importorskip("torch")
    result = run_demo("--check", "--backend", "gloo", "--workers", "2", *SMALL)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "KV CACHE CHECK OK" in result.stdout
