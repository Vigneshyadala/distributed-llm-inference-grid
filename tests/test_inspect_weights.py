import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = str(ROOT / "scripts" / "inspect_weights.py")


def run(*args):
    return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True, timeout=120)


def test_demo_prints_the_shard_plan_for_two_workers():
    result = run("--demo", "--workers", "2")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "each of 2 workers holds" in result.stdout
    assert "q_proj.weight" in result.stdout and "column" in result.stdout and "row" in result.stdout


def test_demo_reports_the_tensors_that_four_workers_cannot_share():
    result = run("--demo", "--workers", "4")
    assert result.returncode == 1
    assert "cannot be split across 4 workers" in result.stdout


def test_a_real_path_needs_a_head_dim(tmp_path):
    result = run(str(tmp_path))
    assert result.returncode != 0
    assert "--head-dim" in result.stderr
