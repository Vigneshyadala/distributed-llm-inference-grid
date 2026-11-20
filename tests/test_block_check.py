import os
import subprocess
import sys
from pathlib import Path

import pytest

from grid.runner import identify, pick_backend, run_workers

ROOT = Path(__file__).resolve().parents[1]
SMALL = ["--hidden", "48", "--heads", "6", "--kv-heads", "2", "--head-dim", "8",
         "--inter", "96", "--tokens", "5", "--repeat", "2"]


def run_block_check(*args):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "block_check.py"), *args],
        capture_output=True, text=True, timeout=300,
    )


def where_am_i(group, convert, to_numpy, offset=0):
    return group.rank + offset


@pytest.mark.parametrize("workers", [1, 2])
def test_block_check_with_threads(workers):
    result = run_block_check("--backend", "threads", "--workers", str(workers), *SMALL)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"BLOCK CHECK OK: {workers} workers" in result.stdout
    assert "2 all-reduces per pass" in result.stdout


def test_block_check_says_so_when_the_heads_cannot_be_split():
    result = run_block_check("--backend", "threads", "--workers", "4", *SMALL)  # 2 kv heads, 4 workers
    assert result.returncode == 2
    assert "COULD NOT RUN" in result.stdout and "cannot be split evenly" in result.stdout
    assert "Traceback" not in result.stderr


def test_block_check_with_real_processes_and_gloo():
    pytest.importorskip("torch")
    result = run_block_check("--backend", "gloo", "--workers", "2", *SMALL)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "BLOCK CHECK OK: 2 workers" in result.stdout


def test_run_workers_returns_results_in_rank_order():
    assert run_workers(where_am_i, 3, "threads", offset=10) == [10, 11, 12]


def test_unknown_backend_is_refused_and_auto_resolves():
    with pytest.raises(ValueError):
        pick_backend("nccl")
    assert pick_backend("auto") in ("gloo", "threads")
    assert pick_backend("threads") == "threads"


def test_gloo_workers_run_in_separate_processes():
    pytest.importorskip("torch")
    infos = run_workers(identify, 2, "gloo")
    assert [info["rank"] for info in infos] == [0, 1]
    assert all(info["world_size"] == 2 for info in infos)
    assert len({info["pid"] for info in infos} | {os.getpid()}) == 3  # two workers, neither is this process


def test_thread_workers_share_this_process():
    infos = run_workers(identify, 2, "threads")
    assert {info["pid"] for info in infos} == {os.getpid()}
