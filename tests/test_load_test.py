import importlib.util
import socket
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
pytest.importorskip("websockets")

import uvicorn  # noqa: E402

from grid.gateway import build_engine, create_app  # noqa: E402

spec = importlib.util.spec_from_file_location("load_test", Path(__file__).resolve().parents[1] / "scripts" / "load_test.py")
load_test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_test)


def test_percentile_interpolates():
    assert load_test.percentile([], 50) is None
    assert load_test.percentile([7], 95) == 7
    assert load_test.percentile([1, 2, 3, 4], 50) == 2.5
    assert load_test.percentile([4, 1, 3, 2], 100) == 4
    assert load_test.percentile(list(range(101)), 95) == 95


def test_summarise_counts_served_and_refused():
    results = [{"ok": True, "tokens": 10, "ttft": 0.01, "total": 0.1, "reason": "length"},
               {"ok": True, "tokens": 20, "ttft": 0.03, "total": 0.3, "reason": "length"},
               {"ok": False, "tokens": 0, "ttft": None, "total": 0.0, "reason": "queue full"}]
    s = load_test.summarise(results, wall=2.0)
    assert (s["requests"], s["served"], s["refused"]) == (3, 2, 1)
    assert s["tokens"] == 30 and s["tokens_per_second"] == 15.0
    assert s["refusal_reasons"] == {"queue full": 1}
    assert s["ttft_ms"]["p50"] == pytest.approx(20.0) and s["total_ms"]["max"] == pytest.approx(300.0)
    assert "refused    1 x queue full" in load_test.format_report(s)


@pytest.fixture
def server():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    engine = build_engine(blocks=64, max_batch_size=4, max_waiting=64, step_seconds=0.002)
    srv = uvicorn.Server(uvicorn.Config(create_app(engine), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not srv.started and time.time() < deadline:
        time.sleep(0.02)
    assert srv.started
    yield f"ws://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(10)


def test_a_real_run_serves_every_client(server):
    s = load_test.run(server, clients=10, concurrency=10, prompt="hello grid", max_new=6)
    assert (s["requests"], s["served"], s["refused"]) == (10, 10, 0)
    assert s["tokens"] == 60
    assert 0 < s["ttft_ms"]["p50"] <= s["total_ms"]["p50"]


def test_refusals_are_counted_not_hidden(server):
    s = load_test.run(server, clients=3, concurrency=3, prompt="word " * 5, max_new=100000)   # over the 512 limit
    assert s["refused"] == 3 and s["served"] == 0 and s["tokens_per_second"] == 0.0
