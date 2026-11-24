import asyncio
import threading
import time

import pytest

from grid.engine import Engine, SimulatedModel
from grid.kvcache import PagedKVCache
from grid.scheduler import Scheduler

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from grid.gateway import build_engine, create_app  # noqa: E402


def make_engine(blocks=32, batch=4, step_seconds=0.0):
    cache = PagedKVCache(2, 2, 4, 4, blocks)
    return Engine(Scheduler(cache, max_batch_size=batch), SimulatedModel(cache, step_seconds))


# ------------------------------------------------------------------ engine subscribers


async def collect(queue, until):
    events = []
    while True:
        event = await asyncio.wait_for(queue.get(), 5)
        events.append(event)
        if until(event):
            return events


async def test_a_subscriber_sees_the_life_of_a_request():
    engine = make_engine()
    await engine.start()
    try:
        queue = engine.subscribe()
        await engine.submit(5, 3).result()
        events = await collect(queue, lambda e: e["type"] == "finished")
    finally:
        await engine.stop()
    assert [e["type"] for e in events] == ["submitted", "step", "step", "step", "finished"]
    first, last = events[1], events[3]
    assert first["prefill"] == ["req-1"] and first["tokens_in"] == 5 and first["ms"] >= 0
    assert last["finished"] == ["req-1"] and last["stats"]["finished"] == 1
    done = events[-1]
    assert (done["id"], done["kind"], done["reason"], done["tokens"]) == ("req-1", "done", "length", 3)
    assert done["latency_steps"] == 3 and all(isinstance(e["t"], float) for e in events)


async def test_rejected_and_cancelled_requests_are_reported():
    engine = make_engine(blocks=16, step_seconds=0.02)       # 16 blocks of 4 tokens = 64 tokens
    await engine.start()
    try:
        queue = engine.subscribe()
        assert (await engine.submit(40, 30).result())["status"] == "rejected"
        job = engine.submit(3, 50)
        await asyncio.sleep(0.05)
        engine.cancel(job)
        await job.result()
        events = await collect(queue, lambda e: e["type"] == "finished")
    finally:
        await engine.stop()
    kinds = [e["type"] for e in events]
    assert kinds[0] == "rejected" and "whole cache" in events[0]["reason"]
    assert events[-1]["kind"] == "cancelled" and events[-1]["id"] == "req-2"


async def test_every_subscriber_gets_every_event_and_unsubscribing_stops_them():
    engine = make_engine()
    await engine.start()
    try:
        a, b = engine.subscribe(), engine.subscribe()
        engine.unsubscribe(b)
        await engine.submit(2, 2).result()
        assert (await collect(a, lambda e: e["type"] == "finished"))[0]["type"] == "submitted"
        assert b.empty()
    finally:
        await engine.stop()


async def test_a_slow_subscriber_loses_old_events_and_never_blocks_the_engine():
    engine = make_engine()
    await engine.start()
    try:
        slow = engine.subscribe(maxsize=3)
        for _ in range(4):
            assert (await engine.submit(2, 3).result())["status"] == "done"   # not slowed down
        assert slow.qsize() == 3
        newest = [slow.get_nowait() for _ in range(3)]
    finally:
        await engine.stop()
    assert newest[-1]["type"] in ("finished", "step")                         # the most recent survive


async def test_subscribers_are_told_when_the_engine_stops():
    engine = make_engine()
    await engine.start()
    queue = engine.subscribe()
    await engine.stop()
    assert (await queue.get())["type"] == "stopped"


# ---------------------------------------------------------------------------- websockets


def client(**kw):
    return TestClient(create_app(build_engine(**kw)))


def receive_until_last(ws):
    events = []
    while True:
        try:
            event = ws.receive_json()
        except Exception:                    # the server closed the socket after its last message
            return events
        events.append(event)


def test_tokens_stream_one_by_one_and_the_socket_closes():
    with client() as c, c.websocket_connect("/v1/stream") as ws:
        ws.send_json({"prompt": "hello streaming grid", "max_new_tokens": 4})
        events = receive_until_last(ws)
    assert [e["type"] for e in events] == ["token"] * 4 + ["done"]
    assert [e["token"] for e in events[:4]] == [1000, 1001, 1002, 1003]
    assert [e["index"] for e in events[:4]] == [0, 1, 2, 3] and events[-1]["reason"] == "length"


@pytest.mark.parametrize("message", [{}, {"prompt": ""}, {"prompt": "x", "max_new_tokens": 0}, {"prompt": 7}])
def test_bad_streaming_requests_get_an_error_message(message):
    with client() as c, c.websocket_connect("/v1/stream") as ws:
        ws.send_json(message)
        event = ws.receive_json()
    assert event["type"] == "error" and event["status"] == 422 and event["detail"]


def test_a_streaming_request_that_never_fits_is_rejected():
    with client(blocks=2) as c, c.websocket_connect("/v1/stream") as ws:
        ws.send_json({"prompt": "word " * 30, "max_new_tokens": 30})
        event = ws.receive_json()
    assert event["type"] == "rejected" and "whole cache" in event["reason"]


def test_a_full_queue_is_reported_on_the_socket():
    with client(max_waiting=0) as c, c.websocket_connect("/v1/stream") as ws:
        ws.send_json({"prompt": "hi", "max_new_tokens": 2})
        event = ws.receive_json()
    assert event["type"] == "error" and event["status"] == 503


def wait_for(condition, seconds=5):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


def test_closing_the_socket_cancels_the_request_and_frees_its_blocks():
    with client(step_seconds=0.02, blocks=64) as c:
        with c.websocket_connect("/v1/stream") as ws:
            ws.send_json({"prompt": "a b c d", "max_new_tokens": 500})
            assert ws.receive_json()["type"] == "token"
        assert wait_for(lambda: c.get("/stats").json()["scheduler"]["aborted"] == 1)
        stats = c.get("/stats").json()
        assert stats["scheduler"]["running"] == 0 and stats["cache"]["blocks_free"] == 64
        assert c.post("/v1/generate", json={"prompt": "ok", "max_new_tokens": 2}).status_code == 200


def test_the_event_feed_starts_with_a_snapshot_and_then_shows_live_steps():
    with client(blocks=64) as c, c.websocket_connect("/ws/events") as feed:
        snapshot = feed.receive_json()
        assert snapshot["type"] == "snapshot" and snapshot["stats"]["blocks_total"] == 64
        assert snapshot["cache"]["blocks_free"] == 64 and snapshot["recent_steps"] == []
        assert c.post("/v1/generate", json={"prompt": "a b c", "max_new_tokens": 3}).status_code == 200
        seen = []
        while not seen or seen[-1]["type"] != "finished":
            seen.append(feed.receive_json())
    assert [e["type"] for e in seen] == ["submitted", "step", "step", "step", "finished"]
    assert seen[1]["prefill"] == ["req-1"] and seen[3]["stats"]["finished"] == 1


def test_two_dashboards_can_watch_at_once_and_leaving_is_clean():
    with client() as c:
        with c.websocket_connect("/ws/events") as one, c.websocket_connect("/ws/events") as two:
            one.receive_json(), two.receive_json()
            c.post("/v1/generate", json={"prompt": "hi", "max_new_tokens": 2})
            assert one.receive_json()["type"] == "submitted" and two.receive_json()["type"] == "submitted"
        assert c.get("/health").status_code == 200                              # the server is fine


# ------------------------------------------------------------- a real server, real sockets


def test_streaming_over_a_real_network_socket():
    websockets = pytest.importorskip("websockets.sync.client")
    import json

    import uvicorn

    server = uvicorn.Server(uvicorn.Config(create_app(build_engine(step_seconds=0.01)),
                                           host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        assert wait_for(lambda: server.started, 10), "the server did not start"
        port = server.servers[0].sockets[0].getsockname()[1]
        stamps = []
        with websockets.connect(f"ws://127.0.0.1:{port}/v1/stream") as ws:
            ws.send(json.dumps({"prompt": "real socket test", "max_new_tokens": 5}))
            events = []
            while True:
                events.append(json.loads(ws.recv(timeout=10)))
                stamps.append(time.perf_counter())
                if events[-1]["type"] == "done":
                    break
        assert [e["type"] for e in events] == ["token"] * 5 + ["done"]
        assert stamps[3] - stamps[0] > 0.015             # tokens arrived over time, not all at the end
    finally:
        server.should_exit = True
        thread.join(10)
