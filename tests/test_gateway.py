import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from grid.gateway import build_engine, create_app  # noqa: E402


def client(**kw):
    return TestClient(create_app(build_engine(**kw)))


def test_health():
    with client() as c:
        assert c.get("/health").json() == {"status": "ok"}


def test_generate_returns_tokens_and_timing():
    with client() as c:
        reply = c.post("/v1/generate", json={"prompt": "hello distributed grid", "max_new_tokens": 4})
        assert reply.status_code == 200
        body = reply.json()
        assert body["tokens"] == [1000, 1001, 1002, 1003] and body["finish_reason"] == "length"
        assert body["prompt_tokens"] == 3 and body["latency_steps"] == 4 and body["wait_steps"] == 0


def test_default_length_is_used_when_not_given():
    with client() as c:
        assert len(c.post("/v1/generate", json={"prompt": "hi"}).json()["tokens"]) == 32


@pytest.mark.parametrize("body", [
    {}, {"prompt": ""}, {"prompt": "x", "max_new_tokens": 0},
    {"prompt": "x", "max_new_tokens": 100000}, {"prompt": 5}, {"prompt": "x", "max_new_tokens": "many"},
])
def test_bad_bodies_are_422(body):
    with client() as c:
        assert c.post("/v1/generate", json=body).status_code == 422


def test_a_request_too_big_for_the_cache_is_413():
    with client(blocks=2, block_size=4) as c:
        reply = c.post("/v1/generate", json={"prompt": "word " * 20, "max_new_tokens": 20})
        assert reply.status_code == 413 and "whole cache" in reply.json()["detail"]
        assert c.post("/v1/generate", json={"prompt": "ok", "max_new_tokens": 2}).status_code == 200


def test_stats_describe_the_scheduler_and_the_cache():
    with client(blocks=64) as c:
        c.post("/v1/generate", json={"prompt": "a b c", "max_new_tokens": 3})
        stats = c.get("/stats").json()
        assert stats["scheduler"]["finished"] == 1 and stats["scheduler"]["blocks_total"] == 64
        assert stats["cache"]["blocks_free"] == 64 and stats["cache"]["sequences"] == 0
        steps = stats["recent_steps"]
        assert [s["step"] for s in steps] == [0, 1, 2] and steps[0]["prefill"] == ["req-1"]


def test_many_clients_at_once_are_all_served():
    from concurrent.futures import ThreadPoolExecutor

    with client(step_seconds=0.005) as c:
        def call(i):
            return c.post("/v1/generate", json={"prompt": "w " * (i + 1), "max_new_tokens": 3 + i % 4})

        with ThreadPoolExecutor(8) as pool:
            replies = list(pool.map(call, range(12)))
        assert all(r.status_code == 200 for r in replies)
        assert [len(r.json()["tokens"]) for r in replies] == [3 + i % 4 for i in range(12)]
        stats = c.get("/stats").json()
        assert stats["scheduler"]["finished"] == 12 and stats["cache"]["blocks_used"] == 0
        assert max(s["running"] for s in stats["recent_steps"]) > 1
