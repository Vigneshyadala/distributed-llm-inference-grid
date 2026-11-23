import asyncio

import pytest

from grid.engine import Engine, QueueFull, SimulatedModel
from grid.kvcache import PagedKVCache
from grid.scheduler import Scheduler


def make_engine(blocks=32, block_size=4, batch=4, step_seconds=0.0, max_waiting=64, execute=None, **kw):
    cache = PagedKVCache(2, 2, 4, block_size, blocks)
    scheduler = Scheduler(cache, max_batch_size=batch, **kw)
    return Engine(scheduler, execute or SimulatedModel(cache, step_seconds), max_waiting=max_waiting)


@pytest.fixture
async def running():
    engines = []

    async def start(**kw):
        engine = make_engine(**kw)
        await engine.start()
        engines.append(engine)
        return engine

    yield start
    for engine in engines:
        await engine.stop()


async def test_one_request_gets_its_tokens_and_the_cache_is_clean_afterwards(running):
    engine = await running()
    result = await engine.submit(5, 4).result()
    assert result == {"status": "done", "tokens": [1000, 1001, 1002, 1003], "reason": "length"}
    cache = engine.scheduler.cache
    assert cache.free_blocks == cache.num_blocks and cache.sequence_ids() == []


async def test_tokens_stream_in_order_with_indexes(running):
    engine = await running()
    events = [e async for e in engine.submit(3, 3).events()]
    assert [e["type"] for e in events] == ["token", "token", "token", "done"]
    assert [e["index"] for e in events[:3]] == [0, 1, 2]


async def test_concurrent_requests_share_batches(running):
    engine = await running(step_seconds=0.01)
    jobs = [engine.submit(4, n) for n in (3, 6, 2)]
    results = await asyncio.gather(*(j.result() for j in jobs))
    assert [len(r["tokens"]) for r in results] == [3, 6, 2]
    assert all(r["status"] == "done" for r in results)
    assert max(t["running"] for t in engine.scheduler.timeline) == 3     # they ran together
    assert len(engine.scheduler.timeline) < 3 + 6 + 2                      # fewer steps than one at a time


async def test_a_request_that_can_never_fit_is_rejected(running):
    engine = await running(blocks=2)
    result = await engine.submit(20, 20).result()
    assert result["status"] == "rejected" and "whole cache" in result["reason"]
    assert (await engine.submit(2, 2).result())["status"] == "done"        # the engine is unharmed


async def test_a_full_queue_refuses_new_requests(running):
    engine = await running(batch=1, step_seconds=0.05, max_waiting=1)
    first = engine.submit(2, 5)
    await asyncio.sleep(0.02)                                              # first is now running
    second = engine.submit(2, 2)                                           # waits
    with pytest.raises(QueueFull):
        engine.submit(2, 2)
    assert (await first.result())["status"] == "done"
    assert (await second.result())["status"] == "done"


async def test_invalid_sizes_are_refused(running):
    engine = await running()
    with pytest.raises(ValueError):
        engine.submit(0, 5)
    with pytest.raises(ValueError):
        engine.submit(5, 0)


async def test_submit_needs_a_running_engine():
    engine = make_engine()
    with pytest.raises(RuntimeError, match="not running"):
        engine.submit(2, 2)


async def test_cancel_frees_the_blocks_and_the_others_carry_on(running):
    engine = await running(step_seconds=0.02)
    keep = engine.submit(3, 6)
    drop = engine.submit(3, 60)
    await asyncio.sleep(0.05)
    engine.cancel(drop)
    assert (await drop.result())["status"] == "cancelled"
    assert (await keep.result())["status"] == "done"
    cache = engine.scheduler.cache
    assert cache.free_blocks == cache.num_blocks


async def test_a_model_error_fails_only_that_step_and_the_engine_recovers(running):
    calls = {"n": 0}

    def flaky(plan):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return {r.id: 1 for r in plan.requests}

    engine = await running(execute=flaky)
    bad = await engine.submit(3, 3).result()
    assert bad["status"] == "error" and "boom" in bad["reason"]
    good = await engine.submit(3, 3).result()
    assert good["status"] == "done" and len(good["tokens"]) == 3
    cache = engine.scheduler.cache
    assert cache.free_blocks == cache.num_blocks


async def test_stopping_tells_open_jobs(running):
    engine = await running(step_seconds=0.05)
    job = engine.submit(3, 100)
    await asyncio.sleep(0.03)
    await engine.stop()
    result = await job.result()
    assert result["status"] == "cancelled" and result["reason"] == "engine stopped"
