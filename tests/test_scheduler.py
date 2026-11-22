import random

import numpy as np
import pytest

from grid.kvcache import OutOfBlocks, PagedKVCache
from grid.scheduler import (ABORTED, FINISHED, REJECTED, RUNNING, WAITING, Request, Scheduler,
                            SchedulerError, run_to_completion)

KV_HEADS, HEAD_DIM = 1, 2


def make_cache(num_blocks, block_size=4, layers=2):
    return PagedKVCache(layers, KV_HEADS, HEAD_DIM, block_size, num_blocks)


def execute(cache, token=7, tokens=None):
    """A fake model: stores keys and values like the real one would and answers `token`."""
    def run(plan):
        out = {}
        for request in plan.prefill + plan.decode:
            count = request.prompt_len if request in plan.prefill else 1
            for layer in range(cache.layers):
                data = np.zeros((1, KV_HEADS, count, HEAD_DIM), dtype=np.float32)
                cache.extend(request.id, layer, data, data)
            out[request.id] = tokens(request) if tokens else token
        return out
    return run


def test_request_validates_sizes():
    with pytest.raises(ValueError):
        Request("a", 0, 3)
    with pytest.raises(ValueError):
        Request("a", 3, -1)
    with pytest.raises(ValueError):
        Request("a", True, 3)
    assert Request("a", 3, 4).total_tokens == 7


def test_scheduler_validates_settings():
    cache = make_cache(4)
    with pytest.raises(ValueError):
        Scheduler(cache, max_batch_size=0)
    with pytest.raises(ValueError):
        Scheduler(cache, max_prefill_tokens=0)


def test_one_request_prefills_then_decodes_then_finishes():
    cache = make_cache(8)
    scheduler = Scheduler(cache)
    request = scheduler.submit(Request("a", prompt_len=5, max_new_tokens=3))
    run = execute(cache)

    plan = scheduler.next_step()
    assert [r.id for r in plan.prefill] == ["a"] and plan.decode == [] and plan.tokens_in == 5
    assert request.state == RUNNING
    scheduler.complete_step(plan, run(plan))

    plan = scheduler.next_step()
    assert plan.prefill == [] and [r.id for r in plan.decode] == ["a"] and plan.tokens_in == 1
    scheduler.complete_step(plan, run(plan))

    plan = scheduler.next_step()
    done = scheduler.complete_step(plan, run(plan))
    assert done == [request] and request.state == FINISHED and request.reason == "length"
    assert request.generated == 3 and request.latency_steps == 3 and request.wait_steps == 0
    assert scheduler.next_step() is None and scheduler.idle
    assert cache.free_blocks == cache.num_blocks and not cache.has_sequence("a")


def test_finished_request_makes_room_in_the_same_batch_for_a_waiting_one():
    cache = make_cache(16)
    scheduler = Scheduler(cache, max_batch_size=2)
    for name, new in (("a", 2), ("b", 6), ("c", 3)):
        scheduler.submit(Request(name, 4, new))
    run_to_completion(scheduler, execute(cache))
    by = {r.id: r for r in scheduler.finished}
    assert by["a"].finished_step == 1
    assert by["c"].admitted_step == 2          # took a's place on the very next step
    assert by["b"].finished_step == 5
    assert max(t["running"] for t in scheduler.timeline) <= 2


def test_batch_size_limit():
    cache = make_cache(64)
    scheduler = Scheduler(cache, max_batch_size=3)
    for i in range(5):
        scheduler.submit(Request(i, 2, 4))
    plan = scheduler.next_step()
    assert [r.id for r in plan.prefill] == [0, 1, 2]
    assert [r.id for r in scheduler.waiting] == [3, 4]


def test_admission_reserves_room_for_the_whole_reply():
    cache = make_cache(num_blocks=4, block_size=4)           # 16 tokens in all
    scheduler = Scheduler(cache)
    scheduler.submit(Request("a", 4, 4))                      # reserves 2 blocks
    scheduler.submit(Request("b", 4, 4))                      # reserves 2 blocks
    scheduler.submit(Request("c", 1, 1))                      # would fit right now, but must wait its turn
    plan = scheduler.next_step()
    assert [r.id for r in plan.prefill] == ["a", "b"]
    assert scheduler.reserved_blocks == 4
    assert [r.id for r in scheduler.waiting] == ["c"]


def test_running_requests_never_run_out_of_blocks():
    cache = make_cache(num_blocks=5, block_size=4)
    scheduler = Scheduler(cache, max_batch_size=4)
    for i in range(6):
        scheduler.submit(Request(i, 6, 5))                    # each reserves 3 blocks, so one at a time
    run_to_completion(scheduler, execute(cache))              # an OutOfBlocks would raise here
    assert len(scheduler.finished) == 6 and cache.free_blocks == 5
    assert max(t["running"] for t in scheduler.timeline) == 1


def test_first_come_first_served_a_big_request_is_not_overtaken():
    cache = make_cache(num_blocks=4, block_size=4)
    scheduler = Scheduler(cache)
    scheduler.submit(Request("small1", 2, 2))                 # 1 block
    scheduler.submit(Request("big", 8, 8))                    # 4 blocks, cannot start beside small1
    scheduler.submit(Request("small2", 2, 2))                 # would fit, but must not jump the queue
    plan = scheduler.next_step()
    assert [r.id for r in plan.prefill] == ["small1"]
    assert [r.id for r in scheduler.waiting] == ["big", "small2"]
    scheduler.complete_step(plan, execute(cache)(plan))
    run_to_completion(scheduler, execute(cache))
    order = [r.id for r in sorted(scheduler.finished, key=lambda r: r.admitted_step)]
    assert order == ["small1", "big", "small2"]


def test_request_that_can_never_fit_is_rejected_at_once():
    cache = make_cache(num_blocks=2, block_size=4)
    scheduler = Scheduler(cache)
    request = scheduler.submit(Request("huge", 6, 6))
    assert request.state == REJECTED and "whole cache has 2" in request.reason
    assert scheduler.rejected == [request] and scheduler.waiting == []
    assert scheduler.next_step() is None
    ok = scheduler.submit(Request("ok", 2, 2))
    assert ok.state == WAITING


def test_prefill_token_budget_spreads_long_prompts_over_steps():
    cache = make_cache(64)
    scheduler = Scheduler(cache, max_prefill_tokens=10)
    for name in "abc":
        scheduler.submit(Request(name, 6, 3))
    first = scheduler.next_step()
    assert [r.id for r in first.prefill] == ["a"]             # a second 6-token prompt would pass 10
    scheduler.complete_step(first, execute(cache)(first))
    second = scheduler.next_step()
    assert [r.id for r in second.prefill] == ["b"] and [r.id for r in second.decode] == ["a"]


def test_a_prompt_longer_than_the_budget_still_runs_when_alone():
    cache = make_cache(64)
    scheduler = Scheduler(cache, max_prefill_tokens=4)
    scheduler.submit(Request("long", 20, 2))
    assert [r.id for r in scheduler.next_step().prefill] == ["long"]


def test_end_of_sequence_token_finishes_early():
    cache = make_cache(16)
    scheduler = Scheduler(cache, eos_token=0)
    scheduler.submit(Request("a", 3, 50))
    scheduler.submit(Request("b", 3, 4))
    run_to_completion(scheduler, execute(cache, tokens=lambda r: 0 if (r.id == "a" and r.generated == 2) else 7))
    a, b = scheduler.get("a"), scheduler.get("b")
    assert a.reason == "eos" and a.generated == 3 and b.reason == "length" and b.generated == 4
    assert cache.free_blocks == cache.num_blocks


def test_abort_waiting_and_running_requests():
    cache = make_cache(8)
    scheduler = Scheduler(cache, max_batch_size=1)
    scheduler.submit(Request("a", 4, 8))
    scheduler.submit(Request("b", 4, 8))
    plan = scheduler.next_step()
    scheduler.complete_step(plan, execute(cache)(plan))
    assert scheduler.abort("b") is True                       # was waiting
    assert scheduler.abort("a", "client left") is True        # was running: its blocks come back
    assert scheduler.get("a").state == ABORTED and scheduler.get("a").reason == "client left"
    assert cache.free_blocks == cache.num_blocks and scheduler.idle
    assert scheduler.abort("a") is False and scheduler.abort("nobody") is False


def test_steps_must_be_completed_in_order():
    cache = make_cache(8)
    scheduler = Scheduler(cache)
    scheduler.submit(Request("a", 2, 2))
    plan = scheduler.next_step()
    with pytest.raises(SchedulerError, match="has not been completed"):
        scheduler.next_step()
    with pytest.raises(SchedulerError, match="has not been completed"):
        scheduler.abort("a")
    with pytest.raises(SchedulerError, match="exactly the planned"):
        scheduler.complete_step(plan, {})
    with pytest.raises(SchedulerError, match="exactly the planned"):
        scheduler.complete_step(plan, {"a": 1, "zzz": 1})
    with pytest.raises(SchedulerError, match="last"):
        scheduler.complete_step(object(), {"a": 1})


def test_duplicate_and_unknown_requests():
    scheduler = Scheduler(make_cache(8))
    scheduler.submit(Request("a", 2, 2))
    with pytest.raises(SchedulerError, match="already"):
        scheduler.submit(Request("a", 2, 2))
    with pytest.raises(SchedulerError, match="unknown"):
        scheduler.get("zzz")


def test_arrivals_during_the_run_join_the_batch():
    cache = make_cache(32)
    scheduler = Scheduler(cache, max_batch_size=4)
    arrivals = {0: [Request("a", 3, 6)], 3: [Request("late", 3, 2)]}
    run_to_completion(scheduler, execute(cache), arrivals)
    late = scheduler.get("late")
    assert late.arrival_step == 3 and late.admitted_step == 3 and late.wait_steps == 0
    record = next(t for t in scheduler.timeline if t["step"] == 3)
    assert record["prefill"] == ["late"] and record["decode"] == ["a"]


def test_idle_gap_before_a_late_arrival():
    cache = make_cache(8)
    scheduler = Scheduler(cache)
    run_to_completion(scheduler, execute(cache), {5: [Request("a", 2, 1)]})
    assert scheduler.get("a").state == FINISHED and scheduler.get("a").admitted_step == 5


def test_stats_and_timeline_describe_each_step():
    cache = make_cache(8)
    scheduler = Scheduler(cache, max_batch_size=2)
    scheduler.submit(Request("a", 4, 2))
    scheduler.submit(Request("b", 4, 2))
    scheduler.submit(Request("c", 4, 2))
    plan = scheduler.next_step()
    stats = scheduler.stats()
    assert (stats["waiting"], stats["running"], stats["reserved_blocks"], stats["blocks_total"]) == (1, 2, 4, 8)
    scheduler.complete_step(plan, execute(cache)(plan))
    record = scheduler.timeline[0]
    assert record["prefill"] == ["a", "b"] and record["decode"] == [] and record["tokens_in"] == 8
    assert record["running"] == 2 and record["waiting"] == 1 and record["blocks_used"] == 2


def test_random_workloads_always_finish_without_leaks():
    for seed in range(25):
        rng = random.Random(seed)
        cache = make_cache(num_blocks=rng.randint(6, 20), block_size=rng.choice([2, 4, 8]))
        scheduler = Scheduler(cache, max_batch_size=rng.randint(1, 6),
                              max_prefill_tokens=rng.choice([None, 8, 30]))
        arrivals = {}
        for i in range(rng.randint(5, 25)):
            arrivals.setdefault(rng.randint(0, 15), []).append(
                Request(i, rng.randint(1, 12), rng.randint(1, 10)))
        total = sum(len(v) for v in arrivals.values())
        try:
            run_to_completion(scheduler, execute(cache), arrivals)
        except OutOfBlocks:                                   # the reservation promises this never happens
            pytest.fail(f"seed {seed}: a running request ran out of blocks")
        assert len(scheduler.finished) + len(scheduler.rejected) == total
        assert cache.free_blocks == cache.num_blocks and cache.sequence_ids() == []
        assert all(t["blocks_used"] <= cache.num_blocks and t["running"] <= scheduler.max_batch_size
                   for t in scheduler.timeline)
