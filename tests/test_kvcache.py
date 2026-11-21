import numpy as np
import pytest

from grid.blocks import ParallelAttention
from grid.kvcache import (
    CacheError,
    OutOfBlocks,
    PagedKVCache,
    blocks_for_budget,
    bytes_per_token,
)
from grid.tp import run_ranks

from test_blocks import build_attention, inputs, make_weights, ref_attention

LAYERS, KV_HEADS, HEAD_DIM = 2, 2, 4


def make_cache(block_size=4, num_blocks=8, layers=LAYERS, kv_heads=KV_HEADS):
    return PagedKVCache(layers, kv_heads, HEAD_DIM, block_size, num_blocks)


def kv(tokens, fill=0.0, kv_heads=KV_HEADS):
    """Keys/values for `tokens` tokens whose values count up from `fill`, so they can be told apart."""
    values = fill + np.arange(tokens, dtype=np.float32)
    return np.broadcast_to(values[None, None, :, None], (1, kv_heads, tokens, HEAD_DIM)).copy()


# ----------------------------------------------------------------- allocation


def test_blocks_are_taken_only_when_the_last_one_is_full():
    cache = make_cache(block_size=4)
    cache.add_sequence("a")
    cache.extend("a", 0, kv(3), kv(3))
    assert cache.blocks_of("a") == [0]            # 3 tokens: one block
    cache.extend("a", 0, kv(1, 3), kv(1, 3))
    assert cache.blocks_of("a") == [0]            # 4 tokens: still one block, exactly full
    cache.extend("a", 0, kv(1, 4), kv(1, 4))
    assert cache.blocks_of("a") == [0, 1]         # the fifth token needs a second block


def test_lowest_free_block_is_handed_out_first_and_freed_blocks_are_reused():
    cache = make_cache(block_size=2, num_blocks=6)
    for seq_id in "abc":
        cache.add_sequence(seq_id)
        cache.extend(seq_id, 0, kv(4), kv(4))     # two blocks each
    assert [cache.blocks_of(s) for s in "abc"] == [[0, 1], [2, 3], [4, 5]]
    assert cache.free_sequence("b") == 2
    cache.add_sequence("d")
    cache.extend("d", 0, kv(3), kv(3))
    assert cache.blocks_of("d") == [2, 3]         # b's old blocks, lowest first
    assert cache.free_blocks == 0


def test_a_full_pool_refuses_cleanly_and_allocates_nothing():
    cache = make_cache(block_size=4, num_blocks=2)
    cache.add_sequence("a")
    cache.extend("a", 0, kv(4), kv(4))
    cache.add_sequence("b")
    with pytest.raises(OutOfBlocks, match="only 1 are free"):
        cache.extend("b", 0, kv(9), kv(9))        # needs 3 blocks
    assert cache.blocks_of("b") == [] and cache.free_blocks == 1
    cache.extend("b", 0, kv(2), kv(2))            # a smaller request still works
    assert cache.blocks_of("b") == [1]


def test_can_fit_counts_what_a_sequence_already_holds():
    cache = make_cache(block_size=4, num_blocks=3)
    cache.add_sequence("a")
    cache.extend("a", 0, kv(6), kv(6))            # 2 blocks, 2 free slots left in the last one
    assert cache.can_fit(2, "a") and cache.can_fit(6, "a")        # 8 tokens: fits in 2 blocks; 12: 3 blocks
    assert not cache.can_fit(7, "a")                              # 13 tokens would need 4 blocks
    assert cache.can_fit(4) and not cache.can_fit(5)              # a new sequence: 1 free block = 4 tokens


def test_same_operations_give_the_same_block_tables_on_every_worker():
    def history(cache):
        for seq_id, n in (("a", 5), ("b", 9), ("c", 2)):
            cache.add_sequence(seq_id)
            cache.extend(seq_id, 0, kv(n), kv(n))
        cache.free_sequence("b")
        cache.add_sequence("d")
        cache.extend("d", 0, kv(7), kv(7))
        return {s: cache.blocks_of(s) for s in cache.sequence_ids()}

    assert history(make_cache(block_size=2, num_blocks=16)) == history(make_cache(block_size=2, num_blocks=16))


# --------------------------------------------------------------- reading back


def test_everything_written_is_read_back_in_order_across_block_boundaries():
    cache = make_cache(block_size=3, num_blocks=8)
    cache.add_sequence("a")
    keys, values = kv(2), kv(2) + 100
    cache.extend("a", 1, keys, values)
    keys2, values2 = kv(5, 2), kv(5, 2) + 100        # crosses two block boundaries
    all_keys, all_values = cache.extend("a", 1, keys2, values2)
    assert all_keys.shape == (1, KV_HEADS, 7, HEAD_DIM)
    np.testing.assert_array_equal(all_keys[0, 0, :, 0], np.arange(7))
    np.testing.assert_array_equal(all_values[0, 1, :, 3], np.arange(7) + 100)


def test_sequences_do_not_disturb_each_other_and_reuse_leaves_no_stale_data():
    cache = make_cache(block_size=2, num_blocks=8)
    cache.add_sequence("a")
    cache.add_sequence("b")
    cache.extend("a", 0, kv(3, 0), kv(3, 0))
    cache.extend("b", 0, kv(3, 50), kv(3, 50))
    a_keys, _ = cache.extend("a", 0, kv(2, 3), kv(2, 3))
    np.testing.assert_array_equal(a_keys[0, 0, :, 0], np.arange(5))
    cache.free_sequence("a")
    cache.add_sequence("c")
    c_keys, _ = cache.extend("c", 0, kv(1, 900), kv(1, 900))     # lands in a's recycled block
    np.testing.assert_array_equal(c_keys[0, 0, :, 0], [900])      # a's old values are not visible
    b_keys, _ = cache.extend("b", 0, kv(1, 53), kv(1, 53))
    np.testing.assert_array_equal(b_keys[0, 0, :, 0], np.arange(50, 54))


def test_layers_are_stored_separately():
    cache = make_cache()
    cache.add_sequence("a")
    cache.extend("a", 0, kv(3, 0), kv(3, 0))
    cache.extend("a", 1, kv(3, 10), kv(3, 10))
    assert cache.layer_length("a", 0) == cache.layer_length("a", 1) == 3
    np.testing.assert_array_equal(cache.view("a", 1).extend(kv(1, 13), kv(1, 13))[0][0, 0, :, 0], np.arange(10, 14))
    assert cache.view("a", 0).length == 3 and cache.view("a", 1).length == 4


def test_a_cache_with_a_smaller_dtype_still_returns_usable_arrays():
    cache = PagedKVCache(1, KV_HEADS, HEAD_DIM, 2, 4, dtype="float16")
    cache.add_sequence("a")
    keys, _ = cache.extend("a", 0, kv(3, 0.5), kv(3, 0.5))
    assert keys.dtype == np.float16
    np.testing.assert_allclose(keys[0, 0, :, 0].astype(np.float32), np.arange(3) + 0.5)


# ------------------------------------------------------------------ accounting


def test_stats_and_block_map_describe_what_is_stored():
    cache = make_cache(block_size=4, num_blocks=6)
    cache.add_sequence("a")
    cache.add_sequence("b")
    cache.extend("a", 0, kv(6), kv(6))
    cache.extend("b", 0, kv(3), kv(3))
    stats = cache.stats()
    assert (stats["blocks_used"], stats["blocks_free"], stats["sequences"]) == (3, 3, 2)
    assert (stats["tokens_stored"], stats["slots_allocated"], stats["wasted_slots"]) == (9, 12, 3)
    assert stats["utilization"] == pytest.approx(0.75)
    assert stats["bytes_per_token"] == 2 * LAYERS * KV_HEADS * HEAD_DIM * 4
    assert stats["bytes_used"] == 3 * 4 * stats["bytes_per_token"]
    assert cache.block_map() == [
        {"seq": "a", "tokens": 4}, {"seq": "a", "tokens": 2}, {"seq": "b", "tokens": 3}, None, None, None]


def test_memory_planning_for_qwen2_5_0_5b():
    per_token = bytes_per_token(layers=24, kv_heads=2, head_dim=64, dtype="bfloat16")
    assert per_token == 12288                                     # 12 KiB per token on one worker
    assert bytes_per_token(24, 1, 64, "bfloat16") == 6144         # half of it per worker with 2 workers
    assert blocks_for_budget(64 * 1024 * 1024, 24, 2, 64, block_size=16, dtype="bfloat16") == 341


# ------------------------------------------------------------------- mistakes


def test_mistakes_are_reported_clearly():
    cache = make_cache()
    cache.add_sequence("a")
    with pytest.raises(CacheError, match="already exists"):
        cache.add_sequence("a")
    with pytest.raises(CacheError, match="unknown sequence"):
        cache.extend("zzz", 0, kv(1), kv(1))
    with pytest.raises(CacheError, match="layer 5"):
        cache.view("a", 5)
    with pytest.raises(CacheError, match="keys must be"):
        cache.extend("a", 0, kv(1, kv_heads=3), kv(1))
    with pytest.raises(CacheError, match="same number of tokens"):
        cache.extend("a", 0, kv(2), kv(3))
    with pytest.raises(ValueError):
        PagedKVCache(1, 1, 1, 0, 4)
    with pytest.raises(ValueError, match="bfloat16"):
        PagedKVCache(1, 1, 1, 1, 1, dtype="bfloat16")


# --------------------------------------------------------- attention + the cache

HEADS, ATT_KV, HD = 8, 2, 4


def cached_attention_outputs(world_size, block_size, plan):
    """Decode with a paged cache. `plan` is a list of (seq_id, start, stop) chunks of two interleaved sequences."""
    w = make_weights(HEADS, ATT_KV, HD)
    x = {"a": inputs(1, 9, seed=11), "b": inputs(1, 6, seed=12)}

    def work(group):
        attention = build_attention(w, group, HD)
        cache = PagedKVCache(1, attention.kv_heads, HD, block_size, 16)
        out = {"a": [], "b": []}
        for seq_id, start, stop in plan:
            if not cache.has_sequence(seq_id):
                cache.add_sequence(seq_id)
            step, _ = attention(x[seq_id][:, start:stop], cache=cache.view(seq_id, 0))
            out[seq_id].append(step)
        return {s: np.concatenate(parts, axis=1) for s, parts in out.items()}, cache.stats()

    expected = {s: ref_attention(w, x[s], HEADS, ATT_KV, HD) for s in x}
    return run_ranks(world_size, work), expected


PLAN = [("a", 0, 5), ("b", 0, 2), ("a", 5, 6), ("b", 2, 3), ("a", 6, 7), ("a", 7, 8), ("b", 3, 6), ("a", 8, 9)]


@pytest.mark.parametrize("world_size,block_size", [(1, 2), (1, 4), (2, 3), (2, 16)])
def test_attention_through_a_paged_cache_matches_the_reference(world_size, block_size):
    results, expected = cached_attention_outputs(world_size, block_size, PLAN)
    for outputs, stats in results:
        for seq_id in "ab":
            np.testing.assert_allclose(outputs[seq_id], expected[seq_id], rtol=1e-3, atol=1e-4)
        assert stats["tokens_stored"] == 9 + 6
        # each worker caches only its own half of the key/value heads
        assert stats["bytes_per_token"] == 2 * 1 * (ATT_KV // world_size) * HD * 4


def test_attention_refuses_a_batch_and_both_past_and_cache():
    w = make_weights(HEADS, ATT_KV, HD)
    attention = build_attention(w, None, HD)
    cache = PagedKVCache(1, ATT_KV, HD, 4, 4)
    cache.add_sequence("a")
    with pytest.raises(ValueError, match="batch must be 1"):
        attention(inputs(2, 3), cache=cache.view("a", 0))
    _, past = attention(inputs(1, 2))
    with pytest.raises(ValueError, match="either past or cache"):
        attention(inputs(1, 1), past=past, cache=cache.view("a", 0))


def test_torch_backend_cache_with_torch_attention():
    torch = pytest.importorskip("torch")
    w = make_weights(HEADS, ATT_KV, HD)
    x = inputs(1, 7, seed=3)
    expected = ref_attention(w, x, HEADS, ATT_KV, HD)
    tw = {k: torch.from_numpy(v) for k, v in w.items()}
    attention = ParallelAttention.from_full(
        tw["q"], tw["k"], tw["v"], tw["o"], head_dim=HD,
        q_bias=tw["qb"], k_bias=tw["kb"], v_bias=tw["vb"], rope_theta=1e6)
    cache = PagedKVCache(1, ATT_KV, HD, 3, 8, dtype="float32", backend="torch")
    cache.add_sequence("a")
    pieces = [attention(torch.from_numpy(x[:, s:e]), cache=cache.view("a", 0))[0] for s, e in ((0, 4), (4, 5), (5, 7))]
    np.testing.assert_allclose(torch.cat(pieces, dim=1).numpy(), expected, rtol=1e-3, atol=1e-4)
    half = PagedKVCache(1, ATT_KV, HD, 3, 8, dtype="float16", backend="torch")
    half.add_sequence("a")
    out, _ = attention(torch.from_numpy(x), cache=half.view("a", 0))   # float16 storage, float32 math
    np.testing.assert_allclose(out.numpy(), expected, rtol=2e-2, atol=2e-2)
