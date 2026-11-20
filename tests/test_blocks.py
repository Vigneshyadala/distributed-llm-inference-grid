import numpy as np
import pytest

from grid.blocks import ParallelAttention, ParallelMLP, apply_rotary, causal_mask, rotary_tables
from grid.tinymodel import write_tiny_checkpoint
from grid.tp import run_ranks
from grid.weights import Checkpoint, ShardError

HIDDEN = 24


def make_weights(heads, kv_heads, head_dim, inter=32, seed=0, hidden=HIDDEN):
    rng = np.random.default_rng(seed)

    def rand(*shape):
        return (rng.standard_normal(shape) / np.sqrt(shape[-1])).astype(np.float32)

    q_rows, kv_rows = heads * head_dim, kv_heads * head_dim
    return {
        "q": rand(q_rows, hidden), "qb": rand(q_rows),
        "k": rand(kv_rows, hidden), "kb": rand(kv_rows),
        "v": rand(kv_rows, hidden), "vb": rand(kv_rows),
        "o": rand(hidden, q_rows),
        "gate": rand(inter, hidden), "up": rand(inter, hidden), "down": rand(hidden, inter),
    }


# ----- plain, loop-based references written differently from the code under test -----


def ref_rotate(vec, position, theta):
    """Rotary embedding of one head vector, as complex numbers: pair i is (x[i], x[i + half])."""
    half = len(vec) // 2
    freqs = 1.0 / theta ** (np.arange(half) * 2.0 / len(vec))
    z = (vec[:half] + 1j * vec[half:]) * np.exp(1j * position * freqs)
    return np.concatenate([z.real, z.imag])


def ref_attention(w, x, heads, kv_heads, head_dim, theta=1e6):
    batch, tokens, _ = x.shape
    q = x @ w["q"].T + w["qb"]
    k = x @ w["k"].T + w["kb"]
    v = x @ w["v"].T + w["vb"]
    out = np.zeros((batch, tokens, heads * head_dim))
    for b in range(batch):
        for h in range(heads):
            kv = h // (heads // kv_heads)
            qs = slice(h * head_dim, (h + 1) * head_dim)
            ks = slice(kv * head_dim, (kv + 1) * head_dim)
            for t in range(tokens):
                qt = ref_rotate(q[b, t, qs], t, theta)
                scores = np.array([
                    qt @ ref_rotate(k[b, s, ks], s, theta) for s in range(t + 1)
                ]) / np.sqrt(head_dim)
                weights = np.exp(scores - scores.max())
                weights /= weights.sum()
                out[b, t, qs] = sum(wt * v[b, s, ks] for s, wt in enumerate(weights))
    return out @ w["o"].T


def ref_mlp(w, x):
    gate = x @ w["gate"].T
    return ((gate / (1 + np.exp(-gate))) * (x @ w["up"].T)) @ w["down"].T


def build_attention(w, group, head_dim):
    return ParallelAttention.from_full(
        w["q"], w["k"], w["v"], w["o"], head_dim=head_dim, group=group,
        q_bias=w["qb"], k_bias=w["kb"], v_bias=w["vb"], rope_theta=1e6,
    )


def inputs(batch=2, tokens=6, seed=5):
    return np.random.default_rng(seed).standard_normal((batch, tokens, HIDDEN)).astype(np.float32)


# ------------------------------------------------------------------------------ tests


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_sharded_mlp_matches_the_reference(world_size):
    w, x = make_weights(4, 2, 4), inputs()
    expected = ref_mlp(w, x)

    def work(group):
        return ParallelMLP.from_full(w["gate"], w["up"], w["down"], group)(x)

    for result in run_ranks(world_size, work):
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("heads,kv_heads,head_dim,world_size", [
    (4, 4, 4, 1), (4, 4, 4, 2), (4, 4, 4, 4),      # every head has its own key/value head
    (8, 2, 4, 1), (8, 2, 4, 2),                     # grouped-query: 4 query heads per kv head
    (6, 2, 8, 2),                                   # 3 query heads per kv head, one kv head per worker
    (8, 4, 4, 4),                                   # one kv head per worker, two query heads each
])
def test_sharded_attention_matches_the_reference(heads, kv_heads, head_dim, world_size):
    w, x = make_weights(heads, kv_heads, head_dim), inputs()
    expected = ref_attention(w, x, heads, kv_heads, head_dim)

    def work(group):
        out, _ = build_attention(w, group, head_dim)(x)
        return out

    for result in run_ranks(world_size, work):
        np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("world_size", [1, 2])
def test_decoding_one_token_at_a_time_gives_the_same_result(world_size):
    heads, kv_heads, hd = 8, 2, 4
    w, x = make_weights(heads, kv_heads, hd), inputs(batch=1, tokens=7)
    expected = ref_attention(w, x, heads, kv_heads, hd)

    def work(group):
        attention = build_attention(w, group, hd)
        out, past = attention(x[:, :4])                 # prefill four tokens
        pieces = [out]
        for t in range(4, 7):                           # then decode three, one at a time
            step, past = attention(x[:, t:t + 1], past)
            pieces.append(step)
        return np.concatenate(pieces, axis=1), past

    for result, past in run_ranks(world_size, work):
        np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-4)
        assert past[0].shape == (1, kv_heads // world_size, 7, hd)   # this worker's share of the cache


def test_a_token_never_sees_later_tokens():
    heads, kv_heads, hd = 4, 2, 4
    w, x = make_weights(heads, kv_heads, hd), inputs(batch=1, tokens=6)
    changed = x.copy()
    changed[:, 4:] += 5.0
    attention = build_attention(w, None, hd)
    np.testing.assert_allclose(attention(x)[0][:, :4], attention(changed)[0][:, :4], rtol=1e-5, atol=1e-6)
    assert not np.allclose(attention(x)[0][:, 5:], attention(changed)[0][:, 5:])


def test_each_block_does_exactly_one_all_reduce():
    heads, kv_heads, hd = 4, 2, 4
    w, x = make_weights(heads, kv_heads, hd), inputs()

    def work(group):
        attention = build_attention(w, group, hd)
        mlp = ParallelMLP.from_full(w["gate"], w["up"], w["down"], group)
        attention(x)
        after_attention = group.reduce_calls
        mlp(x)
        return after_attention, group.reduce_calls

    for after_attention, after_mlp in run_ranks(2, work):
        assert (after_attention, after_mlp) == (1, 2)


def test_each_worker_holds_whole_heads_only():
    w = make_weights(8, 2, 4)
    from grid.tp import thread_groups

    attention = build_attention(w, thread_groups(2)[1], 4)
    assert (attention.heads, attention.kv_heads) == (4, 1)
    assert attention.q.weight.shape == (16, HIDDEN) and attention.k.weight.shape == (4, HIDDEN)
    assert attention.o.weight.shape == (HIDDEN, 16)


def test_too_few_key_value_heads_for_the_workers_is_refused():
    from grid.tp import thread_groups

    w = make_weights(8, 2, 4)
    with pytest.raises(ShardError):
        build_attention(w, thread_groups(4)[0], 4)   # 2 kv heads cannot go to 4 workers


def test_wrong_input_shape_is_refused():
    attention = build_attention(make_weights(4, 2, 4), None, 4)
    with pytest.raises(ValueError, match="batch, tokens, hidden"):
        attention(np.zeros((5, HIDDEN), np.float32))


def test_rotary_keeps_vector_length_and_position_zero_is_the_identity():
    x = np.random.default_rng(1).standard_normal((1, 2, 3, 8)).astype(np.float32)
    cos, sin = rotary_tables(np.arange(3), 8, 10000.0)
    rotated = apply_rotary(x, cos, sin)
    np.testing.assert_allclose(rotated[:, :, 0], x[:, :, 0], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(rotated, axis=-1), np.linalg.norm(x, axis=-1), rtol=1e-5)
    with pytest.raises(ValueError):
        rotary_tables(np.arange(3), 7, 10000.0)


def test_causal_mask_shape_and_offset():
    mask = causal_mask(2, 5)   # two new tokens after three old ones
    assert mask.shape == (2, 5)
    assert (mask[0, :4] == 0).all() and mask[0, 4] == -np.inf
    assert (mask[1] == 0).all()


def test_blocks_can_be_loaded_straight_from_a_checkpoint(tmp_path):
    config, tensors = write_tiny_checkpoint(tmp_path)
    hd, hidden = config["head_dim"], config["hidden"]
    x = np.random.default_rng(2).standard_normal((1, 5, hidden)).astype(np.float32)
    attn_prefix, mlp_prefix = "model.layers.1.self_attn.", "model.layers.1.mlp."
    full = {
        "q": tensors[attn_prefix + "q_proj.weight"], "qb": tensors[attn_prefix + "q_proj.bias"],
        "k": tensors[attn_prefix + "k_proj.weight"], "kb": tensors[attn_prefix + "k_proj.bias"],
        "v": tensors[attn_prefix + "v_proj.weight"], "vb": tensors[attn_prefix + "v_proj.bias"],
        "o": tensors[attn_prefix + "o_proj.weight"],
        "gate": tensors[mlp_prefix + "gate_proj.weight"], "up": tensors[mlp_prefix + "up_proj.weight"],
        "down": tensors[mlp_prefix + "down_proj.weight"],
    }
    expected_attention = ref_attention(full, x, config["heads"], config["kv_heads"], hd)
    expected_mlp = ref_mlp(full, x)

    with Checkpoint(tmp_path, framework="np") as ckpt:
        def work(group):
            attention = ParallelAttention.from_checkpoint(ckpt, attn_prefix, head_dim=hd, group=group)
            mlp = ParallelMLP.from_checkpoint(ckpt, mlp_prefix, group)
            return attention(x)[0], mlp(x)

        # the tiny checkpoint's weights are not scaled, so values reach the hundreds; float32 rounding
        # then differs between the sharded and the full sum by a few millionths of the largest value
        attention_atol = 2e-6 * np.abs(expected_attention).max()
        mlp_atol = 2e-6 * np.abs(expected_mlp).max()
        for attention_out, mlp_out in run_ranks(2, work):
            np.testing.assert_allclose(attention_out, expected_attention, rtol=1e-3, atol=attention_atol)
            np.testing.assert_allclose(mlp_out, expected_mlp, rtol=1e-4, atol=mlp_atol)


def test_torch_tensors_work_in_the_blocks():
    torch = pytest.importorskip("torch")
    heads, kv_heads, hd = 8, 2, 4
    w, x = make_weights(heads, kv_heads, hd), inputs(batch=1, tokens=5)
    expected = ref_attention(w, x, heads, kv_heads, hd)
    tw = {name: torch.from_numpy(value) for name, value in w.items()}
    attention = ParallelAttention.from_full(
        tw["q"], tw["k"], tw["v"], tw["o"], head_dim=hd,
        q_bias=tw["qb"], k_bias=tw["kb"], v_bias=tw["vb"], rope_theta=1e6,
    )
    out, past = attention(torch.from_numpy(x[:, :4]))
    step, _ = attention(torch.from_numpy(x[:, 4:]), past)
    np.testing.assert_allclose(torch.cat([out, step], dim=1).numpy(), expected, rtol=1e-3, atol=1e-4)
    mlp = ParallelMLP.from_full(tw["gate"], tw["up"], tw["down"])
    np.testing.assert_allclose(mlp(torch.from_numpy(x)).numpy(), ref_mlp(w, x), rtol=1e-4, atol=1e-4)
