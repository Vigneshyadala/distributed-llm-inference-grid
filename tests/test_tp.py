import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from grid.tinymodel import write_tiny_checkpoint
from grid.tp import (
    ColumnParallelLinear,
    RowParallelLinear,
    SoloGroup,
    run_ranks,
)
from grid.weights import Checkpoint, ShardError

ROOT = Path(__file__).resolve().parents[1]
HIDDEN, INTER, TOKENS = 16, 32, 5


def relu(x):
    return x * (x > 0)


def make(seed=0):
    rng = np.random.default_rng(seed)

    def rand(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    return {
        "w1": rand(INTER, HIDDEN), "b1": rand(INTER),
        "w2": rand(HIDDEN, INTER), "b2": rand(HIDDEN),
        "x": rand(TOKENS, HIDDEN), "h": rand(TOKENS, INTER),
    }


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_column_parallel_slices_join_into_the_full_output(world_size):
    d = make()
    full = d["x"] @ d["w1"].T + d["b1"]
    pieces = [
        ColumnParallelLinear.from_full(d["w1"], d["b1"], rank=r, world_size=world_size)(d["x"])
        for r in range(world_size)
    ]
    assert all(p.shape == (TOKENS, INTER // world_size) for p in pieces)
    np.testing.assert_allclose(np.concatenate(pieces, axis=-1), full, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_row_parallel_sums_the_partials_on_every_worker(world_size):
    d = make()
    full = d["h"] @ d["w2"].T + d["b2"]
    width = INTER // world_size

    def work(group):
        layer = RowParallelLinear.from_full(d["w2"], d["b2"], group=group)
        return layer(d["h"][:, group.rank * width:(group.rank + 1) * width])

    for result in run_ranks(world_size, work):
        np.testing.assert_allclose(result, full, rtol=1e-4, atol=1e-4)


def test_row_parallel_bias_is_added_once_not_once_per_worker():
    d = make()
    zero_input = np.zeros((TOKENS, INTER // 2), dtype=np.float32)

    def work(group):
        return RowParallelLinear.from_full(d["w2"], d["b2"], group=group)(zero_input)

    for result in run_ranks(2, work):
        np.testing.assert_allclose(result, np.tile(d["b2"], (TOKENS, 1)))


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_column_then_row_equals_the_full_two_layer_network(world_size):
    d = make()
    full = relu(d["x"] @ d["w1"].T + d["b1"]) @ d["w2"].T + d["b2"]

    def work(group):
        up = ColumnParallelLinear.from_full(d["w1"], d["b1"], rank=group.rank, world_size=group.world_size)
        down = RowParallelLinear.from_full(d["w2"], d["b2"], group=group)
        return down(relu(up(d["x"])))

    for result in run_ranks(world_size, work):
        np.testing.assert_allclose(result, full, rtol=1e-4, atol=1e-4)


def test_each_worker_holds_only_its_share_of_the_weights():
    d = make()
    up = ColumnParallelLinear.from_full(d["w1"], rank=1, world_size=4)
    down = RowParallelLinear.from_full(d["w2"], group=run_ranks(4, lambda g: g)[1])
    assert up.weight.shape == (INTER // 4, HIDDEN)
    assert down.weight.shape == (HIDDEN, INTER // 4)
    assert up.weight.base is None and down.weight.base is None  # copies, not views of the full weight


def test_batches_with_extra_leading_dimensions_work():
    d = make()
    x = np.stack([d["x"], d["x"] * 2])
    layer = ColumnParallelLinear.from_full(d["w1"], d["b1"])
    assert layer(x).shape == (2, TOKENS, INTER)


def test_attention_weights_are_split_on_head_boundaries():
    rows, head_dim = 8 * 4, 4  # 8 heads
    weight = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
    layer = ColumnParallelLinear.from_full(weight, rank=1, world_size=4, unit=head_dim)
    assert layer.out_features == 8
    np.testing.assert_array_equal(layer.weight, weight[8:16])
    with pytest.raises(ShardError):
        ColumnParallelLinear.from_full(weight[:12], rank=0, world_size=2, unit=head_dim)  # 3 heads, 2 workers


def test_uneven_split_is_refused():
    with pytest.raises(ShardError):
        ColumnParallelLinear.from_full(np.zeros((10, 4), np.float32), rank=0, world_size=4)
    with pytest.raises(ShardError):
        RowParallelLinear.from_full(np.zeros((4, 10), np.float32), group=run_ranks(4, lambda g: g)[0])


def test_wrong_input_width_gives_a_clear_error():
    d = make()
    with pytest.raises(ValueError, match="width 16"):
        ColumnParallelLinear.from_full(d["w1"])(np.zeros((2, 7), np.float32))
    with pytest.raises(ValueError, match="width 32"):
        RowParallelLinear.from_full(d["w2"])(np.zeros((2, 7), np.float32))


def test_bias_must_match_the_weight():
    with pytest.raises(ValueError):
        ColumnParallelLinear(np.zeros((4, 3), np.float32), np.zeros(5, np.float32))
    with pytest.raises(ValueError):
        RowParallelLinear(np.zeros((4, 3), np.float32), np.zeros(5, np.float32))


def test_all_reduce_is_counted_and_timed():
    def work(group):
        layer = RowParallelLinear(np.ones((2, 3), np.float32), group=group)
        for _ in range(3):
            layer(np.ones((1, 3), np.float32))
        return group.reduce_calls, group.reduce_ms

    for calls, ms in run_ranks(2, work):
        assert calls == 3 and ms >= 0


def test_solo_group_needs_no_communication():
    group = SoloGroup()
    x = np.ones(3)
    assert group.all_reduce(x) is x and group.world_size == 1


def test_an_error_in_one_worker_does_not_hang_the_others():
    def work(group):
        if group.rank == 1:
            raise RuntimeError("worker 1 broke")
        group.all_reduce(np.ones(2))

    with pytest.raises(RuntimeError, match="worker 1 broke"):
        run_ranks(2, work, timeout=5)


def test_layers_can_be_built_from_a_checkpoint_shard(tmp_path):
    config, tensors = write_tiny_checkpoint(tmp_path)
    q = "model.layers.0.self_attn.q_proj."
    o = "model.layers.0.self_attn.o_proj.weight"
    x = np.random.default_rng(3).standard_normal((4, config["hidden"])).astype(np.float32)
    full_q = x @ tensors[q + "weight"].T + tensors[q + "bias"]
    attn = np.random.default_rng(4).standard_normal((4, config["heads"] * config["head_dim"])).astype(np.float32)
    full_o = attn @ tensors[o].T
    hd = config["head_dim"]

    with Checkpoint(tmp_path, framework="np") as ckpt:
        q_pieces = [
            ColumnParallelLinear(
                ckpt.load_shard(q + "weight", r, 2, hd), ckpt.load_shard(q + "bias", r, 2, hd)
            )(x)
            for r in range(2)
        ]
        width = attn.shape[1] // 2

        def work(group):
            layer = RowParallelLinear(ckpt.load_shard(o, group.rank, 2, hd), group=group)
            return layer(attn[:, group.rank * width:(group.rank + 1) * width])

        o_results = run_ranks(2, work)

    np.testing.assert_allclose(np.concatenate(q_pieces, axis=-1), full_q, rtol=1e-4, atol=1e-4)
    for result in o_results:
        np.testing.assert_allclose(result, full_o, rtol=1e-4, atol=1e-4)


def run_check(*args):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "tp_check.py"), *args],
        capture_output=True, text=True, timeout=300,
    )


@pytest.mark.parametrize("workers", [1, 2, 3])
def test_check_script_with_threads(workers):
    result = run_check("--backend", "threads", "--workers", str(workers),
                       "--hidden", "48", "--inter", "96", "--tokens", "3", "--repeat", "2")
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"TP CHECK OK: {workers} workers" in result.stdout


def test_check_script_with_real_processes_and_gloo():
    pytest.importorskip("torch")
    result = run_check("--backend", "gloo", "--workers", "2",
                       "--hidden", "64", "--inter", "128", "--tokens", "3", "--repeat", "2")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TP CHECK OK: 2 workers" in result.stdout


def test_torch_tensors_work_in_the_layers():
    torch = pytest.importorskip("torch")
    d = make()
    w, x = torch.from_numpy(d["w1"]), torch.from_numpy(d["x"])
    pieces = [ColumnParallelLinear.from_full(w, rank=r, world_size=2)(x) for r in range(2)]
    expected = torch.nn.functional.linear(x, w)
    assert torch.allclose(torch.cat(pieces, dim=-1), expected, atol=1e-5)
