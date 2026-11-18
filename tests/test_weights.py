import numpy as np
import pytest

from grid.tinymodel import write_tiny_checkpoint
from grid.weights import (
    Checkpoint,
    ShardError,
    shard_bounds,
    shard_kind,
    shard_shape,
    shard_unit,
)

Q = "model.layers.0.self_attn.q_proj.weight"
K = "model.layers.0.self_attn.k_proj.weight"
O = "model.layers.0.self_attn.o_proj.weight"
GATE = "model.layers.0.mlp.gate_proj.weight"
DOWN = "model.layers.0.mlp.down_proj.weight"
NORM = "model.layers.0.input_layernorm.weight"
EMBED = "model.embed_tokens.weight"


@pytest.fixture
def tiny(tmp_path):
    config, tensors = write_tiny_checkpoint(tmp_path / "tiny")
    return tmp_path / "tiny", config, tensors


def test_shard_kinds_follow_the_tensor_parallel_layout():
    assert shard_kind(Q) == shard_kind(K) == shard_kind(GATE) == "column"
    assert shard_kind("model.layers.3.self_attn.v_proj.bias") == "column"
    assert shard_kind(O) == shard_kind(DOWN) == "row"
    assert shard_kind(EMBED) == shard_kind("lm_head.weight") == "vocab"
    assert shard_kind(NORM) == shard_kind("model.norm.weight") == "replicate"


def test_attention_weights_are_split_by_whole_heads_only():
    assert shard_unit(Q, head_dim=16) == 16
    assert shard_unit(O, head_dim=16) == 16
    assert shard_unit(GATE, head_dim=16) == 1


def test_shard_bounds_cover_a_dimension_without_gaps_or_overlap():
    pieces = [shard_bounds(96, rank, 4) for rank in range(4)]
    assert pieces == [(0, 24), (24, 48), (48, 72), (72, 96)]


def test_shard_bounds_reject_uneven_splits_and_bad_ranks():
    with pytest.raises(ShardError):
        shard_bounds(10, 0, 3)
    with pytest.raises(ShardError):
        shard_bounds(32, 0, 4, unit=16)  # 2 heads cannot be shared by 4 workers
    with pytest.raises(ValueError):
        shard_bounds(8, 2, 2)
    with pytest.raises(ValueError):
        shard_bounds(8, 0, 0)


def test_shard_shape_halves_the_split_dimension(tiny):
    _, config, tensors = tiny
    hd = config["head_dim"]
    assert shard_shape(Q, tensors[Q].shape, 2, hd) == (tensors[Q].shape[0] // 2, tensors[Q].shape[1])
    assert shard_shape(O, tensors[O].shape, 2, hd) == (tensors[O].shape[0], tensors[O].shape[1] // 2)
    assert shard_shape(NORM, tensors[NORM].shape, 2, hd) == tensors[NORM].shape


def test_checkpoint_lists_names_shapes_and_dtypes_without_loading(tiny):
    path, _, tensors = tiny
    with Checkpoint(path, framework="numpy") as ckpt:
        assert ckpt.names() == sorted(tensors)
        assert ckpt.shape(Q) == tensors[Q].shape
        assert ckpt.dtype(Q) == "F32"
        assert ckpt.nbytes(Q) == tensors[Q].nbytes


def test_load_returns_the_exact_tensor(tiny):
    path, _, tensors = tiny
    with Checkpoint(path, framework="numpy") as ckpt:
        assert np.array_equal(ckpt.load(EMBED), tensors[EMBED])


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("name", [Q, K, O, GATE, DOWN, EMBED, NORM, "model.layers.1.self_attn.q_proj.bias"])
def test_shards_put_back_together_equal_the_original(tiny, name, world_size):
    path, config, tensors = tiny
    full = tensors[name]
    with Checkpoint(path, framework="numpy") as ckpt:
        pieces = [ckpt.load_shard(name, r, world_size, config["head_dim"]) for r in range(world_size)]
    dim = {"column": 0, "vocab": 0, "row": 1}.get(shard_kind(name))
    rebuilt = pieces[0] if dim is None else np.concatenate(pieces, axis=dim)
    if dim is None:
        assert all(np.array_equal(piece, full) for piece in pieces)
    assert np.array_equal(rebuilt, full)


def test_a_shard_has_the_shape_the_plan_promised(tiny):
    path, config, _ = tiny
    with Checkpoint(path, framework="numpy") as ckpt:
        for name in ckpt.names():
            piece = ckpt.load_shard(name, 1, 2, config["head_dim"])
            assert piece.shape == shard_shape(name, ckpt.shape(name), 2, config["head_dim"])
            assert piece.nbytes == ckpt.shard_nbytes(name, 2, config["head_dim"])


def test_four_workers_cannot_split_two_kv_heads(tiny):
    path, config, _ = tiny
    with Checkpoint(path, framework="numpy") as ckpt:
        with pytest.raises(ShardError):
            ckpt.load_shard(K, 0, 4, config["head_dim"])
        ckpt.load_shard(Q, 0, 4, config["head_dim"])  # 4 query heads over 4 workers is fine


def test_a_checkpoint_split_over_several_files_loads_the_same(tmp_path):
    _, tensors = write_tiny_checkpoint(tmp_path / "multi", files=3)
    with Checkpoint(tmp_path / "multi", framework="numpy") as ckpt:
        assert ckpt.names() == sorted(tensors)
        assert np.array_equal(ckpt.load(Q), tensors[Q])
        assert np.array_equal(ckpt.load(DOWN), tensors[DOWN])


def test_a_single_file_path_works_too(tiny):
    path, _, tensors = tiny
    with Checkpoint(path / "model.safetensors", framework="numpy") as ckpt:
        assert np.array_equal(ckpt.load(NORM), tensors[NORM])


def test_missing_checkpoint_and_missing_tensor_give_clear_errors(tiny, tmp_path):
    path, _, _ = tiny
    with pytest.raises(FileNotFoundError):
        Checkpoint(tmp_path / "nowhere", framework="numpy")
    with Checkpoint(path, framework="numpy") as ckpt:
        with pytest.raises(KeyError, match="not in this checkpoint"):
            ckpt.shape("no.such.tensor")


def test_torch_shards_match_numpy_shards(tiny):
    torch = pytest.importorskip("torch")
    path, config, tensors = tiny
    with Checkpoint(path, framework="pt") as ckpt:
        piece = ckpt.load_shard(Q, 1, 2, config["head_dim"])
    rows = tensors[Q].shape[0] // 2
    assert isinstance(piece, torch.Tensor)
    assert torch.equal(piece, torch.from_numpy(tensors[Q][rows:]))
