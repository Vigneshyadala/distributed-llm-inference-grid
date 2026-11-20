"""Tensor-parallel transformer blocks: the MLP and the attention, as in Llama and Qwen.

Both blocks cost exactly ONE all-reduce per call, at the very end:

    MLP        gate_proj and up_proj are column-parallel, SiLU(gate) * up is element-wise,
               down_proj is row-parallel.
    attention  q/k/v projections are column-parallel, so each worker owns whole heads and
               runs their attention locally; o_proj is row-parallel.

Attention supports rotary position embeddings, grouped-query attention (several query
heads share one key/value head) and an optional `past` of keys and values, so a new token
can be decoded without recomputing the earlier ones. The keys and values a worker returns
belong to its own heads only; that is the shard of the KV cache this worker must keep.
"""
import numpy as np

from grid.ops import as_like, cat, repeat_heads, silu, softmax
from grid.tp import ColumnParallelLinear, RowParallelLinear, SoloGroup


# ------------------------------------------------------------------------------- MLP


class ParallelMLP:
    """down_proj(silu(gate_proj(x)) * up_proj(x)) with the weights split across workers."""

    def __init__(self, gate, up, down):
        if gate.out_features != up.out_features or down.in_features != gate.out_features:
            raise ValueError("gate, up and down shards do not fit together")
        self.gate, self.up, self.down = gate, up, down

    @classmethod
    def from_full(cls, gate_weight, up_weight, down_weight, group=None):
        group = group if group is not None else SoloGroup()
        return cls(
            ColumnParallelLinear.from_full(gate_weight, rank=group.rank, world_size=group.world_size),
            ColumnParallelLinear.from_full(up_weight, rank=group.rank, world_size=group.world_size),
            RowParallelLinear.from_full(down_weight, group=group),
        )

    @classmethod
    def from_checkpoint(cls, checkpoint, prefix, group=None):
        """Load only this worker's shards; `prefix` is e.g. "model.layers.0.mlp."."""
        group = group if group is not None else SoloGroup()
        rank, size = group.rank, group.world_size
        return cls(
            ColumnParallelLinear(checkpoint.load_shard(prefix + "gate_proj.weight", rank, size)),
            ColumnParallelLinear(checkpoint.load_shard(prefix + "up_proj.weight", rank, size)),
            RowParallelLinear(checkpoint.load_shard(prefix + "down_proj.weight", rank, size), group=group),
        )

    def __call__(self, x):
        return self.down(silu(self.gate(x)) * self.up(x))


# -------------------------------------------------------------------------- attention


def rotary_tables(positions, head_dim, theta):
    """cos and sin tables, shape [len(positions), head_dim], for rotary embeddings."""
    if head_dim % 2:
        raise ValueError("rotary embeddings need an even head_dim")
    inv_freq = 1.0 / theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    angles = np.asarray(positions, dtype=np.float64)[:, None] * inv_freq[None, :]
    angles = np.concatenate([angles, angles], axis=-1)
    return np.cos(angles), np.sin(angles)


def apply_rotary(x, cos, sin):
    """Rotate the two halves of each head by a position-dependent angle. x: [batch, heads, seq, dim]."""
    half = x.shape[-1] // 2
    rotated = cat([-x[..., half:], x[..., :half]], axis=-1)
    return x * as_like(x, cos) + rotated * as_like(x, sin)


def causal_mask(new, total):
    """Additive mask [new, total]: query i (the i-th new token) may not see keys after total-new+i."""
    return np.triu(np.full((new, total), -np.inf, dtype=np.float32), k=total - new + 1)


class ParallelAttention:
    """Multi-head attention where each worker owns a share of the query and key/value heads."""

    def __init__(self, q, k, v, o, head_dim, rope_theta=1_000_000.0):
        if q.out_features % head_dim or k.out_features % head_dim:
            raise ValueError("projection widths must be whole numbers of heads")
        self.q, self.k, self.v, self.o = q, k, v, o
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.heads = q.out_features // head_dim          # query heads on this worker
        self.kv_heads = k.out_features // head_dim       # key/value heads on this worker
        if self.kv_heads == 0 or self.heads % self.kv_heads:
            raise ValueError("query heads must be a multiple of key/value heads")
        if v.out_features != k.out_features or o.in_features != q.out_features:
            raise ValueError("q, k, v and o shards do not fit together")

    @classmethod
    def from_full(cls, q_weight, k_weight, v_weight, o_weight, *, head_dim, group=None,
                  q_bias=None, k_bias=None, v_bias=None, rope_theta=1_000_000.0):
        group = group if group is not None else SoloGroup()
        rank, size = group.rank, group.world_size
        column = lambda w, b: ColumnParallelLinear.from_full(  # noqa: E731
            w, b, rank=rank, world_size=size, unit=head_dim)
        return cls(
            column(q_weight, q_bias), column(k_weight, k_bias), column(v_weight, v_bias),
            RowParallelLinear.from_full(o_weight, group=group, unit=head_dim),
            head_dim, rope_theta,
        )

    @classmethod
    def from_checkpoint(cls, checkpoint, prefix, *, head_dim, group=None, rope_theta=1_000_000.0):
        """Load only this worker's shards; `prefix` is e.g. "model.layers.0.self_attn."."""
        group = group if group is not None else SoloGroup()
        rank, size = group.rank, group.world_size

        def column(name):
            weight = checkpoint.load_shard(prefix + name + ".weight", rank, size, head_dim)
            bias_name = prefix + name + ".bias"
            bias = checkpoint.load_shard(bias_name, rank, size, head_dim) if bias_name in checkpoint.names() else None
            return ColumnParallelLinear(weight, bias)

        o = RowParallelLinear(checkpoint.load_shard(prefix + "o_proj.weight", rank, size, head_dim), group=group)
        return cls(column("q_proj"), column("k_proj"), column("v_proj"), o, head_dim, rope_theta)

    def __call__(self, x, past=None):
        """Attend over `x` ([batch, new_tokens, hidden]), and over `past` if given.

        `past` is the (keys, values) this worker returned earlier. Returns (output, present):
        the full hidden-size output, and this worker's keys and values for every token so far,
        each [batch, kv_heads_on_this_worker, total_tokens, head_dim], to pass back next time.
        """
        if len(x.shape) != 3:
            raise ValueError(f"attention expects [batch, tokens, hidden], got shape {tuple(x.shape)}")
        batch, new, _ = x.shape
        start = 0 if past is None else past[0].shape[2]
        hd = self.head_dim

        def heads(y, count):
            return y.reshape(batch, new, count, hd).swapaxes(1, 2)  # [batch, count, new, hd]

        q = heads(self.q(x), self.heads)
        k = heads(self.k(x), self.kv_heads)
        v = heads(self.v(x), self.kv_heads)

        cos, sin = rotary_tables(np.arange(start, start + new), hd, self.rope_theta)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)

        if past is not None:
            k, v = cat([past[0], k], axis=2), cat([past[1], v], axis=2)
        present = (k, v)

        group_size = self.heads // self.kv_heads
        keys, values = repeat_heads(k, group_size), repeat_heads(v, group_size)
        scores = (q @ keys.swapaxes(-1, -2)) * (hd ** -0.5)
        scores = scores + as_like(scores, causal_mask(new, start + new))
        context = softmax(scores) @ values                       # [batch, heads, new, hd]
        merged = context.swapaxes(1, 2).reshape(batch, new, self.heads * hd)
        return self.o(merged), present

