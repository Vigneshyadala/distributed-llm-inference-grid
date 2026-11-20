# Tensor-parallel linear layers

`src/grid/tp.py` holds the two building blocks of tensor parallelism. A linear layer computes
`y = x @ W.T + b` with `W` stored as `[out_features, in_features]`.

| Layer | Each worker keeps | Input | Output | Communication |
|---|---|---|---|---|
| `ColumnParallelLinear` | a block of the **rows** of `W` (and of `b`) | the full `x` | its own slice of `y` | none |
| `RowParallelLinear` | a block of the **columns** of `W` | its own slice of `x` | the full `y` | one all-reduce (sum) |

The bias of a row-parallel layer is added once, after the sum. Adding it on every worker
before the sum would count it `world_size` times.

## Why the pair needs only one all-reduce

```
x (full) --> [column-parallel: W1 rows] --> slice of h --> ReLU/SiLU --> [row-parallel: W2 columns] --> partial y --> all-reduce --> y (full)
```

An element-wise function acts on each value separately, so it can run on a slice. The slice
of `h` that worker `r` produced is exactly the slice of the input that worker `r`'s block of
`W2` needs. So a whole MLP costs one all-reduce, not one per layer. Attention works the same
way: the heads are split across workers, and only the output projection needs the sum.

Attention weights are cut on head boundaries (`unit=head_dim`), never through the middle of
a head. A model whose number of heads cannot be divided evenly raises `ShardError`; see
[weights.md](weights.md).

## Communication groups

The layers only use `@`, `.T`, `+` and slicing, so they work on numpy arrays and torch
tensors. Communication goes through a group object:

| Group | Workers are | Used for |
|---|---|---|
| `SoloGroup` | one worker | single-process runs |
| `ThreadGroup` | threads of one process (numpy) | tests, and the check on machines without torch |
| `TorchGroup` | separate processes, `torch.distributed` with `gloo` | the real grid |

Each group counts its all-reduce calls and the time spent in them (`reduce_ms`). The time
includes waiting for the slowest worker, which is what the dashboard will later show.

## Check it

```
python scripts/tp_check.py --workers 2
```

This builds a transformer-sized MLP (Qwen2.5-0.5B widths: 896 and 4864), runs it on the full
weights and on `N` workers, and prints the largest difference, the weights each worker holds,
the all-reduce time, and the time per pass. With torch installed it uses separate processes
and `gloo`; otherwise it falls back to threads. Use `--backend threads` or `--backend gloo`
to choose.

On a 2-core CPU the workers share the same cores, so sharding is shown for correctness and
memory (each worker holds `1/N` of the weights), not for speed. The script says so when the
sharded run is not faster.

## MLP and attention blocks

`src/grid/blocks.py` builds the two halves of a transformer layer from those layers. Each costs
exactly **one** all-reduce, so a layer costs two.

| Block | Column-parallel (no communication) | Local work | Row-parallel (one all-reduce) |
|---|---|---|---|
| `ParallelMLP` | `gate_proj`, `up_proj` | `silu(gate) * up` | `down_proj` |
| `ParallelAttention` | `q_proj`, `k_proj`, `v_proj` | rotary embeddings, attention over the worker's own heads | `o_proj` |

**Whole heads per worker.** The attention weights are cut on head boundaries, so every worker owns
complete query heads and the complete key/value heads they use. Grouped-query attention (several
query heads sharing one key/value head) stays consistent because both counts are divided by the
same number of workers. That is why the number of workers must divide both the query heads and the
key/value heads: Qwen2.5-0.5B has 14 and 2, so it runs on 1 or 2 workers. Asking for 4 gives a
clear "cannot be split evenly" message, not a wrong answer.

**Rotary embeddings and the causal mask** are applied per head, locally. A token only attends to
itself and earlier tokens.

**Incremental decoding.** `attention(x, past)` takes the keys and values returned by the previous
call and returns the extended ones. Feeding a prompt once and then one token at a time gives the
same answer as running the whole sequence in one go (tested). The `(keys, values)` a worker returns
are its own heads only, which is exactly the share of the KV cache that worker must store. The KV
cache manager, which will own that memory, builds on this.

**Loading.** `from_checkpoint(checkpoint, prefix, ...)` reads only the worker's shards from the
memory-mapped safetensors files (see [weights.md](weights.md)); `from_full(...)` cuts them from
full weights already in memory.

### Check a whole block

```
python scripts/block_check.py --workers 2
```

The block has the real shape of a Qwen2.5-0.5B layer (hidden 896, 14 query and 2 key/value heads of
64, MLP width 4864) with random weights. The script compares the sharded block with the same block
on the full weights, then checks token-by-token decoding with a cache, and prints the heads and
megabytes each worker holds, the errors, the all-reduce count and the timings.
