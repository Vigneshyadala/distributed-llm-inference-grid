# Paged KV cache

`src/grid/kvcache.py` is the memory manager for attention. Generating a token needs the keys and
values of every earlier token, so they are kept instead of recomputed. The question is how to keep
them for many sequences of unknown length without wasting memory.

## How it works

The cache owns one **pool of equal-sized blocks**. A block holds the keys and values of
`block_size` tokens for every layer. A sequence owns a **block table**, the list of its block
numbers in token order, and takes one more block only when its last block is full. When it
finishes, its blocks go back to the pool and the next sequence can use them at once.

```
pool of 16 blocks of 4 tokens         [AAADDCCCCad.....]
A has 13 tokens in blocks 0 1 2 9, C has 16 tokens in blocks 5 6 7 8, D has 9 tokens in blocks 3 4 10
capital letter = full block, small letter = partly filled, dot = free block
```

This avoids the two usual problems: no memory is reserved for tokens that may never be generated,
and no gaps appear when sequences of different lengths finish in a different order. The only
waste is the unused part of each sequence's last block (at most `block_size - 1` tokens each),
which `stats()` reports as `wasted_slots`.

## Rules the manager follows

- **Lowest free block first.** Blocks are handed out in ascending order. Workers that perform the
  same operations therefore get the same block tables, so the master can plan the layout once for
  the whole grid and every worker follows it.
- **All or nothing.** If a request needs more blocks than are free, `OutOfBlocks` is raised and
  nothing is allocated. The scheduler can wait or refuse the request; the machine never runs out
  of RAM in the middle of a request. `can_fit(tokens, seq_id)` answers the question beforehand.
- **Fixed memory.** The whole pool is allocated up front, so the memory the cache needs is known
  before any request arrives. `blocks_for_budget(...)` turns a memory budget into a block count.
- **Each worker caches only its own heads.** With 2 workers each one holds one of the two key/value
  heads of Qwen2.5-0.5B, so each needs half the memory per token.

## Using it with attention

```python
cache = PagedKVCache(layers=24, kv_heads=attention.kv_heads, head_dim=64, block_size=16, num_blocks=300)
cache.add_sequence("chat-1")
for layer, attention in enumerate(layers):
    x, _ = attention(x, cache=cache.view("chat-1", layer))
cache.free_sequence("chat-1")
```

`attention(x, cache=...)` stores the new keys and values in the cache and attends over everything
stored for that sequence. It handles one sequence per call; batching several sequences into one
step comes with continuous batching.

## Memory for Qwen2.5-0.5B

24 layers, 2 key/value heads of 64, a bfloat16 cache:

| Workers | Per token (per worker) | A 2048-token chat | Tokens in 256 MiB |
|---|---|---|---|
| 1 | 12.0 KiB | 24.0 MiB | 21,845 |
| 2 | 6.0 KiB | 12.0 MiB | 43,690 |

The model weights are far larger than the cache, so on this 12 GB laptop the cache is not the
limit; it is the length of the queue the scheduler allows.

## "Asynchronous"

The project plan calls this an asynchronous KV-cache manager. The cache itself is a plain
data structure: allocating a block takes microseconds, so it is deliberately synchronous and easy
to test. What is asynchronous is how the scheduler uses it: requests wait in a queue and are
admitted when `can_fit` says there is room, while the requests already running keep decoding. That
part arrives with continuous batching.

## Try it

```
python scripts/kv_demo.py                 the block map above, with memory figures
python scripts/kv_demo.py --check         cached decoding on 2 worker processes vs full attention
```
