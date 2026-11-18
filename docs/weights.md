# Weight loading and sharding

Code: [`src/grid/weights.py`](../src/grid/weights.py). Inspect a checkpoint with `python scripts/inspect_weights.py`.

## Memory-mapped loading

safetensors files start with a small header that lists every tensor's name, dtype, shape and byte range, followed by the raw data. `Checkpoint` opens the file with `safe_open`, which memory-maps it:

- opening a checkpoint, and asking for names, shapes, dtypes or sizes, reads only the header;
- `load(name)` reads one whole tensor;
- `load_shard(name, rank, world_size, head_dim)` reads only the slice that belongs to one worker, so a worker never has to hold the full weights in memory.

Folders with several files (`model.safetensors.index.json`) are supported as well as a single `model.safetensors`.

Weights are read with `framework="pt"` (PyTorch tensors) by default. bfloat16 weights, which Qwen2.5 uses, can only be read that way, because NumPy has no bfloat16 type; the tests use `framework="numpy"` with float32 test weights.

`python scripts/inspect_weights.py --rss-demo` measures this on your machine: it reports how much process memory grows after opening a 64 MB file, after loading one quarter of the tensor, and after loading all of it.

## Shard plan

Linear weights are stored as `[out_features, in_features]`. The split follows the Megatron-style layout used by Llama and Qwen:

| Kind | Tensors | Split | Why |
|------|---------|-------|-----|
| column | `q_proj`, `k_proj`, `v_proj` (weight and bias), `gate_proj`, `up_proj` | rows (dimension 0) | each worker computes its own heads or its own part of the hidden layer |
| row | `o_proj`, `down_proj` | columns (dimension 1) | each worker consumes the output of its own part; partial results are summed by an all-reduce |
| vocab | `embed_tokens`, `lm_head` | rows (dimension 0) | the vocabulary is shared out between workers |
| replicate | layer norms and everything else | not split | small, and every worker needs all of it |

## Limits that come from the model

- An attention head must never be cut in half, so `q`, `k`, `v` and `o` are split in whole heads (`head_dim` rows or columns at a time).
- The number of query heads and of key/value heads must both divide evenly by the number of workers. If they do not, `load_shard` raises `ShardError` and the plan reports which tensors cannot be split.
- Qwen2.5-0.5B (hidden size 896, 14 attention heads of 64, 2 key/value heads, 24 layers, tied embeddings, bfloat16 weights, values from its `config.json`) can therefore run on **1 or 2 workers**: with 2 workers each one holds 7 query heads and 1 key/value head. Four workers would not divide 14 heads or 2 key/value heads. These numbers are checked again when the model is downloaded.
- With tied embeddings the file has no separate `lm_head.weight`; the output layer reuses `embed_tokens`. The loader will treat that case when the model is wired up.

## Test model

`grid.tinymodel.write_tiny_checkpoint` writes a small Llama-style checkpoint with random weights, with real tensor names and shapes, as one file or split over several. It is what the tests use, so nothing has to be downloaded to test the sharding code.
