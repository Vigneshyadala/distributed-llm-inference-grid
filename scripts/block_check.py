"""Check a whole tensor-parallel transformer block (attention + MLP) against one worker.

The block has the real shape of a Qwen2.5-0.5B layer (hidden 896, 14 query heads and 2
key/value heads of 64, MLP width 4864) with random weights. It is run:

  * on the full weights by one worker,
  * split across N workers, each holding only its share of the weights,
  * token by token with a key/value cache, to check incremental decoding.

    python scripts/block_check.py                   auto: gloo if torch is installed, else threads
    python scripts/block_check.py --workers 2
    python scripts/block_check.py --backend threads

Exit code 0 means every sharded result matched the full one.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grid.blocks import ParallelAttention, ParallelMLP  # noqa: E402
from grid.runner import pick_backend, run_workers  # noqa: E402
from grid.tp import SoloGroup  # noqa: E402
from grid.weights import ShardError  # noqa: E402

TOLERANCE = 1e-3


def megabytes(layers):
    total = 0
    for layer in layers:
        weight = layer.weight
        total += weight.nbytes if hasattr(weight, "nbytes") else weight.numel() * weight.element_size()
    return total / 1e6


def check_block(group, convert, to_numpy, hidden, heads, kv_heads, head_dim, inter, tokens, repeat, seed=0):
    rng = np.random.default_rng(seed)  # same seed on every worker: same full weights

    def rand(*shape):
        return convert((rng.standard_normal(shape) / np.sqrt(shape[-1])).astype(np.float32))

    q_rows, kv_rows = heads * head_dim, kv_heads * head_dim
    w = {
        "q": rand(q_rows, hidden), "qb": rand(q_rows), "k": rand(kv_rows, hidden), "kb": rand(kv_rows),
        "v": rand(kv_rows, hidden), "vb": rand(kv_rows), "o": rand(hidden, q_rows),
        "gate": rand(inter, hidden), "up": rand(inter, hidden), "down": rand(hidden, inter),
    }
    x = rand(1, tokens, hidden)

    def build(g):
        attention = ParallelAttention.from_full(
            w["q"], w["k"], w["v"], w["o"], head_dim=head_dim, group=g,
            q_bias=w["qb"], k_bias=w["kb"], v_bias=w["vb"])
        return attention, ParallelMLP.from_full(w["gate"], w["up"], w["down"], g)

    full_attention, full_mlp = build(SoloGroup())   # reference: all weights, no communication
    attention, mlp = build(group)                   # this worker's share

    def run(att, ml):
        return ml(att(x)[0])

    expected = to_numpy(run(full_attention, full_mlp))
    block_error = float(np.abs(to_numpy(run(attention, mlp)) - expected).max())

    # incremental decoding: prefill all but two tokens, then decode the last two one by one
    split = tokens - 2
    out, past = attention(x[:, :split])
    pieces = [out]
    for t in range(split, tokens):
        step, past = attention(x[:, t:t + 1], past)
        pieces.append(step)
    cached = np.concatenate([to_numpy(p) for p in pieces], axis=1)
    cache_error = float(np.abs(cached - to_numpy(full_attention(x)[0])).max())
    cache_mb = sum(
        (a.nbytes if hasattr(a, "nbytes") else a.numel() * a.element_size()) for a in past) / 1e6

    full_ms = 0.0
    if group.rank == 0:  # time the full block on rank 0 alone, so the others do not compete for the CPU
        start = time.perf_counter()
        for _ in range(repeat):
            run(full_attention, full_mlp)
        full_ms = 1000 * (time.perf_counter() - start) / repeat
    group.barrier()

    group.reset_stats()
    start = time.perf_counter()
    for _ in range(repeat):
        run(attention, mlp)
    group.barrier()
    sharded_ms = 1000 * (time.perf_counter() - start) / repeat
    return {
        "rank": group.rank, "block_error": block_error, "cache_error": cache_error,
        "full_ms": full_ms, "sharded_ms": sharded_ms, "reduce_ms": group.reduce_ms,
        "reduces_per_pass": group.reduce_calls / repeat,
        "weights_mb": megabytes([attention.q, attention.k, attention.v, attention.o, mlp.gate, mlp.up, mlp.down]),
        "heads": f"{attention.heads} q + {attention.kv_heads} kv", "cache_mb": cache_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="Check a tensor-parallel attention + MLP block.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--backend", choices=["auto", "gloo", "threads"], default="auto")
    parser.add_argument("--hidden", type=int, default=896)
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--inter", type=int, default=4864)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()

    backend = pick_backend(args.backend)
    try:
        results = run_workers(
            check_block, args.workers, backend,
            hidden=args.hidden, heads=args.heads, kv_heads=args.kv_heads, head_dim=args.head_dim,
            inter=args.inter, tokens=args.tokens, repeat=args.repeat,
        )
    except (ShardError, RuntimeError) as exc:
        # a model that cannot be split evenly is a normal answer, not a crash
        print(f"BLOCK CHECK COULD NOT RUN with {args.workers} workers: {exc}")
        print(f"hint: {args.heads} query heads and {args.kv_heads} kv heads must both divide evenly "
              f"across the workers (Qwen2.5-0.5B: 1 or 2 workers).")
        return 2

    print(f"tensor-parallel block check: {args.workers} workers, backend {backend}, hidden {args.hidden}, "
          f"{args.heads} query / {args.kv_heads} kv heads of {args.head_dim}, MLP {args.inter}, {args.tokens} tokens")
    print(f"{'rank':>4}  {'heads held':>12}  {'weights':>9}  {'kv cache':>9}  {'block err':>10}  {'cache err':>10}")
    for r in results:
        print(f"{r['rank']:>4}  {r['heads']:>12}  {r['weights_mb']:>6.1f} MB  {r['cache_mb']:>6.3f} MB  "
              f"{r['block_error']:>10.2e}  {r['cache_error']:>10.2e}")
    full_ms = results[0]["full_ms"]
    sharded_ms = max(r["sharded_ms"] for r in results)
    reduces = results[0]["reduces_per_pass"]
    print(f"one worker, full weights : {full_ms:8.3f} ms per pass")
    print(f"{args.workers} workers, sharded      : {sharded_ms:8.3f} ms per pass "
          f"({reduces:g} all-reduces per pass, {results[0]['reduce_ms']:.3f} ms each on average)")
    if sharded_ms > full_ms:
        print("note: sharding is not faster here. On a 2-core CPU the workers share the same cores,")
        print("      so tensor parallelism is shown for correctness and memory, not for speed.")

    worst = max(max(r["block_error"], r["cache_error"]) for r in results)
    if worst > TOLERANCE:
        print(f"BLOCK CHECK FAILED: max error {worst:.2e} is above {TOLERANCE:.0e}")
        return 1
    print(f"BLOCK CHECK OK: {args.workers} workers match the full block, with and without the cache "
          f"(max error {worst:.2e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
