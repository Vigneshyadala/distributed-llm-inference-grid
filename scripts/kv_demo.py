"""Watch the paged KV cache work, plan its memory, and check it across workers.

    python scripts/kv_demo.py                       a small demo: sequences grow, finish, and free their blocks
    python scripts/kv_demo.py --check               check cached decoding against full attention (2 workers)
    python scripts/kv_demo.py --check --workers 1 --backend threads

The demo needs only numpy. In the block map every block is one character: a letter is the
sequence that owns it (capital = full, small = partly filled) and a dot is a free block.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grid.blocks import ParallelAttention  # noqa: E402
from grid.kvcache import OutOfBlocks, PagedKVCache, bytes_per_token  # noqa: E402
from grid.ops import is_torch  # noqa: E402
from grid.runner import pick_backend, run_workers  # noqa: E402
from grid.tp import SoloGroup  # noqa: E402
from grid.weights import ShardError  # noqa: E402

MIB = 1024 * 1024
TOLERANCE = 1e-3


# ------------------------------------------------------------------------------ demo


def draw(cache):
    chars = []
    for entry in cache.block_map():
        if entry is None:
            chars.append(".")
        else:
            letter = str(entry["seq"])
            chars.append(letter.upper() if entry["tokens"] == cache.block_size else letter.lower())
    return "".join(chars)


def show(cache, event):
    s = cache.stats()
    print(f"{event:<38} [{draw(cache)}]  blocks {s['blocks_used']:>2}/{s['blocks_total']}  "
          f"tokens {s['tokens_stored']:>3}  unused slots {s['wasted_slots']}")


def grow(cache, seq_id, tokens):
    one = np.zeros((1, 1, tokens, 2), dtype=np.float32)
    cache.extend(seq_id, 0, one, one)


def demo():
    cache = PagedKVCache(layers=1, kv_heads=1, head_dim=2, block_size=4, num_blocks=16)
    print("A pool of 16 blocks, 4 tokens per block. Capital = full block, small = partly filled.\n")
    show(cache, "empty pool")
    for seq_id, prompt in (("a", 10), ("b", 6), ("c", 13)):
        cache.add_sequence(seq_id)
        grow(cache, seq_id, prompt)
        show(cache, f"prompt of {prompt} tokens arrives ({seq_id.upper()})")
    for _ in range(3):
        for seq_id in "abc":
            grow(cache, seq_id, 1)
    show(cache, "3 decoding steps for A, B and C")
    freed = cache.free_sequence("b")
    show(cache, f"B finishes, {freed} blocks freed")
    cache.add_sequence("d")
    grow(cache, "d", 9)
    show(cache, "D arrives (9 tokens) and reuses them")
    cache.add_sequence("e")
    try:
        grow(cache, "e", 30)
    except OutOfBlocks as exc:
        show(cache, "E needs 30 tokens: refused")
        print(f"   -> {exc}")
    print("\nMemory for Qwen2.5-0.5B (24 layers, 2 key/value heads of 64, bfloat16 cache):")
    print(f"  {'workers':>7}  {'per token':>10}  {'2048-token chat':>16}  {'tokens in 256 MiB':>18}")
    for workers in (1, 2):
        per_token = bytes_per_token(24, 2 // workers, 64, "bfloat16")
        print(f"  {workers:>7}  {per_token / 1024:>7.1f} KiB  {per_token * 2048 / MIB:>12.1f} MiB  "
              f"{256 * MIB // per_token:>18,}")
    print("  (per worker: with 2 workers each holds one key/value head, so half the memory each)")
    return 0


# ------------------------------------------------------------------------------ check


def check_cache(group, convert, to_numpy, hidden, heads, kv_heads, head_dim, layers, tokens, block_size, seed=0):
    rng = np.random.default_rng(seed)  # same weights on every worker

    def rand(*shape):
        return convert((rng.standard_normal(shape) / np.sqrt(shape[-1])).astype(np.float32))

    q_rows, kv_rows = heads * head_dim, kv_heads * head_dim
    weights = [{
        "q": rand(q_rows, hidden), "qb": rand(q_rows), "k": rand(kv_rows, hidden), "kb": rand(kv_rows),
        "v": rand(kv_rows, hidden), "vb": rand(kv_rows), "o": rand(hidden, q_rows),
    } for _ in range(layers)]
    inputs = {"a": rand(1, tokens, hidden), "b": rand(1, tokens - 3, hidden)}

    def build(g):
        return [ParallelAttention.from_full(
            w["q"], w["k"], w["v"], w["o"], head_dim=head_dim, group=g,
            q_bias=w["qb"], k_bias=w["kb"], v_bias=w["vb"]) for w in weights]

    full, mine = build(SoloGroup()), build(group)
    expected = {}
    for seq_id, x in inputs.items():          # reference: whole sequence, all weights, no cache
        for attention in full:
            x = attention(x)[0]
        expected[seq_id] = to_numpy(x)

    backend = "torch" if is_torch(convert(np.zeros(1, np.float32))) else "numpy"
    blocks = 2 * -(-tokens // block_size) + 2
    cache = PagedKVCache(layers, mine[0].kv_heads, head_dim, block_size, blocks, backend=backend)
    for seq_id in inputs:
        cache.add_sequence(seq_id)
    outputs = {"a": [], "b": []}
    prefill = 4                               # both prompts are 4 tokens, then they decode one token at a time, alternating
    schedule = [(s, 0, prefill) for s in inputs]
    for position in range(prefill, tokens):
        schedule += [(s, position, position + 1) for s in inputs if position < inputs[s].shape[1]]
    for seq_id, start, stop in schedule:
        x = inputs[seq_id][:, start:stop]
        for layer, attention in enumerate(mine):
            x = attention(x, cache=cache.view(seq_id, layer))[0]
        outputs[seq_id].append(to_numpy(x))
    error = max(float(np.abs(np.concatenate(outputs[s], axis=1) - expected[s]).max()) for s in inputs)
    stats = cache.stats()
    return {"rank": group.rank, "error": error, "kv_heads": mine[0].kv_heads, "blocks": stats["blocks_used"],
            "tokens": stats["tokens_stored"], "bytes_per_token": stats["bytes_per_token"]}


def check(args):
    backend = pick_backend(args.backend)
    try:
        results = run_workers(
            check_cache, args.workers, backend, hidden=args.hidden, heads=args.heads, kv_heads=args.kv_heads,
            head_dim=args.head_dim, layers=args.layers, tokens=args.tokens, block_size=args.block_size)
    except (ShardError, RuntimeError) as exc:
        print(f"KV CACHE CHECK COULD NOT RUN with {args.workers} workers: {exc}")
        return 2
    print(f"cached decoding check: {args.workers} workers, backend {backend}, {args.layers} layers, "
          f"two interleaved sequences of {args.tokens} and {args.tokens - 3} tokens, {args.block_size}-token blocks")
    print(f"{'rank':>4}  {'kv heads':>8}  {'blocks used':>11}  {'tokens':>6}  {'bytes/token':>11}  {'max error':>10}")
    for r in results:
        print(f"{r['rank']:>4}  {r['kv_heads']:>8}  {r['blocks']:>11}  {r['tokens']:>6}  "
              f"{r['bytes_per_token']:>11,}  {r['error']:>10.2e}")
    worst = max(r["error"] for r in results)
    if worst > TOLERANCE:
        print(f"KV CACHE CHECK FAILED: max error {worst:.2e} is above {TOLERANCE:.0e}")
        return 1
    print(f"KV CACHE CHECK OK: cached decoding matches full attention (max error {worst:.2e})")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Demo and check of the paged KV cache.")
    parser.add_argument("--check", action="store_true", help="check cached decoding across workers")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--backend", choices=["auto", "gloo", "threads"], default="auto")
    parser.add_argument("--hidden", type=int, default=896)
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=12)
    parser.add_argument("--block-size", type=int, default=4)
    args = parser.parse_args()
    return check(args) if args.check else demo()


if __name__ == "__main__":
    sys.exit(main())
