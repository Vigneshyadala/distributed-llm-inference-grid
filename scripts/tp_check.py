"""Check that tensor-parallel layers give the same answer as one ordinary layer pair.

Each worker holds a slice of a column-parallel layer, a ReLU, and a row-parallel layer
(the shape of a transformer MLP). The result of the workers is compared with the same
computation done on the full weights, and the times are printed.

    python scripts/tp_check.py                      auto: gloo if torch is installed, else threads
    python scripts/tp_check.py --workers 2          number of workers (default 2)
    python scripts/tp_check.py --backend gloo       separate processes, torch.distributed (CPU)
    python scripts/tp_check.py --backend threads    threads and numpy, no torch needed

Exit code 0 means the sharded result matched the full one.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grid.runner import pick_backend, run_workers  # noqa: E402
from grid.tp import ColumnParallelLinear, RowParallelLinear  # noqa: E402

TOLERANCE = 1e-4


def megabytes(array):
    """Size of a numpy array or torch tensor in MB."""
    if hasattr(array, "nbytes"):
        return array.nbytes / 1e6
    return array.numel() * array.element_size() / 1e6


def relu(x):
    return x * (x > 0)


def check_rank(group, convert, to_numpy, hidden, inter, tokens, repeat, seed=0):
    """Run the check on one worker. `convert` makes native arrays, `to_numpy` turns them back."""
    rng = np.random.default_rng(seed)  # same seed everywhere: every worker sees the same full weights

    def rand(*shape):
        return (rng.standard_normal(shape) / np.sqrt(shape[-1])).astype(np.float32)

    w1, b1, w2, b2, x = (convert(a) for a in (
        rand(inter, hidden), rand(inter), rand(hidden, inter), rand(hidden), rand(tokens, hidden),
    ))

    up = ColumnParallelLinear.from_full(w1, b1, rank=group.rank, world_size=group.world_size)
    down = RowParallelLinear.from_full(w2, b2, group=group)

    def full():
        return relu(x @ w1.T + b1) @ w2.T + b2

    def sharded():
        return down(relu(up(x)))

    error = float(np.abs(to_numpy(sharded()) - to_numpy(full())).max())

    # time the full computation on rank 0 alone, so the other workers do not compete for the CPU
    full_ms = 0.0
    if group.rank == 0:
        start = time.perf_counter()
        for _ in range(repeat):
            full()
        full_ms = 1000 * (time.perf_counter() - start) / repeat
    group.barrier()

    group.reset_stats()
    start = time.perf_counter()
    for _ in range(repeat):
        sharded()
    group.barrier()
    sharded_ms = 1000 * (time.perf_counter() - start) / repeat
    return {
        "rank": group.rank, "error": error, "full_ms": full_ms, "sharded_ms": sharded_ms,
        "reduce_ms": group.reduce_ms, "shard_mb": (megabytes(up.weight) + megabytes(down.weight)),
    }


def main():
    parser = argparse.ArgumentParser(description="Check tensor-parallel layers against full layers.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--backend", choices=["auto", "gloo", "threads"], default="auto")
    parser.add_argument("--hidden", type=int, default=896, help="model width (Qwen2.5-0.5B: 896)")
    parser.add_argument("--inter", type=int, default=4864, help="MLP width (Qwen2.5-0.5B: 4864)")
    parser.add_argument("--tokens", type=int, default=8, help="tokens in the test batch")
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    backend = pick_backend(args.backend)
    results = run_workers(
        check_rank, args.workers, backend,
        hidden=args.hidden, inter=args.inter, tokens=args.tokens, repeat=args.repeat,
    )

    print(f"tensor-parallel check: {args.workers} workers, backend {backend}, "
          f"hidden {args.hidden}, MLP width {args.inter}, {args.tokens} tokens")
    print(f"{'rank':>4}  {'max error':>10}  {'weights held':>12}  {'all-reduce':>10}")
    for r in results:
        print(f"{r['rank']:>4}  {r['error']:>10.2e}  {r['shard_mb']:>9.1f} MB  {r['reduce_ms']:>7.3f} ms")
    full_ms = results[0]["full_ms"]
    sharded_ms = max(r["sharded_ms"] for r in results)
    print(f"one worker, full weights : {full_ms:8.3f} ms per pass")
    print(f"{args.workers} workers, sharded      : {sharded_ms:8.3f} ms per pass")
    if sharded_ms > full_ms:
        print("note: sharding is not faster here. On a 2-core CPU the workers share the same cores,")
        print("      so tensor parallelism is shown for correctness and memory, not for speed.")

    worst = max(r["error"] for r in results)
    if worst > TOLERANCE:
        print(f"TP CHECK FAILED: max error {worst:.2e} is above {TOLERANCE:.0e}")
        return 1
    print(f"TP CHECK OK: {args.workers} workers match the full layers (max error {worst:.2e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
