"""Show a checkpoint and how it would be sharded across workers. Changes nothing.

    python scripts/inspect_weights.py --demo                 tiny random model, 2 workers
    python scripts/inspect_weights.py --demo --workers 4
    python scripts/inspect_weights.py PATH --workers 2 --head-dim 64
    python scripts/inspect_weights.py --rss-demo             memory use of loading one shard
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grid.tinymodel import write_tiny_checkpoint  # noqa: E402
from grid.weights import Checkpoint, ShardError, shard_kind  # noqa: E402

MB = 1024 * 1024


def show_plan(ckpt, workers, head_dim):
    header = f"{'tensor':<46} {'shape':<14} {'dtype':<5} {'MB':>8} {'split':<9} {'MB/worker':>9}"
    print(header)
    print("-" * len(header))
    total = per_worker = replicated = 0
    unsplittable = []
    for name in ckpt.names():
        size = ckpt.nbytes(name)
        kind = shard_kind(name)
        shape = "x".join(str(n) for n in ckpt.shape(name))
        try:
            piece = ckpt.shard_nbytes(name, workers, head_dim)
        except ShardError as exc:
            unsplittable.append((name, str(exc)))
            print(f"{name:<46} {shape:<14} {ckpt.dtype(name):<5} {size / MB:>8.3f} {kind:<9} {'CANNOT':>9}")
            continue
        total += size
        per_worker += piece
        if kind == "replicate":
            replicated += size
        print(f"{name:<46} {shape:<14} {ckpt.dtype(name):<5} {size / MB:>8.3f} {kind:<9} {piece / MB:>9.3f}")
    if unsplittable:
        print(f"\n{len(unsplittable)} tensors cannot be split across {workers} workers, for example:")
        print(f"  {unsplittable[0][0]}: {unsplittable[0][1]}")
        return 1
    print(f"\ntotal {total / MB:.2f} MB; each of {workers} workers holds {per_worker / MB:.2f} MB "
          f"({replicated / MB:.2f} MB of that is replicated on every worker)")
    return 0


def rss_demo():
    """Load one shard of a 64 MB tensor and report how much memory the process gained."""
    import psutil
    from safetensors.numpy import save_file

    process = psutil.Process()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "model.safetensors"
        name = "model.layers.0.mlp.gate_proj.weight"
        save_file({name: np.ones((4096, 4096), dtype=np.float32)}, str(path))
        print(f"file on disk: {path.stat().st_size / MB:.0f} MB, one tensor {name}")
        before_open = process.memory_info().rss
        with Checkpoint(path, framework="numpy") as ckpt:
            before = process.memory_info().rss
            quarter = ckpt.load_shard(name, 0, 4)
            after_quarter = process.memory_info().rss
            full = ckpt.load(name)
            after_full = process.memory_info().rss
        print(f"opened the checkpoint and read its header: memory grew by {(before - before_open) / MB:.1f} MB")
        print(f"loaded 1 of 4 shards ({quarter.nbytes / MB:.0f} MB): memory grew by {(after_quarter - before) / MB:.0f} MB")
        print(f"then loaded the whole tensor ({full.nbytes / MB:.0f} MB): memory grew by a further {(after_full - after_quarter) / MB:.0f} MB")
        print("(measured resident memory: the copy of the data plus the file pages the operating system mapped to read it)")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Inspect a safetensors checkpoint and its shard plan.")
    parser.add_argument("path", nargs="?", help="a model folder or a .safetensors file")
    parser.add_argument("--demo", action="store_true", help="use a tiny random model")
    parser.add_argument("--rss-demo", action="store_true", help="measure memory use of loading one shard")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=None, help="attention head size (the demo model sets its own)")
    parser.add_argument("--framework", default="pt", choices=["pt", "numpy"])
    args = parser.parse_args()

    if args.rss_demo:
        return rss_demo()
    if args.demo:
        with tempfile.TemporaryDirectory() as folder:
            config, _ = write_tiny_checkpoint(folder)
            print(f"tiny model: hidden {config['hidden']}, {config['heads']} heads of {config['head_dim']}, "
                  f"{config['kv_heads']} KV heads, {config['layers']} layers\n")
            with Checkpoint(folder, framework="numpy") as ckpt:
                return show_plan(ckpt, args.workers, config["head_dim"])
    if not args.path:
        parser.error("give a PATH, or use --demo or --rss-demo")
    if args.head_dim is None:
        parser.error("--head-dim is required for a real model (for example 64)")
    with Checkpoint(args.path, framework=args.framework) as ckpt:
        return show_plan(ckpt, args.workers, args.head_dim)


if __name__ == "__main__":
    sys.exit(main())
