"""Report what this machine can do for the project. Changes nothing.

    python scripts/check_env.py              versions and hardware
    python scripts/check_env.py --allreduce  also run a 2-process gloo all-reduce
"""
import argparse
import importlib
import os
import platform
import shutil
import sys
import tempfile

REQUIRED = ["torch", "safetensors", "numpy", "grpc", "fastapi", "uvicorn", "websockets", "psutil"]


def version_of(name):
    try:
        module = importlib.import_module(name)
    except ImportError:
        return None
    return getattr(module, "__version__", "installed")


def _worker(rank, world_size, init_method, queue):
    import torch
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)
    tensor = torch.tensor([float(rank + 1)])
    dist.all_reduce(tensor)
    queue.put((rank, tensor.item()))
    dist.destroy_process_group()


def run_allreduce(world_size=2):
    import torch.multiprocessing as mp

    path = os.path.join(tempfile.mkdtemp(), "rendezvous")
    if os.name == "nt":
        init_method = "file:///" + path.replace("\\", "/")
    else:
        init_method = "file://" + path
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, world_size, init_method, queue)) for r in range(world_size)]
    for proc in procs:
        proc.start()
    try:
        results = [queue.get(timeout=90) for _ in procs]
        for proc in procs:
            proc.join(timeout=30)
    except Exception as exc:
        print(f"  all-reduce FAILED: {type(exc).__name__}: {exc}")
        return False
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
    expected = float(sum(range(1, world_size + 1)))
    good = all(value == expected for _, value in results)
    print(f"  all-reduce {'OK' if good else 'WRONG'}: each of {world_size} workers got "
          f"{sorted(v for _, v in results)} (expected {expected} on every worker)")
    return good


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allreduce", action="store_true", help="run a 2-process gloo all-reduce test")
    args = parser.parse_args()

    print("Machine")
    print(f"  python    : {sys.version.split()[0]} ({platform.platform()})")
    print(f"  cpu cores : {os.cpu_count()} logical")
    psutil_version = version_of("psutil")
    if psutil_version:
        import psutil

        mem = psutil.virtual_memory()
        print(f"  ram       : {mem.total / 2**30:.1f} GB total, {mem.available / 2**30:.1f} GB free")
    free = shutil.disk_usage(".").free / 2**30
    print(f"  disk free : {free:.1f} GB (drive of the current folder)")

    print("Libraries")
    missing = []
    for name in REQUIRED:
        found = version_of(name)
        print(f"  {name:<12}: {found or 'MISSING'}")
        if not found:
            missing.append(name)

    if "torch" not in missing:
        import torch
        import torch.distributed as dist

        print("PyTorch")
        print(f"  cuda available : {torch.cuda.is_available()}")
        print(f"  distributed    : {dist.is_available()}")
        print(f"  gloo backend   : {dist.is_gloo_available()}")
        print(f"  nccl backend   : {dist.is_nccl_available()}")
        if args.allreduce:
            print("All-reduce test")
            if not run_allreduce():
                return 1

    if missing:
        print("Missing: " + ", ".join(missing) + "  (run: pip install -r requirements-dev.txt)")
        return 1
    print("Environment OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
