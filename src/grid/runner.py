"""Run one task on every worker of a small local grid, in threads or in separate processes.

A task is a top-level function `task(group, convert, to_numpy, **params)`:

    group      the communication group of this worker (see grid.tp)
    convert    turns a numpy array into this backend's array type (identity, or torch.from_numpy)
    to_numpy   turns a result back into numpy

`run_workers` returns the tasks' return values ordered by rank. The `gloo` backend starts one
process per worker and uses torch.distributed; the `threads` backend needs only numpy.
"""
import os
import tempfile

from grid.tp import run_ranks

BACKENDS = ("auto", "gloo", "threads")


def pick_backend(name="auto"):
    """Resolve "auto" to gloo when torch is installed, otherwise threads."""
    if name not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, not {name!r}")
    if name != "auto":
        return name
    try:
        import torch  # noqa: F401
    except ImportError:
        return "threads"
    return "gloo"


def _identity(value):
    return value


def identify(group, convert, to_numpy):
    """A tiny task: which worker am I, and in which process? Used to check the grid itself."""
    return {"rank": group.rank, "world_size": group.world_size, "pid": os.getpid()}


def _gloo_worker(rank, world_size, init_method, task, params, queue):
    try:
        import torch

        from grid.tp import TorchGroup

        torch.set_num_threads(1)  # one math thread per worker; the CPU has few cores
        group = TorchGroup(rank, world_size, init_method)
        try:
            result = task(group, torch.from_numpy, lambda t: t.numpy(), **params)
        finally:
            group.close()
        queue.put((rank, True, result))
    except BaseException as exc:  # noqa: BLE001 - report, never hang the parent
        queue.put((rank, False, f"{type(exc).__name__}: {exc}"))


def _run_gloo(task, world_size, params, timeout):
    import torch.multiprocessing as mp

    path = os.path.join(tempfile.mkdtemp(), "rendezvous")
    init_method = ("file:///" + path.replace("\\", "/")) if os.name == "nt" else "file://" + path
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_gloo_worker, args=(rank, world_size, init_method, task, params, queue))
             for rank in range(world_size)]
    for proc in procs:
        proc.start()
    try:
        replies = [queue.get(timeout=timeout) for _ in procs]
        for proc in procs:
            proc.join(timeout=30)
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
    for rank, ok, value in replies:
        if not ok:
            raise RuntimeError(f"worker {rank} failed: {value}")
    return [value for _, _, value in sorted(replies, key=lambda reply: reply[0])]


def run_workers(task, world_size, backend="auto", timeout=120, **params):
    """Run `task` once per worker and return the results ordered by rank."""
    backend = pick_backend(backend)
    if backend == "gloo":
        return _run_gloo(task, world_size, params, timeout)
    return run_ranks(world_size, lambda group: task(group, _identity, _identity, **params))
