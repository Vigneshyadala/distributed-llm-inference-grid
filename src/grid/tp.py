"""Tensor-parallel linear layers.

A linear layer computes y = x @ W.T + b, with W stored as [out_features, in_features].
Two ways of splitting W across workers let a pair of layers run with ONE communication:

    column-parallel  each worker keeps a block of the output rows of W (and of b).
                     It needs the full input x and produces its own slice of y.
                     No communication.
    row-parallel     each worker keeps a block of the input columns of W.
                     It needs only its own slice of x and produces a partial y of the
                     full size. The partial results are summed with an all-reduce.
                     The bias is added once, after the sum.

Put a column-parallel layer first, then an element-wise function (ReLU, SiLU, ...), then
a row-parallel layer, and the sliced output of the first is exactly the sliced input the
second needs. That is the Megatron-LM trick used for the MLP and attention blocks.

The layers only use `@`, `.T`, `+` and slicing, so they work on numpy arrays and torch
tensors alike. Communication goes through a small "group" object:

    SoloGroup    one worker, nothing to communicate
    ThreadGroup  several workers as threads of one process (numpy; for tests)
    TorchGroup   several worker processes using torch.distributed with the gloo backend
"""
import threading
import time

from grid.weights import shard_bounds


def _own(array):
    """A copy that owns its memory, so a worker does not keep the full weight alive."""
    return array.clone() if hasattr(array, "clone") else array.copy()


# ---------------------------------------------------------------- communication groups


class Group:
    """What a layer needs from the communication layer, plus simple timing statistics."""

    def __init__(self, rank=0, world_size=1):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is not valid in a group of {world_size}")
        self.rank = rank
        self.world_size = world_size
        self.reduce_calls = 0
        self.reduce_seconds = 0.0

    def all_reduce(self, x):
        """Return the element-wise sum of `x` over all workers (same result on every worker)."""
        start = time.perf_counter()
        total = self._sum(x)
        self.reduce_seconds += time.perf_counter() - start
        self.reduce_calls += 1
        return total

    @property
    def reduce_ms(self):
        """Average milliseconds per all-reduce so far (includes waiting for slower workers)."""
        return 1000 * self.reduce_seconds / self.reduce_calls if self.reduce_calls else 0.0

    def reset_stats(self):
        self.reduce_calls = 0
        self.reduce_seconds = 0.0

    def barrier(self):
        """Wait until every worker has reached this point."""

    def _sum(self, x):
        raise NotImplementedError


class SoloGroup(Group):
    """A group of one worker: the sum is the value itself."""

    def _sum(self, x):
        return x


class _Hub:
    """State shared by the ThreadGroup members of one group."""

    def __init__(self, world_size, timeout):
        self.timeout = timeout
        self.barrier = threading.Barrier(world_size)
        self.slots = [None] * world_size


class ThreadGroup(Group):
    """Workers that are threads in one process, exchanging numpy arrays through memory."""

    def __init__(self, hub, rank, world_size):
        super().__init__(rank, world_size)
        self._hub = hub

    def barrier(self):
        self._hub.barrier.wait(self._hub.timeout)

    def _sum(self, x):
        hub = self._hub
        hub.slots[self.rank] = x
        hub.barrier.wait(hub.timeout)
        total = hub.slots[0]
        for part in hub.slots[1:]:
            total = total + part
        hub.barrier.wait(hub.timeout)  # nobody overwrites a slot before everyone has read
        return total


def thread_groups(world_size, timeout=30.0):
    """One ThreadGroup per rank, all connected to each other."""
    hub = _Hub(world_size, timeout)
    return [ThreadGroup(hub, rank, world_size) for rank in range(world_size)]


def run_ranks(world_size, function, timeout=30.0):
    """Run `function(group)` once per rank, each in its own thread; return the results by rank.

    If any rank raises, the others are released and the first error is raised here.
    """
    groups = thread_groups(world_size, timeout)
    results = [None] * world_size
    errors = []

    def target(group):
        try:
            results[group.rank] = function(group)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)
            group._hub.barrier.abort()

    threads = [threading.Thread(target=target, args=(group,)) for group in groups]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        # a broken barrier is only a symptom; show the error that caused it first
        real = [e for e in errors if not isinstance(e, threading.BrokenBarrierError)]
        raise (real or errors)[0]
    return results


class TorchGroup(Group):
    """Worker processes talking through torch.distributed (gloo backend, CPU)."""

    def __init__(self, rank, world_size, init_method):
        super().__init__(rank, world_size)
        import torch.distributed as dist

        self._dist = dist
        dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)

    def barrier(self):
        self._dist.barrier()

    def _sum(self, x):
        out = x.contiguous().clone()
        self._dist.all_reduce(out)
        return out

    def close(self):
        if self._dist.is_initialized():
            self._dist.destroy_process_group()


# ------------------------------------------------------------------------------ layers


def _check_input(x, in_features, what):
    if x.shape[-1] != in_features:
        raise ValueError(f"{what} expects an input of width {in_features}, got {x.shape[-1]}")


class ColumnParallelLinear:
    """One worker's block of the output rows of a linear layer. Needs no communication.

    `weight` is this worker's shard, [out_features // world_size, in_features]. The output
    is this worker's slice of y: the slices of all workers, joined along the last
    dimension in rank order, are the full output.
    """

    def __init__(self, weight, bias=None):
        if bias is not None and bias.shape[0] != weight.shape[0]:
            raise ValueError("bias must have one value per row of the weight shard")
        self.weight = weight
        self.bias = bias

    @classmethod
    def from_full(cls, weight, bias=None, *, rank=0, world_size=1, unit=1):
        """Cut this worker's rows out of a full weight (`unit` = smallest piece, e.g. head_dim)."""
        start, stop = shard_bounds(weight.shape[0], rank, world_size, unit)
        return cls(
            _own(weight[start:stop]),
            None if bias is None else _own(bias[start:stop]),
        )

    @property
    def in_features(self):
        return self.weight.shape[1]

    @property
    def out_features(self):
        """Width of this worker's slice of the output."""
        return self.weight.shape[0]

    def __call__(self, x):
        _check_input(x, self.in_features, "ColumnParallelLinear")
        y = x @ self.weight.T
        return y if self.bias is None else y + self.bias


class RowParallelLinear:
    """One worker's block of the input columns of a linear layer. One all-reduce per call.

    `weight` is this worker's shard, [out_features, in_features // world_size]. The input
    is this worker's slice of x (the last dimension). Every worker returns the full,
    summed output. The bias, if any, is full size and is added once after the sum.
    """

    def __init__(self, weight, bias=None, group=None):
        if bias is not None and bias.shape[0] != weight.shape[0]:
            raise ValueError("bias must have one value per output row")
        self.weight = weight
        self.bias = bias
        self.group = group if group is not None else SoloGroup()

    @classmethod
    def from_full(cls, weight, bias=None, *, group=None, unit=1):
        """Cut this worker's columns out of a full weight; rank and size come from `group`."""
        group = group if group is not None else SoloGroup()
        start, stop = shard_bounds(weight.shape[1], group.rank, group.world_size, unit)
        return cls(
            _own(weight[:, start:stop]),
            None if bias is None else _own(bias),
            group,
        )

    @property
    def in_features(self):
        """Width of this worker's slice of the input."""
        return self.weight.shape[1]

    @property
    def out_features(self):
        return self.weight.shape[0]

    def __call__(self, x):
        _check_input(x, self.in_features, "RowParallelLinear")
        y = self.group.all_reduce(x @ self.weight.T)
        return y if self.bias is None else y + self.bias
