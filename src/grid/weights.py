"""Memory-mapped loading of safetensors checkpoints, and the tensor-parallel shard plan.

safetensors files are opened with mmap, so opening a checkpoint reads only its small
header. A tensor, or a slice of one, is read from disk only when it is asked for. That is
what lets a worker load just its own shard of a weight instead of the whole model.

Sharding follows the Megatron-style layout used by Llama and Qwen models, where every
linear weight is stored as [out_features, in_features]:

    column-parallel  q/k/v projections (and their biases), gate_proj, up_proj
                     -> split the output rows (dimension 0)
    row-parallel     o_proj, down_proj
                     -> split the input columns (dimension 1); partial outputs are summed
                        with an all-reduce
    vocab-parallel   embed_tokens, lm_head -> split the vocabulary rows (dimension 0)
    replicated       layer norms and anything else -> every worker keeps a full copy
"""
import json
from contextlib import ExitStack
from math import prod
from pathlib import Path

from safetensors import safe_open

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}

COLUMN_PARALLEL = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight",
    "q_proj.bias", "k_proj.bias", "v_proj.bias",
    "gate_proj.weight", "up_proj.weight",
)
ROW_PARALLEL = ("o_proj.weight", "down_proj.weight")
VOCAB_PARALLEL = ("embed_tokens.weight", "lm_head.weight")
ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")


class ShardError(ValueError):
    """A tensor cannot be split the requested way."""


def shard_kind(name):
    """How a tensor is distributed: column, row, vocab or replicate."""
    if name.endswith(COLUMN_PARALLEL):
        return "column"
    if name.endswith(ROW_PARALLEL):
        return "row"
    if name.endswith(VOCAB_PARALLEL):
        return "vocab"
    return "replicate"


def shard_dim(kind):
    """The tensor dimension that is split for a kind, or None when it is replicated."""
    return {"column": 0, "vocab": 0, "row": 1}.get(kind)


def shard_unit(name, head_dim):
    """Smallest piece that may not be cut: one attention head for attention weights."""
    return head_dim if any(part in name for part in ATTENTION) else 1


def shard_bounds(size, rank, world_size, unit=1):
    """Start and stop of one worker's piece of a dimension of length `size`."""
    if world_size < 1:
        raise ValueError("world_size must be at least 1")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} is outside a grid of {world_size} workers")
    if size % unit:
        raise ShardError(f"a dimension of {size} is not a whole number of {unit}-sized units")
    units = size // unit
    if units % world_size:
        raise ShardError(
            f"{units} units of {unit} cannot be split evenly across {world_size} workers"
        )
    per_worker = units // world_size * unit
    return rank * per_worker, (rank + 1) * per_worker


def shard_shape(name, shape, world_size, head_dim=1):
    """Shape of one worker's piece of a tensor (all workers get the same shape)."""
    dim = shard_dim(shard_kind(name))
    if dim is None or world_size == 1:
        return tuple(shape)
    start, stop = shard_bounds(shape[dim], 0, world_size, shard_unit(name, head_dim))
    piece = list(shape)
    piece[dim] = stop - start
    return tuple(piece)


class Checkpoint:
    """A safetensors checkpoint: one file, or a folder (with or without an index file).

    Use it as a context manager so the memory maps are released:

        with Checkpoint("models/qwen") as ckpt:
            weight = ckpt.load_shard("model.layers.0.self_attn.q_proj.weight", rank=0, world_size=2, head_dim=64)
    """

    def __init__(self, path, framework="pt"):
        self.path = Path(path)
        self.framework = framework
        self._stack = ExitStack()
        self._handles = {}
        self._file_of = {}
        try:
            for file, names in self._discover().items():
                handle = self._stack.enter_context(safe_open(str(file), framework=framework))
                self._handles[file] = handle
                for name in names if names is not None else handle.keys():
                    self._file_of[name] = file
        except BaseException:
            self._stack.close()
            raise

    def _discover(self):
        """Map each file to the tensor names it should supply (None means all of them)."""
        if self.path.is_file():
            return {self.path: None}
        if not self.path.is_dir():
            raise FileNotFoundError(f"no checkpoint at {self.path}")
        index = self.path / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            files = {}
            for name, file in weight_map.items():
                files.setdefault(self.path / file, []).append(name)
            return files
        single = self.path / "model.safetensors"
        if single.exists():
            return {single: None}
        raise FileNotFoundError(f"no model.safetensors or model.safetensors.index.json in {self.path}")

    def close(self):
        self._stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def names(self):
        return sorted(self._file_of)

    def _slice(self, name):
        try:
            return self._handles[self._file_of[name]].get_slice(name)
        except KeyError:
            raise KeyError(f"tensor {name!r} is not in this checkpoint") from None

    def shape(self, name):
        """Shape of a tensor. Reads only the file header, not the weights."""
        return tuple(self._slice(name).get_shape())

    def dtype(self, name):
        """safetensors dtype name such as F32 or BF16. Reads only the header."""
        return self._slice(name).get_dtype()

    def nbytes(self, name):
        return prod(self.shape(name)) * DTYPE_BYTES[self.dtype(name)]

    def load(self, name):
        """The whole tensor."""
        return self._handles[self._file_of[name]].get_tensor(name)

    def load_shard(self, name, rank, world_size, head_dim=1):
        """Only this worker's piece of a tensor; the rest of the file is never read."""
        dim = shard_dim(shard_kind(name))
        if dim is None or world_size == 1:
            return self.load(name)
        shape = self.shape(name)
        start, stop = shard_bounds(shape[dim], rank, world_size, shard_unit(name, head_dim))
        piece = self._slice(name)
        return piece[start:stop] if dim == 0 else piece[:, start:stop]

    def shard_nbytes(self, name, world_size, head_dim=1):
        """Bytes one worker holds for a tensor, worked out from the header alone."""
        return prod(shard_shape(name, self.shape(name), world_size, head_dim)) * DTYPE_BYTES[self.dtype(name)]
