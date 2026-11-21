"""Small array operations that behave the same on numpy arrays and torch tensors.

The model code is written once and runs on either: numpy in tests and on machines without
torch, torch tensors on the real grid. Only the few operations whose names or semantics
differ between the two libraries are wrapped here.
"""
import numpy as np


def is_torch(x):
    return type(x).__module__.split(".")[0] == "torch"


def as_like(x, array):
    """Turn a numpy array into the same kind and dtype as `x` (numpy array or torch tensor)."""
    if is_torch(x):
        import torch

        return torch.from_numpy(np.ascontiguousarray(array)).to(dtype=x.dtype)
    return np.asarray(array).astype(x.dtype, copy=False)


def silu(x):
    """x * sigmoid(x), the activation of Llama and Qwen MLPs."""
    if is_torch(x):
        import torch

        return torch.nn.functional.silu(x)
    return x * (0.5 * (1.0 + np.tanh(0.5 * x)))  # sigmoid written so it cannot overflow


def softmax(x):
    """Softmax over the last dimension. A row of -inf entries is allowed as long as one is finite."""
    if is_torch(x):
        import torch

        return torch.softmax(x, dim=-1)
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def cat(parts, axis=-1):
    """Join arrays or tensors along an axis."""
    if is_torch(parts[0]):
        import torch

        return torch.cat(list(parts), dim=axis)
    return np.concatenate(parts, axis=axis)


def repeat_heads(x, repeats):
    """[batch, kv_heads, seq, dim] -> [batch, kv_heads * repeats, seq, dim].

    Each kv head is repeated next to itself (head 0, 0, 1, 1, ...), which is how grouped-query
    attention lines up query heads with their shared key/value head.
    """
    if repeats == 1:
        return x
    if is_torch(x):
        return x.repeat_interleave(repeats, dim=1)
    return np.repeat(x, repeats, axis=1)


def cast_like(x, reference):
    """`x` converted to the dtype of `reference` (a cache may store a smaller dtype than the math uses)."""
    if is_torch(x):
        return x.to(reference.dtype)
    return x.astype(reference.dtype, copy=False)


ELEMENT_BYTES = {"float64": 8, "float32": 4, "float16": 2, "bfloat16": 2}


def element_bytes(dtype):
    """Bytes per element of a dtype given by name, e.g. "bfloat16"."""
    try:
        return ELEMENT_BYTES[dtype]
    except KeyError:
        raise ValueError(f"unsupported dtype {dtype!r}; use one of {sorted(ELEMENT_BYTES)}") from None


def zeros(shape, dtype="float32", backend="numpy"):
    """A zero-filled array (backend "numpy") or tensor (backend "torch") of the named dtype."""
    element_bytes(dtype)
    if backend == "torch":
        import torch

        return torch.zeros(shape, dtype=getattr(torch, dtype))
    if backend != "numpy":
        raise ValueError(f"backend must be 'numpy' or 'torch', not {backend!r}")
    if dtype == "bfloat16":
        raise ValueError("numpy has no bfloat16; use the torch backend")
    return np.zeros(shape, dtype=dtype)
