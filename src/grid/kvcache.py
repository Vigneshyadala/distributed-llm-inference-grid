"""A paged key/value cache: the memory of a worker's attention heads for all running sequences.

Generating a token needs the keys and values of every earlier token. Keeping them in one big
array per sequence wastes memory (the final length is unknown) and fragments it as sequences
come and go. A paged cache, as in vLLM, instead owns one pool of equal-sized BLOCKS, each
holding the keys and values of `block_size` tokens. A sequence owns a list of block numbers
(its block table) and takes one more block only when the last one is full. A finished
sequence gives its blocks back, and the next sequence can use them at once.

    pool of 8 blocks of 4 tokens, A has 6 tokens, B has 3:     A A b . . . . .
                                                               (A: 2 blocks, B: 1 block)

Each worker has its own cache holding only ITS key/value heads (see grid.blocks). Blocks are
always handed out lowest number first, so workers that perform the same operations end up with
the same block tables. The master can therefore decide the layout once for the whole grid.

The whole pool is allocated up front, so how much memory the cache needs is known and fixed
before any request arrives, and a full cache is reported as `OutOfBlocks` (the scheduler can
then wait or refuse) instead of the machine running out of RAM in the middle of a request.
"""
import heapq
from math import ceil

from grid.ops import cat, element_bytes, zeros


class CacheError(RuntimeError):
    """The cache was used incorrectly (unknown or duplicate sequence, bad shape, ...)."""


class OutOfBlocks(CacheError):
    """There are not enough free blocks. Nothing was allocated."""


def bytes_per_token(layers, kv_heads, head_dim, dtype="float32"):
    """Memory one token needs in the cache: keys and values, in every layer."""
    return 2 * layers * kv_heads * head_dim * element_bytes(dtype)


def blocks_for_budget(budget_bytes, layers, kv_heads, head_dim, block_size, dtype="float32"):
    """How many blocks fit in `budget_bytes` of memory."""
    return int(budget_bytes // (bytes_per_token(layers, kv_heads, head_dim, dtype) * block_size))


class _Sequence:
    def __init__(self, layers):
        self.blocks = []
        self.lengths = [0] * layers  # tokens stored per layer; equal between decoding steps


class LayerCache:
    """One sequence's view of one layer: what `ParallelAttention` is given as `cache`."""

    def __init__(self, cache, seq_id, layer):
        self._cache, self.seq_id, self.layer = cache, seq_id, layer

    @property
    def length(self):
        """Tokens of this sequence already stored for this layer."""
        return self._cache.layer_length(self.seq_id, self.layer)

    def extend(self, keys, values):
        """Store new keys/values and return all of them, [1, kv_heads, total_tokens, head_dim]."""
        return self._cache.extend(self.seq_id, self.layer, keys, values)


class PagedKVCache:
    """Blocks of keys and values for `layers` layers, shared by many sequences."""

    def __init__(self, layers, kv_heads, head_dim, block_size, num_blocks, dtype="float32", backend="numpy"):
        for name, value in (("layers", layers), ("kv_heads", kv_heads), ("head_dim", head_dim),
                            ("block_size", block_size), ("num_blocks", num_blocks)):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive whole number, not {value!r}")
        self.layers, self.kv_heads, self.head_dim = layers, kv_heads, head_dim
        self.block_size, self.num_blocks = block_size, num_blocks
        self.dtype, self.backend = dtype, backend
        shape = (num_blocks, kv_heads, block_size, head_dim)
        self._keys = [zeros(shape, dtype, backend) for _ in range(layers)]
        self._values = [zeros(shape, dtype, backend) for _ in range(layers)]
        self._free = list(range(num_blocks))  # a heap: the lowest free block number comes out first
        heapq.heapify(self._free)
        self._sequences = {}

    # ------------------------------------------------------------------ sequences

    def add_sequence(self, seq_id):
        if seq_id in self._sequences:
            raise CacheError(f"sequence {seq_id!r} already exists")
        self._sequences[seq_id] = _Sequence(self.layers)

    def free_sequence(self, seq_id):
        """Give all of a sequence's blocks back to the pool. Returns how many were freed."""
        seq = self._sequence(seq_id)
        for block in seq.blocks:
            heapq.heappush(self._free, block)
        del self._sequences[seq_id]
        return len(seq.blocks)

    def has_sequence(self, seq_id):
        return seq_id in self._sequences

    def sequence_ids(self):
        return list(self._sequences)

    def length(self, seq_id):
        """Tokens stored for a sequence."""
        return max(self._sequence(seq_id).lengths)

    def layer_length(self, seq_id, layer):
        return self._sequence(seq_id).lengths[self._layer(layer)]

    def blocks_of(self, seq_id):
        """The sequence's block table: block numbers in token order."""
        return list(self._sequence(seq_id).blocks)

    def view(self, seq_id, layer):
        """What to pass to `ParallelAttention` as `cache=` for this sequence and layer."""
        self._sequence(seq_id)
        return LayerCache(self, seq_id, self._layer(layer))

    # ----------------------------------------------------------------- allocation

    def blocks_for_tokens(self, tokens):
        return ceil(tokens / self.block_size)

    @property
    def free_blocks(self):
        return len(self._free)

    def can_fit(self, tokens, seq_id=None):
        """Would `tokens` more tokens fit? For an existing sequence, counts what it already holds."""
        have = 0
        if seq_id is not None:
            seq = self._sequence(seq_id)
            have = len(seq.blocks)
            tokens += max(seq.lengths)
        return self.blocks_for_tokens(tokens) - have <= self.free_blocks

    def _ensure(self, seq, total_tokens):
        """Give `seq` enough blocks for `total_tokens` tokens, all or nothing."""
        missing = self.blocks_for_tokens(total_tokens) - len(seq.blocks)
        if missing <= 0:
            return
        if missing > len(self._free):
            raise OutOfBlocks(
                f"need {missing} more blocks of {self.block_size} tokens, only {len(self._free)} are free")
        seq.blocks.extend(heapq.heappop(self._free) for _ in range(missing))

    # ------------------------------------------------------------ writing and reading

    def extend(self, seq_id, layer, keys, values):
        """Append new tokens' keys and values for one layer and return everything stored so far.

        keys and values are [1, kv_heads, new_tokens, head_dim]; the result has the same layout
        with all of the sequence's tokens, in the cache's dtype.
        """
        seq = self._sequence(seq_id)
        layer = self._layer(layer)
        for name, array in (("keys", keys), ("values", values)):
            shape = tuple(array.shape)
            if len(shape) != 4 or (shape[0], shape[1], shape[3]) != (1, self.kv_heads, self.head_dim):
                raise CacheError(f"{name} must be [1, {self.kv_heads}, tokens, {self.head_dim}], got {shape}")
        new = keys.shape[2]
        if values.shape[2] != new:
            raise CacheError("keys and values must cover the same number of tokens")
        if new < 1:
            raise CacheError("nothing to store: at least one token is needed")

        start = seq.lengths[layer]
        self._ensure(seq, start + new)
        written = 0
        while written < new:
            position = start + written
            block, offset = seq.blocks[position // self.block_size], position % self.block_size
            count = min(self.block_size - offset, new - written)
            self._keys[layer][block, :, offset:offset + count, :] = keys[0, :, written:written + count, :]
            self._values[layer][block, :, offset:offset + count, :] = values[0, :, written:written + count, :]
            written += count
        seq.lengths[layer] = start + new
        return self._gather(seq, layer, start + new)

    def _gather(self, seq, layer, total):
        key_parts, value_parts, remaining = [], [], total
        for block in seq.blocks:
            if remaining <= 0:
                break
            count = min(self.block_size, remaining)
            key_parts.append(self._keys[layer][block, :, :count, :])
            value_parts.append(self._values[layer][block, :, :count, :])
            remaining -= count
        return cat(key_parts, axis=1)[None], cat(value_parts, axis=1)[None]

    # ------------------------------------------------------------------ accounting

    @property
    def bytes_per_token(self):
        return bytes_per_token(self.layers, self.kv_heads, self.head_dim, self.dtype)

    def stats(self):
        """Numbers for the memory ledger: blocks, tokens, bytes, and how much allocated room is unused."""
        used = self.num_blocks - len(self._free)
        tokens = sum(max(seq.lengths) for seq in self._sequences.values())
        slots = used * self.block_size
        per_block = self.bytes_per_token * self.block_size
        return {
            "block_size": self.block_size, "blocks_total": self.num_blocks,
            "blocks_used": used, "blocks_free": len(self._free),
            "sequences": len(self._sequences), "tokens_stored": tokens,
            "slots_allocated": slots, "wasted_slots": slots - tokens,
            "utilization": tokens / slots if slots else 1.0,
            "bytes_per_token": self.bytes_per_token,
            "bytes_total": self.num_blocks * per_block, "bytes_used": used * per_block,
        }

    def block_map(self):
        """One entry per block: None if free, else {"seq": id, "tokens": tokens held in this block}."""
        owners = [None] * self.num_blocks
        for seq_id, seq in self._sequences.items():
            length = max(seq.lengths)
            for index, block in enumerate(seq.blocks):
                owners[block] = {"seq": seq_id, "tokens": min(self.block_size, max(0, length - index * self.block_size))}
        return owners

    # --------------------------------------------------------------------- checks

    def _sequence(self, seq_id):
        try:
            return self._sequences[seq_id]
        except KeyError:
            raise CacheError(f"unknown sequence {seq_id!r}") from None

    def _layer(self, layer):
        if not 0 <= layer < self.layers:
            raise CacheError(f"layer {layer} does not exist (the cache has {self.layers} layers)")
        return layer
