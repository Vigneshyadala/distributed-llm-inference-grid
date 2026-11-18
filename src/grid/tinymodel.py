"""A tiny Llama-style checkpoint with random weights, for tests and demos.

Tensor names and shapes follow the Hugging Face Llama/Qwen layout, so the sharding code
can be exercised without downloading a real model.
"""
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


def tiny_config(hidden=64, heads=4, kv_heads=2, layers=2, intermediate=128, vocab=96):
    if hidden % heads:
        raise ValueError("hidden must be divisible by heads")
    return {
        "hidden": hidden, "heads": heads, "kv_heads": kv_heads, "layers": layers,
        "intermediate": intermediate, "vocab": vocab, "head_dim": hidden // heads,
    }


def tiny_tensors(config, seed=0):
    rng = np.random.default_rng(seed)
    h, hd = config["hidden"], config["head_dim"]
    q_rows, kv_rows = config["heads"] * hd, config["kv_heads"] * hd
    inter, vocab = config["intermediate"], config["vocab"]

    def rand(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    tensors = {"model.embed_tokens.weight": rand(vocab, h)}
    for i in range(config["layers"]):
        p = f"model.layers.{i}."
        tensors.update({
            p + "input_layernorm.weight": rand(h),
            p + "self_attn.q_proj.weight": rand(q_rows, h),
            p + "self_attn.q_proj.bias": rand(q_rows),
            p + "self_attn.k_proj.weight": rand(kv_rows, h),
            p + "self_attn.k_proj.bias": rand(kv_rows),
            p + "self_attn.v_proj.weight": rand(kv_rows, h),
            p + "self_attn.v_proj.bias": rand(kv_rows),
            p + "self_attn.o_proj.weight": rand(h, q_rows),
            p + "post_attention_layernorm.weight": rand(h),
            p + "mlp.gate_proj.weight": rand(inter, h),
            p + "mlp.up_proj.weight": rand(inter, h),
            p + "mlp.down_proj.weight": rand(h, inter),
        })
    tensors["model.norm.weight"] = rand(h)
    tensors["lm_head.weight"] = rand(vocab, h)
    return tensors


def write_tiny_checkpoint(directory, files=1, seed=0, **config_overrides):
    """Write a tiny checkpoint into `directory` and return (config, tensors).

    files=1 writes model.safetensors; files>1 splits the tensors over several files
    with a model.safetensors.index.json, like larger real models.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = tiny_config(**config_overrides)
    tensors = tiny_tensors(config, seed)
    if files == 1:
        save_file(tensors, str(directory / "model.safetensors"))
        return config, tensors
    names = sorted(tensors)
    weight_map = {}
    for index in range(files):
        file = f"model-{index + 1:05d}-of-{files:05d}.safetensors"
        chunk = {name: tensors[name] for name in names[index::files]}
        save_file(chunk, str(directory / file))
        weight_map.update({name: file for name in chunk})
    total = sum(array.nbytes for array in tensors.values())
    index_file = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index_file, indent=2), encoding="utf-8")
    return config, tensors
