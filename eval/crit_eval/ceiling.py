"""The model's arithmetic per token, for the prefill floor, its weight bytes, for the memory
floor of a short prompt, and its KV bytes per token, for the decode floor at long context. Counted once from the safetensors headers and config when the
suite is prepared, so every engine evaluated against one data directory gets the same ceiling.

Prefill is compute bound. Its floor for a prompt of N tokens is

    matmul      N x sum over weight formats of FLOPs per token / that format's GEMM rate
    attention   N(N+1)/2 query-key pairs x FLOPs per pair / the BF16 rate
    recurrent   N x linear-attention FLOPs per token / the BF16 rate (chunked form, approximate)

A projection weight of R x K elements costs 2RK FLOPs per token in the format it is stored in:
BF16/F16/F32 `weight` (bf16), F8_E4M3 `weight` (fp8) or a U8 `weight_packed` with an F8_E4M3
`weight_scale` of one scale per 16 elements (nvfp4, K = 2 x packed columns). Embedding tables
(gathered), vision towers, norms, convolutions and the output head (one row of logits per
prompt) are left out. Engines may compute in other formats; the floor uses the stored ones.

Time to first token cannot beat either floor: the prompt's arithmetic, or reading the weights
once (language-model tensors and the output head, not embeddings, vision or draft layers)
plus writing the prompt's KV, at the device bandwidth.
"""
import os
from pathlib import Path

from crit_core.model import CONFIG_NAME, inspect_model, inspect_safetensors, load_model_config, parse_model_config

from .tensors import DRAFT, tensor_role


def model_config(model):
    """MODEL_DIR's config.json read with Core's rules (crit_core.model.load_model_config); an empty config when
    the model has none. Rules read the text section (`text_config` over the root)."""
    path = Path(model) / CONFIG_NAME
    return load_model_config(path) if os.path.lexists(path) else parse_model_config({})


def weight_shards(model, drafts=False):
    """The safetensors files of MODEL_DIR, validated by crit_core.model: the checkpoint inspect_model resolves
    (model.safetensors.index.json, else model.safetensors, else the only .safetensors file). Without `drafts`,
    files with `mtp` in their name are left out; with it, draft files beside the checkpoint join it. [] when the
    directory holds no safetensors file."""
    model = Path(model)
    if not model.is_dir() or not any(model.glob("*.safetensors*")):
        return []
    shards = list(inspect_model(model).shards)
    if not drafts:
        return [shard for shard in shards if DRAFT not in shard.path.name]
    named = {shard.path.name for shard in shards}
    # Draft layers sometimes ship beside the index in their own file.
    return shards + [inspect_safetensors(path) for path in sorted(model.glob("*mtp*.safetensors"))
                     if path.name not in named]


def _read(role):
    # A pass of the language model reads its layers, experts and head; embedding rows are gathered.
    return role.kind in ("layer", "expert", "head") and not role.draft


def weight_bytes(tensors, file_name=""):
    """Bytes of the tensors one forward pass reads (crit_eval.tensors: layers, experts and the output head, not
    embeddings, other modalities or draft layers). tensors: {name: crit_core.model.TensorInfo} of one file."""
    return sum(t.nbytes for name, t in tensors.items() if _read(tensor_role(name, file_name)))


def matmul_flops(tensors, file_name=""):
    """FLOPs per token by stored weight format, and how many matrices each format has, over one file's
    {name: TensorInfo}. The output head is left out (one row of logits per prompt)."""
    flops, matrices = {}, {}
    for name, tensor in tensors.items():
        role = tensor_role(name, file_name)
        if not _read(role) or role.kind == "head" or len(tensor.shape) != 2:
            continue
        rows, cols = tensor.shape
        fmt = None
        if (name.endswith(".weight_packed") or name.endswith(".weight")) and tensor.dtype == "U8":
            # compressed-tensors (weight_packed) or ModelOpt (weight) NVFP4: packed E2M1 with E4M3 group-16 scales
            scale = tensors.get(name[:-len("weight_packed" if name.endswith("_packed") else "weight")] + "weight_scale")
            if scale and scale.dtype == "F8_E4M3" and scale.shape[-1] * 16 == cols * 2:
                fmt, cols = "nvfp4", cols * 2
        elif name.endswith(".weight"):
            fmt = {"F8_E4M3": "fp8", "BF16": "bf16", "F16": "bf16", "F32": "bf16"}.get(tensor.dtype)
        if fmt:
            flops[fmt] = flops.get(fmt, 0) + 2 * rows * cols
            matrices[fmt] = matrices.get(fmt, 0) + 1
    return flops, matrices


def model_ceiling(model):
    """Per-token arithmetic and KV bytes of the model in MODEL_DIR, or None without safetensors."""
    shards = weight_shards(model)
    if not shards:
        return None
    flops, matrices, read = {}, {}, 0
    for shard in shards:
        tensors = {tensor.name: tensor for tensor in shard.tensors}
        read += weight_bytes(tensors, shard.path.name)
        f, m = matmul_flops(tensors, shard.path.name)
        for key in f:
            flops[key] = flops.get(key, 0) + f[key]
            matrices[key] = matrices.get(key, 0) + m[key]
    config = model_config(model)
    shape = config.attention_shape()
    out = {"matmul_flops_per_token": flops, "matrices": matrices, "weight_bytes": read,
           "attention_layers": None, "attention_flops_per_pair": None, "kv_bytes_per_token": None,
           "recurrent_flops_per_token": 0}
    if shape:
        full, heads, head_dim, kv_heads = shape
        # Scores (QK^T) and weighted values (PV): 2 x 2 x heads x head_dim FLOPs per query-key pair.
        out.update(attention_layers=full, attention_flops_per_pair=4 * heads * head_dim * full,
                   kv_bytes_per_token=2 * kv_heads * head_dim * 2 * full)  # K and V, BF16
    linear = config.layer_types.count("linear_attention")
    attention = config.linear
    hv, dk, dv = ((attention.num_value_heads, attention.key_head_dim, attention.value_head_dim) if attention
                  else (None, None, None))
    if linear and hv and dk and dv:
        chunk = 64  # chunked gated delta rule: state read/update plus intra-chunk scores and values
        out["recurrent_flops_per_token"] = linear * hv * (4 * dk * dv + 2 * chunk * (dk + dv))
    return out


def prefill_floor_ms(ceiling, tflops, prompt_tokens):
    """Floor of one prefill in ms with its parts, or (None, missing formats)."""
    if not ceiling or not tflops:
        return None, ["device GEMM rates (--device)"] if not tflops else ["model arithmetic (re-run prepare)"]
    missing = [f for f in ceiling["matmul_flops_per_token"] if not tflops.get(f)]
    if not tflops.get("bf16"):
        missing.append("bf16")
    if missing:
        return None, [f"{f} GEMM rate" for f in sorted(set(missing))]
    n = prompt_tokens
    matmul = sum(n * flops / (tflops[f] * 1e12) for f, flops in ceiling["matmul_flops_per_token"].items()) * 1000
    pairs = n * (n + 1) / 2
    # Attention at the faster of the BF16 and FP8 rates: engines run Q K^T and P V on FP8 tensor cores within
    # the quality budget, so a BF16-only floor is one they beat.
    attention_rate = max(tflops["bf16"], tflops.get("fp8") or 0)
    attention = pairs * (ceiling.get("attention_flops_per_pair") or 0) / (attention_rate * 1e12) * 1000
    recurrent = n * (ceiling.get("recurrent_flops_per_token") or 0) / (tflops["bf16"] * 1e12) * 1000
    return {"matmul_ms": matmul, "attention_ms": attention, "recurrent_ms": recurrent,
            "total_ms": matmul + attention + recurrent}, []


def ttft_floor_ms(ceiling, tflops, bandwidth_gbs, prompt_tokens, bytes_per_pass=None):
    """Floor of the time to first token: the larger of the prompt's compute floor and one read of
    the weights (bytes_per_pass when the adapter declares it) plus the prompt's KV writes."""
    compute, missing = prefill_floor_ms(ceiling, tflops, prompt_tokens)
    weights = bytes_per_pass or (ceiling or {}).get("weight_bytes")
    if not weights or not bandwidth_gbs:
        missing = missing + ["device bandwidth (--device)" if not bandwidth_gbs else "model weight bytes (re-run prepare)"]
    if missing:
        return None, missing
    memory = (weights + (ceiling.get("kv_bytes_per_token") or 0) * prompt_tokens) / (bandwidth_gbs * 1e9) * 1000
    return {"compute_ms": compute["total_ms"], "memory_ms": memory, "total_ms": max(compute["total_ms"], memory),
            "bound": "compute" if compute["total_ms"] >= memory else "memory"}, []
