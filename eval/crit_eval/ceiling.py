"""The model's arithmetic per token, for the prefill floor, its weight bytes, for the memory
floor of a short prompt, and its KV bytes per token, for the decode floor at long context. Counted once from the weight headers and config when the
suite is prepared, so every engine evaluated against one data directory gets the same ceiling.

Prefill is compute bound. Its floor for a prompt of N tokens is

    matmul      N x sum over weight formats of FLOPs per token / that format's GEMM rate
    attention   N(N+1)/2 query-key pairs x FLOPs per pair / the BF16 rate
    recurrent   N x linear-attention FLOPs per token / the BF16 rate (chunked form, approximate)

A projection weight of R x K elements costs 2RK FLOPs per token at the rate of the format it is stored in
(crit_eval.schemes): BF16/F16/F32 (bf16), FP8 (fp8), NVFP4 (nvfp4), weight-only integer codes (AWQ, GPTQ,
MLX, bitsandbytes NF4, compressed-tensors pack-quantized: bf16, as engines widen them for the GEMM) and 8-bit
integer weights with per-row scales (bitsandbytes LLM.int8, compressed-tensors int-quantized: fp8, the int8 rate).
A GGUF's ggml block types
count at the fp8 rate (engines multiply their integer codes on 8-bit tensor cores, which run int8 and FP8 at
the same rate), its MXFP4 and NVFP4 at the nvfp4 rate and its F32/F16/BF16 at bf16. Embedding tables
(gathered), vision towers, norms, convolutions and the output head (one row of logits per
prompt) are left out. Engines may compute in other formats; the floor uses the stored ones.

Time to first token cannot beat either floor: the prompt's arithmetic, or reading the weights
once (language-model tensors and the output head, not embeddings, vision or draft layers)
plus writing the prompt's KV, at the device bandwidth.
"""
import math
import os

from crit_core.model import open_model, parse_model_config
from crit_core.model.gguf import TYPE_BY_NAME, TensorInfo as GgufTensor

from .schemes import PRIMARY, quantization_of, refuse_unread, stored
from .tensors import tensor_role


def _source(model):
    # A model path that does not exist reads as a model without files, as a folder without them does.
    return open_model(model) if os.path.lexists(model) else None


def model_config(model):
    """The model's config.json read with Core's rules (crit_core.model.ModelSource.config): the folder's own, or
    the one a GGUF's metadata derives; an empty config when the model has none. Rules read the text section
    (`text_config` over the root)."""
    source = _source(model)
    return source.config() if source else parse_model_config({})


def weight_files(model, drafts=False):
    """The weight files of the model (a folder or a weight file, crit_core.model.open_model) with the draft blocks
    their tensors' roles need: (files, draft_blocks). Without `drafts`, draft-layer files are left out; with it,
    those beside the checkpoint join it. ([], no blocks) for a folder without weights."""
    source = _source(model)
    return (list(source.weight_files(drafts)), source.draft_blocks) if source else ([], frozenset())


def weight_shards(model, drafts=False):
    """The safetensors files of the model, validated by crit_core.model (weight_files of a safetensors model);
    [] for any other model."""
    source = _source(model)
    return list(source.weight_files(drafts)) if source and source.format == "safetensors" else []


def _read(role):
    # A pass of the language model reads its layers, experts and head; embedding rows are gathered.
    return role.kind in ("layer", "expert", "head") and not role.draft


def weight_bytes(tensors, file_name="", draft_blocks=()):
    """Bytes of the tensors one forward pass reads (crit_eval.tensors: layers, experts and the output head, not
    embeddings, other modalities or draft layers). tensors: {name: crit_core.model.TensorInfo} of one file."""
    return sum(t.nbytes for name, t in tensors.items() if _read(tensor_role(name, file_name, draft_blocks)))


def _ggml_format(dtype):
    """A GGUF tensor's GEMM rate key (see the module docstring); None for the integer types."""
    if dtype in ("F32", "F16", "BF16"):
        return "bf16"
    if dtype in ("MXFP4", "NVFP4"):
        return "nvfp4"
    return "fp8" if TYPE_BY_NAME[dtype].quantized else None


def matmul_flops(tensors, file_name="", draft_blocks=(), quantization=None):
    """FLOPs per token by stored weight format, and how many matrices each format has, over one file's
    {name: TensorInfo}. The output head is left out (one row of logits per prompt). A 3-D expert tensor counts as
    one matrix per expert, as the per-expert matrices of a checkpoint that stores them apart do. quantization: the
    config's quantization_config (or MLX's quantization), which names the scheme of packed integer weights
    (crit_eval.schemes)."""
    flops, matrices, seen = {}, {}, set()
    for name, tensor in tensors.items():
        role = tensor_role(name, file_name, draft_blocks)
        stacked = len(tensor.shape) == 3 and role.kind == "expert"
        if not _read(role) or role.kind == "head":
            continue
        count = tensor.shape[0] if stacked else 1
        if isinstance(tensor, GgufTensor):
            if "conv" in name or (len(tensor.shape) != 2 and not stacked):   # GGUF squeezes conv kernels to 2-D
                continue
            fmt = _ggml_format(tensor.dtype) if name.endswith(".weight") else None
            elements = math.prod(tensor.shape)
        else:
            base = next((name[:-len(suffix)] for suffix in PRIMARY if name.endswith(suffix)), None)
            found = None if base is None or base in seen else stored(tensors, base, quantization)
            # Plain and FP8 weights of other ranks are norms, biases and convolution kernels, not matrices.
            if found is None or (found[0] in ("plain", "fp8") and len(tensor.shape) != 2 and not stacked):
                continue
            seen.add(base)
            _, fmt, elements = found
        if fmt:
            flops[fmt] = flops.get(fmt, 0) + 2 * elements
            matrices[fmt] = matrices.get(fmt, 0) + count
    return flops, matrices


def model_ceiling(model):
    """Per-token arithmetic and KV bytes of the model, or None for a model without weight files."""
    shards, drafts = weight_files(model)
    if not shards:
        return None
    config = model_config(model)
    quantization = quantization_of(config)
    flops, matrices, read = {}, {}, 0
    for shard in shards:
        tensors = {tensor.name: tensor for tensor in shard.tensors}
        refuse_unread(quantization, tensors)
        read += weight_bytes(tensors, shard.path.name, drafts)
        f, m = matmul_flops(tensors, shard.path.name, drafts, quantization)
        for key in f:
            flops[key] = flops.get(key, 0) + f[key]
            matrices[key] = matrices.get(key, 0) + m[key]
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
