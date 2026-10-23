"""A GGUF read as the checkpoint transformers' modeling code expects, for Eval's reference.

GGUF's reference converter (convert_hf_to_gguf.py and its conversion package) renames every tensor of a Hugging Face
checkpoint and, for some architectures, reorders or transforms tensors on the way: Llama's Q and K rows are
permuted for its RoPE layout, Qwen3.5's gated delta net has its value heads tiled instead of grouped, its
zero-centred norms are stored as 1 + w and its decay as -exp(A_log), expert matrices are stacked. The reference
must run the stored model through the official modeling code, so this module undoes exactly those steps:

    HF tensor = inverse(rule, decode(GGUF tensor))

decode is gguf-py's quants.dequantize, the format's own reference decoder, to FP32. Names map through gguf-py's
TensorNameMap for the file's architecture, the table the converter maps with. The inverse steps are layout moves
(exact) and the inverses of the converter's elementwise transforms in FP32 (w - 1, log(-a)), which round once.

An architecture is read only when LAYOUTS has its rules; each rule set is the inverse of the converter class for
that architecture and is proven by a round trip (tests/test_eval_gguf.py: a model converted by the reference
converter comes back tensor for tensor). A Llama-architecture GGUF is assumed converted from the Hugging Face
layout (Q and K permuted); one converted from Mistral's native format is not. Requires numpy, torch and gguf.
"""
import math
import re

from crit_core.model.gguf import TYPE_BY_NAME

BLOCK = re.compile(r"(?:^|\.)layers\.(\d+)\.")


class GgufLayoutError(ValueError):
    """A GGUF the reference cannot read as its Hugging Face checkpoint."""


def _swap_heads(tensor, dim, first, second, head_dim):
    """Reorder dimension `dim` from [first, second, head_dim] to [second, first, head_dim]."""
    shape = list(tensor.shape)
    dim %= len(shape)
    view = tensor.reshape(*shape[:dim], first, second, head_dim, *shape[dim + 1:])
    return view.transpose(dim, dim + 1).reshape(shape)


class Layout:
    """Plain renames: the converter maps the tensor through TensorNameMap and changes nothing else."""

    def __init__(self, gguf, config):
        self.gguf, self.config = gguf, config

    def alias(self, name):
        """The Hugging Face name the converter maps for HF tensor `name` (some rename before mapping)."""
        return name

    def inverse(self, name, tensor):
        """HF tensor `name` from the decoded GGUF tensor (torch FP32, row-major as stored)."""
        return tensor

    def compose(self, name, present):
        """An HF tensor that no GGUF tensor maps to one to one: (GGUF names, combine(decoded tensors)) built from
        the names in `present` (the file's tensor names), or None."""
        return None


class LlamaLayout(Layout):
    """LlamaModel.permute(q, n_head, n_head) and permute(k, n_head, n_kv_head): pairs of rotary rows are interleaved
    per head. Undone by the opposite reshape."""

    def inverse(self, name, tensor):
        heads, kv = self.config.num_attention_heads, getattr(self.config, "num_key_value_heads", None)
        if name.endswith(("q_proj.weight", "q_proj.bias")):
            return self._unpermute(tensor, heads, heads)
        if name.endswith(("k_proj.weight", "k_proj.bias")):
            return self._unpermute(tensor, heads, kv)
        return tensor

    @staticmethod
    def _unpermute(tensor, heads, kv_heads):
        if kv_heads is not None and heads != kv_heads:
            heads = kv_heads
        shape = tensor.shape
        return (tensor.reshape(heads, shape[0] // heads // 2, 2, *shape[1:]).transpose(1, 2).reshape(shape))


class ExpertLayout(Layout):
    """Qwen2MoeModel: per-expert or fused HF experts become stacked ffn_{gate,up,down}_exps; transformers' fused
    gate_up_proj [experts, 2 x width, hidden] is gate then up."""

    def compose(self, name, present):
        match = re.search(r"layers\.(\d+)\.mlp\.experts\.gate_up_proj$", name)
        if not match:
            return None
        fused = f"blk.{match[1]}.ffn_gate_up_exps.weight"
        if fused in present:
            return (fused,), lambda tensor: tensor
        parts = (f"blk.{match[1]}.ffn_gate_exps.weight", f"blk.{match[1]}.ffn_up_exps.weight")
        if not all(part in present for part in parts):
            return None
        import torch
        return parts, lambda gate, up: torch.cat([gate, up], dim=1)


class GatedDeltaNetLayout(ExpertLayout):
    """Qwen3NextModel and _LinearAttentionVReorderBase (Qwen3.5): norms other than the delta net's own are stored as
    1 + w; A_log as -exp(A_log); dt_bias is renamed dt_proj.bias; the convolution kernel is squeezed; and with fewer
    key than value heads, the value heads of in_proj_qkv's V rows, in_proj_z, in_proj_a/b, A_log, dt_bias, the
    convolution's V channels and out_proj's columns are reordered from grouped by key head to tiled."""

    def __init__(self, gguf, config):
        super().__init__(gguf, config)
        if any(key.startswith("prism.hadamard.") for key in gguf.metadata):
            raise GgufLayoutError("the GGUF's delta-net projections are Hadamard-folded "
                                  "(prism.hadamard.*); the reference cannot unfold them")
        c = config
        self.k_heads, self.v_heads = c.linear_num_key_heads, c.linear_num_value_heads
        self.k_dim, self.v_dim = c.linear_key_head_dim, c.linear_value_head_dim
        self.reordered = bool(self.k_heads and self.v_heads and self.k_heads != self.v_heads)

    def alias(self, name):
        return name[:-len(".dt_bias")] + ".dt_proj.bias" if name.endswith(".dt_bias") else name

    def _tiled_to_grouped(self, tensor, dim, head_dim):
        return _swap_heads(tensor, dim, self.v_heads // self.k_heads, self.k_heads, head_dim)

    def inverse(self, name, tensor):
        import torch
        if name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
            return tensor - 1
        if ".linear_attn." not in name:
            return tensor
        if name.endswith(".A_log"):
            tensor = torch.log(-tensor)
        if not self.reordered:
            return tensor
        if name.endswith(".in_proj_qkv.weight"):
            qk = 2 * self.k_heads * self.k_dim
            return torch.cat([tensor[:qk], self._tiled_to_grouped(tensor[qk:], 0, self.v_dim)])
        if name.endswith(".in_proj_z.weight"):
            return self._tiled_to_grouped(tensor, 0, self.v_dim)
        if name.endswith((".in_proj_a.weight", ".in_proj_b.weight", ".A_log", ".dt_bias")):
            return self._tiled_to_grouped(tensor, 0, 1)
        if name.endswith(".conv1d.weight"):
            qk = 2 * self.k_heads * self.k_dim
            return torch.cat([tensor[:qk], self._tiled_to_grouped(tensor[qk:], 0, self.v_dim)])
        if name.endswith(".out_proj.weight"):
            return self._tiled_to_grouped(tensor, 1, self.v_dim)
        return tensor


# GGUF general.architecture -> the inverse of the converter class that writes it.
LAYOUTS = {
    "llama": LlamaLayout,
    "qwen2": Layout,
    "qwen3": Layout,
    "qwen3moe": ExpertLayout,
    "qwen35": GatedDeltaNetLayout,
    "qwen35moe": GatedDeltaNetLayout,
}


class GgufCheckpoint:
    """The reference's checkpoint interface (has, raw, dense, prefix) over a GgufModel (a ModelSource's weights,
    crit_core.model.open_model), keyed by Hugging Face names under "model." plus "lm_head.weight". config:
    transformers' text config."""

    def __init__(self, gguf, config):
        from importlib.metadata import version
        import gguf as gguf_py
        self.gguf, self.config = gguf, config
        self.version = version("gguf")
        architecture = gguf.architecture
        rules = LAYOUTS.get(architecture)
        if rules is None:
            raise GgufLayoutError(f"GGUF architecture {architecture!r} has no layout rules in crit_eval.gguf_layout "
                                  f"(supported: {', '.join(sorted(LAYOUTS))})")
        enums = {name: arch for arch, name in gguf_py.MODEL_ARCH_NAMES.items()}
        if architecture not in enums:
            raise GgufLayoutError(f"gguf {self.version} does not know architecture {architecture!r}")
        self.layout = rules(gguf, config)
        self.names = gguf_py.get_tensor_name_map(enums[architecture], gguf.get(".block_count"))
        self.quants = gguf_py.quants
        self.types = gguf_py.GGMLQuantizationType

    @property
    def description(self):
        """What the reference's implementation record adds for a GGUF."""
        return (f"GGUF decoded by gguf {self.version} with the reference converter's layout undone "
                f"({self.gguf.architecture}); ")

    def prefix(self, names=()):
        return "model."

    def _gguf_name(self, name):
        """The GGUF tensor HF tensor `name` maps to one to one, or None."""
        if name == "lm_head.weight":
            return "output.weight" if "output.weight" in self.gguf.tensors else None
        mapped = self.names.get_name(self.layout.alias(name), try_suffixes=(".weight", ".bias"))
        if mapped is None:
            return None
        for candidate in (mapped, mapped + ".weight"):
            if candidate in self.gguf.tensors:
                return candidate
        return None

    def decode(self, gguf_name):
        """GGUF tensor `gguf_name` decoded to FP32 (torch, CPU, row-major), or None when the file lacks it."""
        import numpy as np
        import torch
        tensor = self.gguf.tensors.get(gguf_name)
        if tensor is None:
            return None
        ggml = TYPE_BY_NAME[tensor.dtype]
        data = np.memmap(tensor.shard, dtype=np.uint8, mode="r", offset=tensor.data_begin, shape=(tensor.nbytes,))
        rows = math.prod(tensor.shape[:-1])
        data = data.reshape(rows, tensor.nbytes // rows)
        if ggml.name in ("BF16", "F32", "F16") or ggml.quantized:
            try:
                values = self.quants.dequantize(np.ascontiguousarray(data), self.types(ggml.id))
            except NotImplementedError:
                raise GgufLayoutError(f"gguf {self.version} cannot decode {ggml.name} ({gguf_name})") from None
        else:
            raise GgufLayoutError(f"{gguf_name}: {ggml.name} tensors are not weights the reference decodes")
        return torch.from_numpy(np.asarray(values, dtype=np.float32).reshape(tensor.shape).copy())

    def has(self, key):
        return self._gguf_name(key) is not None or self.layout.compose(key, self.gguf.tensors) is not None

    def raw(self, key):
        """HF tensor `key` in FP32 with its Hugging Face layout (a squeezed GGUF kernel takes the shape back in
        dense's caller); KeyError when the GGUF holds nothing for it."""
        gguf_name = self._gguf_name(key)
        if gguf_name is not None:
            return self.layout.inverse(key, self.decode(gguf_name))
        composed = self.layout.compose(key, self.gguf.tensors)
        if composed is None:
            raise KeyError(f"the GGUF holds no tensor for {key!r}")
        names, combine = composed
        return self.layout.inverse(key, combine(*(self.decode(name) for name in names)))

    def dense(self, base, device):
        return self.raw(base + ".weight").to(device)

    def embedding(self, base):
        return self.raw(base + ".weight")

    def fit(self, key, tensor, shape):
        """The tensor in the module's shape: GGUF drops size-1 dimensions (a convolution kernel [C, 1, K] is stored
        [C, K]), so only those may differ."""
        shape = tuple(shape)
        if tuple(tensor.shape) == shape:
            return tensor
        if [d for d in tensor.shape if d != 1] != [d for d in shape if d != 1]:
            raise GgufLayoutError(f"{key}: the GGUF gives shape {list(tensor.shape)}, the model expects {list(shape)}")
        return tensor.reshape(shape)
