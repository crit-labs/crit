"""How a checkpoint keyed by Hugging Face names stores its weight matrices, decided in one place for crit-eval's
ceiling and reference and for engines that load the matrices as stored. Header-only: no tensor bytes, no torch.

A matrix `base` (a module path such as model.layers.0.mlp.down_proj) is stored as one scheme, recognised from the
tensors under `base.` and the model's quantization config (quantization_config, or MLX's quantization):

    plain         base.weight in BF16, F16 or F32
    fp8           base.weight F8_E4M3 with base.weight_scale per tensor, per row or per block (compressed-tensors
                  block_structure), or base.weight_scale_inv per block (weight_block_size: DeepSeek and Qwen FP8)
    nvfp4         E2M1 pairs (low nibble first) with E4M3 group-16 scales and an FP32 global factor:
                  compressed-tensors (U8 base.weight_packed, base.weight_global_scale as a divisor) or ModelOpt
                  (U8 base.weight, base.weight_scale_2 as a multiplier)
    int           compressed-tensors pack-quantized (I32 base.weight_packed) and int-quantized (I8 base.weight with
                  base.weight_scale)
    awq           AutoAWQ GEMM (base.qweight, base.qzeros, base.scales)
    gptq          GPTQ (base.qweight, base.qzeros, base.scales, base.g_idx)
    mlx           MLX (U32 base.weight, base.scales per group; affine, mxfp4, nvfp4, mxfp8)
    bnb4          bitsandbytes NF4/FP4 (U8 base.weight with base.weight.absmax)
    bnb8          bitsandbytes LLM.int8 (I8 base.weight, base.SCB per row)
    nbits         ONNX Runtime's MatMulNBits (base.nbits_weight, as crit_core.model.onnx names a node's inputs)

stored(tensors, base, quantization) gives the scheme's name, its GEMM rate key for the prefill floor and the
matrix's element count. crit-eval decodes every scheme for its reference (crit_eval.limits.schemes); matrix_parts names the
tensors and scale conventions of the schemes an engine runs as stored (nvfp4, fp8, plain), and layer_prefix finds
the decoder layers. A new scheme is one entry in RECOGNISERS.
"""
import math
import re
from dataclasses import dataclass


class SchemeError(ValueError):
    """A matrix stored in a way crit does not decode, or whose tensors disagree with its scheme."""


def quantization_of(config):
    """The quantization config of a crit_core.model.ModelConfig as plain data: quantization_config, else MLX's
    quantization (mlx-lm writes both, alike), else {}."""
    root = config.to_dict()
    return root.get("quantization_config") or root.get("quantization") or {}


# Schemes whose only decoders are exllama's CUDA kernels, so the reference has no CPU dequantization to match, and
# the tensor each stores per matrix in place of .weight.
UNREAD = {"exl2": ".q_weight", "exl3": ".trellis"}


def refuse_unread(quantization, names=()):
    """Raise SchemeError for a checkpoint in a scheme of UNREAD, named by its quantization config or by its tensor
    names, so that it is refused rather than read without its matrices."""
    method = quant_method(quantization)
    if method not in UNREAD:
        method = next((m for m, suffix in UNREAD.items() for name in names if name.endswith(suffix)), None)
    if method:
        raise SchemeError(f"{method.upper()} weights are not read: exllama decodes them only in CUDA kernels, so "
                          "there is no CPU dequantization to check crit's reference against; use the base model or "
                          "another quantization of it")


def quant_method(quantization):
    return str((quantization or {}).get("quant_method") or "").lower()


def _shape(tensors, name):
    tensor = tensors.get(name)
    return None if tensor is None else tuple(tensor.shape)


def _dtype(tensors, name):
    tensor = tensors.get(name)
    return None if tensor is None else tensor.dtype


def mlx_settings(quantization, base):
    """MLX's (bits, group_size, mode) for module `base`, None when it is not quantized."""
    override = (quantization or {}).get(base)
    if override is False:
        return None
    source = override if isinstance(override, dict) else quantization or {}
    return int(source.get("bits", 4)), int(source.get("group_size", 64)), str(source.get("mode", "affine"))


def ct_group(quantization, base, linear=False):
    """The compressed-tensors config group with weights whose targets name matrix `base` (its module name, or `re:`
    a pattern matched from its start, as compressed-tensors matches them), or None; with `linear`, else the first that
    targets the class Linear (every linear matrix the config does not ignore)."""
    groups = [group for group in ((quantization or {}).get("config_groups") or {}).values()
              if base is not None and isinstance(group, dict) and group.get("weights")]
    for group in groups:
        if any(target == base or target.startswith("re:") and re.match(target[3:], base)
               for target in group.get("targets") or ()):
            return group
    return next((group for group in groups if "Linear" in (group.get("targets") or ())), None) if linear else None


def ct_weight_args(quantization, base=None):
    """compressed-tensors' weight quantization args (num_bits, strategy, group_size, symmetric) for matrix `base`:
    of the config groups with weights, the one whose targets name it (ct_group), else the first. A mixed-precision
    checkpoint has a group per scheme (DeepSeek V4 W4A16-FP8: FP8 blocks for attention, packed int4 for the
    experts)."""
    group = ct_group(quantization, base)
    if group is not None:
        return group["weights"]
    groups = [group["weights"] for group in ((quantization or {}).get("config_groups") or {}).values()
              if isinstance(group, dict) and group.get("weights")]
    return groups[0] if groups else {}


def _plain_stored(tensors, base, quantization):
    rate = {"BF16": "bf16", "F16": "bf16", "F32": "bf16"}.get(_dtype(tensors, base + ".weight"))
    return (rate, math.prod(_shape(tensors, base + ".weight"))) if rate else None


def _fp8_stored(tensors, base, quantization):
    if _dtype(tensors, base + ".weight") not in ("F8_E4M3", "F8_E5M2"):
        return None
    return "fp8", math.prod(_shape(tensors, base + ".weight"))


def _nvfp4_stored(tensors, base, quantization):
    for packed in (".weight_packed", ".weight"):
        shape, scale = _shape(tensors, base + packed), _shape(tensors, base + ".weight_scale")
        if (_dtype(tensors, base + packed) == "U8" and _dtype(tensors, base + ".weight_scale") == "F8_E4M3"
                and scale[-1] * 16 == shape[-1] * 2):
            return "nvfp4", math.prod(shape) * 2
    return None


def _ct_int_stored(tensors, base, quantization):
    if quant_method(quantization) != "compressed-tensors":
        return None
    if _dtype(tensors, base + ".weight_packed") == "I32":
        bits = int(ct_weight_args(quantization, base).get("num_bits", 4))
        rows, words = _shape(tensors, base + ".weight_packed")[-2:]
        experts = math.prod(_shape(tensors, base + ".weight_packed")[:-2])
        return "bf16", experts * rows * (words * 32 // bits)
    if _dtype(tensors, base + ".weight") == "I8" and base + ".weight_scale" in tensors:
        return "fp8", math.prod(_shape(tensors, base + ".weight"))
    return None


def _awq_stored(tensors, base, quantization):
    if quant_method(quantization) != "awq" or base + ".qweight" not in tensors:
        return None
    cols, packed = _shape(tensors, base + ".qweight")
    return "bf16", cols * packed * 8


def gptq_bits(quantization):
    return int(quantization.get("bits", 4))


def _gptq_stored(tensors, base, quantization):
    if quant_method(quantization) != "gptq" or base + ".qweight" not in tensors:
        return None
    packed, rows = _shape(tensors, base + ".qweight")
    cols = (_shape(tensors, base + ".g_idx") or (packed * 32 // gptq_bits(quantization),))[0]
    return "bf16", rows * cols


def _mlx_stored(tensors, base, quantization):
    if _dtype(tensors, base + ".weight") != "U32" or base + ".scales" not in tensors:
        return None
    found = mlx_settings(quantization, base)
    if found is None:
        return None
    bits, _, mode = found
    rows, words = _shape(tensors, base + ".weight")[-2:]
    elements = math.prod(_shape(tensors, base + ".weight")[:-2]) * rows * words * 32 // bits
    return {"affine": "bf16", "mxfp4": "nvfp4", "nvfp4": "nvfp4", "mxfp8": "fp8"}.get(mode, "bf16"), elements


def _bnb4_stored(tensors, base, quantization):
    if _dtype(tensors, base + ".weight") != "U8" or base + ".weight.absmax" not in tensors:
        return None
    return "bf16", math.prod(_shape(tensors, base + ".weight")) * 2


def _bnb8_stored(tensors, base, quantization):
    if _dtype(tensors, base + ".weight") != "I8" or base + ".SCB" not in tensors:
        return None
    return "fp8", math.prod(_shape(tensors, base + ".weight"))


def _nbits_stored(tensors, base, quantization):
    tensor = tensors.get(base + ".nbits_weight")
    if tensor is None:
        return None
    attributes = getattr(tensor, "attributes", {})
    return "bf16", int(attributes.get("N", 0)) * int(attributes.get("K", 0))


# Each scheme's recogniser, (tensors, base, quantization) -> (rate, elements) or None from headers alone. Checked in
# order; the first whose tensors are present names the matrix. A new scheme is one entry here (and its decoder in
# crit-eval's schemes).
RECOGNISERS = {
    "nvfp4": _nvfp4_stored,
    "fp8": _fp8_stored,
    "int": _ct_int_stored,
    "awq": _awq_stored,
    "gptq": _gptq_stored,
    "mlx": _mlx_stored,
    "bnb4": _bnb4_stored,
    "bnb8": _bnb8_stored,
    "nbits": _nbits_stored,
    "plain": _plain_stored,
}
# The tensor that names each scheme's matrix, for the ceiling's walk over a file's tensors.
PRIMARY = (".weight_packed", ".qweight", ".nbits_weight", ".weight")


def stored(tensors, base, quantization=None):
    """(scheme, rate, elements) of matrix `base` from the tensors' headers ({name: TensorInfo}), or None."""
    for name, recognise in RECOGNISERS.items():
        found = recognise(tensors, base, quantization or {})
        if found:
            return (name,) + tuple(found)
    return None



LAYER_ANCHOR = "layers.0.input_layernorm.weight"
# A decoder layer's tensor: the prefix before `layers.<index>.` at a name boundary.
LAYER_TENSOR = re.compile(r"(?:^|(?<=\.))layers\.(\d+)\.")


def layer_prefix(names):
    """The decoder-layer prefix (for example `model.` before `layers.`) the reference also uses: the one of its layer
    0's input norm, else, for layers without one (such as hyper-connection layers, which norm their streams), of the
    prefixes under which a layer 0 is stored (draft and vision layers have their own), the one holding the most layer
    tensors."""
    names = list(names)
    for name in names:
        if name.endswith(LAYER_ANCHOR):
            return name[: -len(LAYER_ANCHOR)]
    counts, zero = {}, set()
    for name in names:
        found = LAYER_TENSOR.search(name)
        if found:
            prefix = name[: found.start()]
            counts[prefix] = counts.get(prefix, 0) + 1
            if found.group(1) == "0":
                zero.add(prefix)
    if not zero:
        raise ValueError(f"cannot find the decoder layers: no tensor ends with {LAYER_ANCHOR} or holds layers.0.")
    return max(sorted(zero), key=lambda prefix: counts[prefix])


@dataclass(frozen=True)
class StoredMatrix:
    """The stored tensors of one matrix an engine runs without decoding (matrix_parts).

    `shape` is the logical [..., out, in] (leading dimensions for stacked experts); `codes` the tensor of the values
    (nvfp4: U8 E2M1 pairs, low nibble first, [..., out, in / 2]; fp8: F8_E4M3 [..., out, in]; plain: the weight);
    `scales` the tensor of its scales: nvfp4 E4M3 per 16 inputs [..., out, in / 16]; fp8 `scale_granularity`
    "tensor", "row" (one per output row) or "block" (`block` rows x columns), always multipliers. `global_scale`
    (nvfp4) is the FP32 factor, a divisor when `global_divides` (compressed-tensors weight_global_scale) and a
    multiplier otherwise (ModelOpt weight_scale_2): weight = code * scale / global, or code * scale * global.
    `input_scale` names the static activation scale the checkpoint declares for its own inference, with
    `input_divides` alike (compressed-tensors input_global_scale divides, ModelOpt and FP8 input_scale multiply);
    None for dynamic or unquantized activations. `layout` is "compressed-tensors", "modelopt" or None."""
    base: str
    scheme: str
    layout: str | None
    shape: tuple
    codes: str
    scales: str | None = None
    scale_granularity: str | None = None
    block: tuple | None = None
    global_scale: str | None = None
    global_divides: bool = False
    input_scale: str | None = None
    input_divides: bool = False


# Schemes matrix_parts describes; the others are decoded (crit-eval) or converted by the engine instead.
PART_SCHEMES = ("nvfp4", "fp8", "plain")


def matrix_parts(tensors, base, quantization=None):
    """The StoredMatrix of matrix `base` ({name: TensorInfo} headers, the model's quantization config), None when no
    matrix is stored there. SchemeError for a matrix in a scheme outside PART_SCHEMES (stored() names it) or whose
    tensors disagree with its scheme."""
    found = stored(tensors, base, quantization)
    if found is None:
        return None
    name = found[0]
    if name not in PART_SCHEMES:
        raise SchemeError(f"{base}: stored as {name}, which crit describes for decoding, not for running as stored "
                          f"({', '.join(PART_SCHEMES)})")
    quantization = quantization or {}
    compressed = quant_method(quantization) == "compressed-tensors"
    has = lambda suffix: base + suffix in tensors
    if name == "plain":
        return StoredMatrix(base, name, None, _shape(tensors, base + ".weight"), base + ".weight")
    if name == "nvfp4":
        if has(".weight_packed"):
            packed, layout, global_scale, divides = ".weight_packed", "compressed-tensors", ".weight_global_scale", True
            input_scale, input_divides = ".input_global_scale", True
        else:
            packed, layout, global_scale, divides = ".weight", "modelopt", ".weight_scale_2", False
            input_scale, input_divides = ".input_scale", False
        if not has(global_scale):
            raise SchemeError(f"{base}: NVFP4 {layout} weights without {base}{global_scale}")
        if _elements(tensors, base + global_scale) != 1:
            raise SchemeError(f"{base}{global_scale}: expected one global scale, found shape "
                              f"{list(_shape(tensors, base + global_scale))}")
        shape = _shape(tensors, base + packed)
        return StoredMatrix(base, name, layout, shape[:-1] + (shape[-1] * 2,), base + packed, base + ".weight_scale",
                            "group16", None, base + global_scale, divides,
                            base + input_scale if has(input_scale) else None, input_divides)
    # fp8: per tensor, per row or per block, a multiplier whatever its name (weight_scale_inv included)
    shape = _shape(tensors, base + ".weight")
    scale = ".weight_scale_inv" if has(".weight_scale_inv") else ".weight_scale"
    if not has(scale):
        raise SchemeError(f"{base}: FP8 weights without {base}.weight_scale or {base}.weight_scale_inv")
    scale_shape, block = _shape(tensors, base + scale), None
    if math.prod(scale_shape) == 1:
        granularity = "tensor"
    elif scale_shape in (shape[:-1] + (1,), shape[:-1]):
        granularity = "row"
    else:
        granularity = "block"
        block = tuple(int(n) for n in (quantization.get("weight_block_size")
                                       or ct_weight_args(quantization, base).get("block_structure") or (128, 128)))
        if len(shape) < 2 or len(block) != 2 or scale_shape[-2:] != (-(-shape[-2] // block[0]),
                                                                      -(-shape[-1] // block[1])):
            raise SchemeError(f"{base}{scale}: shape {list(scale_shape)} is neither one scale, one per row nor one "
                              f"per {list(block)} block of {list(shape)}")
    layout = "compressed-tensors" if compressed else ("modelopt" if quant_method(quantization) == "modelopt" else None)
    return StoredMatrix(base, name, layout, shape, base + ".weight", base + scale, granularity, block,
                        input_scale=base + ".input_scale" if has(".input_scale") else None)


def _elements(tensors, name):
    return math.prod(_shape(tensors, name))
