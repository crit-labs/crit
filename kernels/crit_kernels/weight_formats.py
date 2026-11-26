"""The weight formats the projection benches key their shapes by, and their INT4 helpers, in one place.

WEIGHTS: the format codes of the GEMV and row kernels' run option (decode-gemv's gemv-cuda key, which mma-rows takes
for the formats it multiplies). INT4 with a zero point a group (int4_zp) is how AWQ and GPTQ checkpoints store their
matrices once repack_awq / repack_gptq put them in crit's rows: codes [N][K / 2] unsigned (element 2i in the low nibble
of byte i), FP16 scales and uint8 zero points [N][K / group], a value (code - zero) * scale."""
import torch

WEIGHTS = {"nvfp4": 0, "fp8": 1, "mxfp8": 2, "bf16": 3, "int4_zp": 4}
INT4_GROUP = 128   # a shape's group where it names none (AWQ's and GPTQ's usual)


def int4_shapes(model):
    """The model's AWQ or GPTQ matrices (qweight int32: AWQ [K][N / 8], GPTQ [K / 8][N]) as int4_zp keys."""
    quant = model.quantization or {}
    method = str(quant.get("quant_method", "")).lower()
    if method not in ("awq", "gptq") or int(quant.get("bits", 4)) != 4:
        return []
    group = int(quant.get("group_size") or INT4_GROUP)
    found = []
    for name, (dtype, shape) in model.tensors.items():
        if name.endswith(".qweight") and dtype == "I32" and len(shape) == 2:
            n, k = (shape[1] * 8, shape[0]) if method == "awq" else (shape[1], shape[0] * 8)
            if group > 0 and group % 32 == 0 and k % group == 0:
                found.append({"name": name[:-8], "weights": WEIGHTS["int4_zp"], "N": n, "K": k, "group": group})
    return found


def int4_matrix(n, k, group, generator):
    """Random INT4 rows in crit's layout from the generator (on the matrix's device): codes [N][K / 2], FP16 scales and
    zero points (0..16) [N][K / group]."""
    device = generator.device
    codes = torch.randint(0, 256, (n, k // 2), generator=generator, dtype=torch.uint8, device=device)
    scales = ((torch.rand(n, k // group, generator=generator, device=device) + 0.5) / 64).to(torch.float16)
    zeros = torch.randint(0, 17, (n, k // group), generator=generator, dtype=torch.uint8, device=device)
    return codes, scales, zeros


def int4_values(codes, scales, zeros, group, lo=0, hi=None):
    """Rows lo..hi of an INT4 matrix in crit's rows as FP64 values (code - zero) * scale."""
    c = codes[lo:hi].long()
    q = torch.stack((c & 15, c >> 4), dim=-1).reshape(c.shape[0], -1).double()
    z = zeros[lo:hi].double().repeat_interleave(group, 1)
    s = scales[lo:hi].double().repeat_interleave(group, 1)
    return (q - z) * s


def int4_yardstick(cache, x, codes, scales, zeros, group):
    """torch's W4A16 INT4 group GEMM (torch._weight_int4pack_mm, tinygemm) on the same codes, repacked once into the
    cache: a BF16 scale and offset a group ((code - 8) * scale + offset, offset (8 - zero) * scale). None where torch
    lacks it or refuses the shape."""
    if not hasattr(torch, "_weight_int4pack_mm") or not hasattr(torch, "_convert_weight_to_int4pack"):
        return None
    n, k = codes.shape[0], codes.shape[1] * 2
    if ("int4pack", n, k, group) not in cache:
        try:
            packed = torch._convert_weight_to_int4pack(codes.contiguous(), 8)
            sz = torch.stack((scales.float(), (8 - zeros.float()) * scales.float()), -1)   # [N][K / group][2]
            sz = sz.transpose(0, 1).contiguous().to(torch.bfloat16)                          # [K / group][N][2]
            torch._weight_int4pack_mm(x, packed, group, sz)
        except (RuntimeError, TypeError):
            return None
        cache[("int4pack", n, k, group)] = (packed, sz)
    packed, sz = cache[("int4pack", n, k, group)]
    return lambda: torch._weight_int4pack_mm(x, packed, group, sz)
