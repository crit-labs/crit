"""A stored matrix's exact values and its re-encoding to NVFP4 or FP8 per row, for matrices only a drafter reads.

A draft layer's matrices may be re-encoded at load (the stored target verifies every draft): a projections plugin
re-encodes a draft matrix it loads (load(..., encode=)), a model plugin a draft layer's routed experts stored in
another format than the target's (an NVFP4 checkpoint whose MTP experts are FP8 in 128 x 128 blocks).

values(tensors, format, device, first, last): rows first..last of a stored matrix as FP64, exactly (an E4M3 code times
an FP32 scale fits FP64). Formats: bf16 (or any 16/32-bit dtype), fp8:channel (scale [rows]), fp8:tensor (scale []),
fp8:block (scale [rows / b0][columns / b1], block (b0, b1)).

encode(tensors, format, encode, device): (tensors, format) of the matrix re-encoded.

    nvfp4: global scale g = FP32(6 * 448 / amax) over the matrix (1 for an all-zero one); per 16 values the E4M3
    scale e4m3(amax(block) * g / 6) and E2M1 codes e2m1(v * g / scale), round to nearest even, saturating at 448
    and 6, zero for a zero scale (the NVFP4 row quantizer's rule at g, code for code). Returned as
    weight [rows][columns / 2] uint8, scale [rows][columns / 16] uint8, global_scale [] (a divisor: W = e2m1 *
    scale / global_scale), with no input scale (W4A16).
    fp8: per row the FP32 scale amax / 448 (1 for an all-zero row) and E4M3 codes e4m3(v / scale), the same
    rounding (fp8:channel).

The encoders run on `device` (the weight's own unless given) a slice of rows at a time (ENCODE_BYTES of FP64)."""

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)   # magnitudes of E2M1 codes 0..7; bit 3 is the sign
E2M1_MAX, E4M3_MAX = 6.0, 448.0
ENCODE_BYTES = 256 << 20
SOURCES = ("bf16", "fp8:channel", "fp8:tensor", "fp8:block")
_TABLES = {}


def tables(device):
    """E4M3's 127 finite nonnegative values and E2M1's 8 magnitudes in FP64, ascending, with the midpoints between
    neighbours (exact in FP64), on `device`."""
    import torch
    key = str(device)
    if key not in _TABLES:
        e4m3 = torch.arange(0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float64).to(device)
        e2m1 = torch.tensor(E2M1, dtype=torch.float64, device=device)
        _TABLES[key] = {"e4m3": e4m3, "e4m3_mids": (e4m3[1:] + e4m3[:-1]) / 2,
                        "e2m1": e2m1, "e2m1_mids": (e2m1[1:] + e2m1[:-1]) / 2}
        # The process keeps them for every engine on the device: named on its memory, so a closed engine's close
        # is measured without them (crit_plugins.lifetime).
        from .memory import of
        _TABLES[key] = {name: of(device).process("encodings", name, table) for name, table in _TABLES[key].items()}
    return _TABLES[key]


def nearest(x, table, mids):
    """For x >= 0 (FP64): the index of the nearest entry of `table` (ascending), ties to the even index, values past
    the last entry saturating to it. The two neighbours of x are at most a factor of 2 apart, so their distances to x
    are exact in FP64 and a tie is exactly x == midpoint."""
    import torch
    index = torch.bucketize(x, mids, right=False)        # midpoints strictly below x
    at = mids[index.clamp(max=mids.numel() - 1)]
    return index + ((index < mids.numel()) & (at == x) & (index % 2 == 1)).to(index.dtype)


def values(tensors, format, device, first, last):
    """Rows first..last of a stored matrix as FP64 values, exactly."""
    import torch
    weight = tensors["weight"][first:last]
    if format == "bf16":
        return weight.to(device=device, dtype=torch.float64)
    if format not in SOURCES:
        raise ValueError(f"a stored matrix is {', '.join(SOURCES)} to be read exactly, not {format!r}")
    codes = weight.to(device).view(torch.float8_e4m3fn).to(torch.float64)
    scale = tensors["scale"].to(device=device, dtype=torch.float32)
    if format == "fp8:channel":
        return codes * scale.reshape(-1)[first:last].to(torch.float64)[:, None]
    if format == "fp8:tensor":
        return codes * scale.reshape(-1)[0].to(torch.float64)
    b0, b1 = tensors["block"]
    rows = torch.arange(first, last, device=device) // b0
    factors = scale[rows].to(torch.float64).repeat_interleave(b1, dim=1)[:, :codes.shape[1]]
    return codes * factors


def encode(tensors, format, encode, device=None):
    """The matrix's tensors re-encoded to `encode` (nvfp4 or fp8): (tensors, format)."""
    import torch
    if encode not in ("nvfp4", "fp8"):
        raise ValueError(f"a matrix re-encodes to nvfp4 or fp8, not {encode!r}")
    weight = tensors["weight"]
    device = torch.device(device) if device is not None else weight.device
    rows, columns = weight.shape
    if encode == "nvfp4" and columns % 16:
        raise ValueError(f"an nvfp4 matrix has a multiple of 16 columns, not {columns}")
    step = max(1, ENCODE_BYTES // (columns * 8))
    spans = [(r, min(rows, r + step)) for r in range(0, rows, step)]
    found = tables(device)
    e4m3, e4m3_mids, e2m1_mids = found["e4m3"], found["e4m3_mids"], found["e2m1_mids"]
    if encode == "fp8":
        codes = torch.empty(rows, columns, dtype=torch.uint8, device=device)
        scale = torch.empty(rows, dtype=torch.float32, device=device)
        for first, last in spans:
            v = values(tensors, format, device, first, last)
            if not bool(torch.isfinite(v).all()):
                raise ValueError("a matrix with non-finite values does not re-encode")
            amax = v.abs().amax(dim=-1)
            s = torch.where(amax > 0, amax / E4M3_MAX, torch.ones_like(amax)).to(torch.float32)
            q = v / s.to(torch.float64)[:, None]
            code = nearest(q.abs().clamp(max=E4M3_MAX), e4m3, e4m3_mids)
            code = code | (((q < 0) & (code > 0)).to(code.dtype) << 7)
            codes[first:last], scale[first:last] = code.to(torch.uint8), s
        return {"weight": codes.view(torch.float8_e4m3fn), "scale": scale}, "fp8:channel"
    amax = 0.0
    for first, last in spans:
        v = values(tensors, format, device, first, last)
        if not bool(torch.isfinite(v).all()):
            raise ValueError("a matrix with non-finite values does not re-encode")
        amax = max(amax, float(v.abs().max()))
    g32 = torch.tensor(E2M1_MAX * E4M3_MAX / amax if amax > 0 else 1.0, dtype=torch.float32)
    g = float(g32)
    codes = torch.empty(rows, columns // 2, dtype=torch.uint8, device=device)
    scales = torch.empty(rows, columns // 16, dtype=torch.uint8, device=device)
    for first, last in spans:
        blocks = values(tensors, format, device, first, last).view(last - first, -1, 16)
        s = nearest((blocks.abs().amax(dim=-1) * g / E2M1_MAX).clamp(max=E4M3_MAX), e4m3, e4m3_mids)
        decoded = e4m3[s][..., None]
        q = torch.where(decoded > 0, blocks * g / decoded.clamp(min=e4m3[1]), torch.zeros_like(blocks))
        code = nearest(q.abs().clamp(max=E2M1_MAX), found["e2m1"], e2m1_mids)
        code = (code | (((q < 0) & (code > 0)).to(code.dtype) << 3)).view(last - first, columns)
        codes[first:last] = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
        scales[first:last] = s.to(torch.uint8)
    return {"weight": codes, "scale": scales, "global_scale": g32.to(device)}, "nvfp4"
