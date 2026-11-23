"""The CPU reference a grouped expert projection is checked against before an engine trusts it (FP64, standard library).

A routed SwiGLU block over a prompt chunk: each token's routes in order, each expert's gate/up and down matrices
decoded from their stored format, SwiGLU between them, the outputs summed with the routing weights. Any kernel
(a provider's grouped GEMM, one of crit's kernels, the engine's own) fed the same routes, weights and activations must land
within its arithmetic's tolerance of moe_block, and route_order states the order a stable route sort must produce.

FP64 in pure Python is slow: check a few tokens of a real chunk (their rows only), not a whole prompt.
"""
import math

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FORMATS = ("bf16", "fp8_e4m3_row", "nvfp4", "mxfp4")


def e2m1(code):
    return -E2M1[code & 7] if code & 8 else E2M1[code & 7]


def e4m3(byte):
    e, m = (byte >> 3) & 15, byte & 7
    if e == 15 and m == 7:
        return math.nan
    value = math.ldexp(m, -9) if e == 0 else math.ldexp(8 + m, e - 10)
    return -value if byte & 0x80 else value


def bf16(word):
    import struct
    return struct.unpack("<f", struct.pack("<I", word << 16))[0]


def decode_matrix(fmt, codes, scales, rows, cols, post=1.0):
    """[rows][cols] values of one stored matrix. bf16: 2 little-endian bytes an element; fp8_e4m3_row: one byte an
    element and one little-endian FP32 scale per row; nvfp4: E2M1 codes two a byte (low nibble the even element), an
    E4M3 scale per 16; mxfp4: the same codes, an E8M0 scale per 32. post multiplies every value (nvfp4's 1 / global
    scale)."""
    import struct
    if fmt not in FORMATS:
        raise ValueError("unknown format")
    out = []
    for n in range(rows):
        if fmt == "bf16":
            row = [bf16(codes[2 * (n * cols + k)] | codes[2 * (n * cols + k) + 1] << 8) for k in range(cols)]
        elif fmt == "fp8_e4m3_row":
            scale = struct.unpack_from("<f", scales, 4 * n)[0]
            row = [e4m3(codes[n * cols + k]) * scale for k in range(cols)]
        else:
            block = 16 if fmt == "nvfp4" else 32
            row = []
            for k in range(cols):
                byte = codes[(n * cols + k) // 2]
                code = byte & 15 if k % 2 == 0 else byte >> 4
                s = scales[(n * cols + k) // block]
                row.append(e2m1(code) * (e4m3(s) if fmt == "nvfp4" else math.ldexp(1.0, s - 127)))
        out.append([v * post for v in row])
    return out


def route_order(ids, experts):
    """The routes (token * top_k + j) sorted by expert, routes in increasing order within an expert; ids outside
    [0, experts) are skipped. ids: [tokens][top_k]."""
    top_k = len(ids[0]) if ids else 0
    flat = [(e, t * top_k + j) for t, row in enumerate(ids) for j, e in enumerate(row) if 0 <= e < experts]
    return [route for _, route in sorted(flat)]


def silu(v):
    return v / (1.0 + math.exp(-v))


# a routed block's gate: silu(gate) * up (SwiGLU), GELU's tanh form, or SwiGLU whose linear gate factor and up are
# capped by a tanh each (softcapped_silu: caps (gate, up), cap * tanh(v / cap); the sigmoid sees the raw gate)
ACTIVATIONS = ("silu", "gelu_tanh", "softcapped_silu")


def gelu_tanh(v):
    return 0.5 * v * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (v + 0.044715 * v * v * v)))


def _capped(v, cap):
    return cap * math.tanh(v / cap)


def moe_block(x, ids, weights, gate_up, down, inter, tokens=None, activation="silu", caps=None):
    """[token][hidden] outputs of a routed gated block (SwiGLU by default). x [tokens][hidden]; ids, weights [tokens]
    [top_k]; gate_up(e) and down(e) return expert e's decoded [2 * inter][hidden] and [hidden][inter] matrices
    (decode_matrix), called once per expert used. tokens: the token rows to compute (all by default). activation: the
    gate of the block's product, act(gate) * up (ACTIVATIONS); caps: softcapped_silu's (gate cap, up cap)."""
    if activation not in ACTIVATIONS:
        raise ValueError(f"a routed block's activation is {' or '.join(ACTIVATIONS)}, not {activation!r}")
    if activation == "softcapped_silu":
        if not caps or len(caps) != 2 or min(caps) <= 0:
            raise ValueError("softcapped_silu takes two positive caps: (gate, up)")
        gate_cap, up_cap = caps

        def silu(v):   # the linear gate factor capped, the sigmoid of the raw gate
            return _capped(v, gate_cap) / (1.0 + math.exp(-v))

        def up(v):
            return _capped(v, up_cap)
    else:
        silu = globals()["silu"] if activation == "silu" else gelu_tanh

        def up(v):
            return v
    cache_gu, cache_dn = {}, {}
    rows = range(len(x)) if tokens is None else tokens
    out = {}
    for t in rows:
        acc = [0.0] * len(x[t])
        for e, w in zip(ids[t], weights[t]):
            if e < 0:
                continue
            gu = cache_gu.get(e) or cache_gu.setdefault(e, gate_up(e))
            dn = cache_dn.get(e) or cache_dn.setdefault(e, down(e))
            act = [silu(math.fsum(a * b for a, b in zip(gu[i], x[t])))
                   * up(math.fsum(a * b for a, b in zip(gu[inter + i], x[t]))) for i in range(inter)]
            for h in range(len(acc)):
                acc[h] += w * math.fsum(a * b for a, b in zip(dn[h], act))
        out[t] = acc
    return [out[t] for t in rows]
