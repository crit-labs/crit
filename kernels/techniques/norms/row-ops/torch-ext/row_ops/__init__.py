"""Row operations around a model's blocks, one Triton launch a call (two where a reduction spans programs): what a
mixer or a residual computes on each row between its matmuls, where torch would take a chain of elementwise kernels.

    head_clip(x, threshold, scale, tau, round_bf16, out_dtype)   each head vector (times a learned scale) shortened to
                                                                  an L2 length, optionally times a per-token factor
    small_product(r, proj)                  each row's few values times a small matrix [R][E], BF16 rounding in and out
    sink_gate(a, lse, sink, gate)           a * sigmoid(lse - sink) * sigmoid(gate): a sink logit with no value and
                                            a sigmoid output gate on an attention's rows
    scaled_add(y, x, scale, out_dtype)      (y + x) * scale
    summary_mix(blocks, h, query)           softmax attention of learned queries over a token's normed summary rows
                                            and its own row, without a weight (residuals that attend over depth)
    pair_norm(x, message)                   [rms(x), rms(message)] in BF16, each without a weight
    gated_join(x, message, f, a)            (1 - tanh(f)) * x + tanh(a) * message
    low_rank_gates(x, message, down, up)    several gated joins at once with their low-rank gate products inside
    span_attention(q, k, v, first, last)    queries over a span of each row's cached positions, no causal mask

Inputs BF16, FP16 or FP32 on a CUDA device (the BF16 inputs a function names); FP32 inside. Each row's bits are
independent of the rows beside it and of the row count: a launch never specializes on the row count, and a row's
reduction order follows from its width and the structure's order knobs (looked up by the width alone). Nothing
synchronizes with the host and every allocation comes from torch's caching allocator, so CUDA graphs capture every
call once its shape has been compiled.

Knobs: each kernel's launch constants (blocks, chunks, warps) come from the build's tuning (_crit_tuning, written by
crit_kernels.build from a sweep's table or kernel.json's defaults). The structures whose programs reduce (head-clip,
summary-mix, pair-norm, low-rank-gates, span-attention) look their knobs up by the shape alone, so a row's sums keep
one order at every row count; the elementwise ones (small-product, sink-gate, scaled-add, gated-join) by the row count
too, their knobs changing speed only. The FP64 references are the operations'
(operations.json of the model techniques that compute them, crit_kernels.contracts).
"""
import torch

from . import _crit_tuning, _triton

__all__ = ["gated_join", "head_clip", "low_rank_gates", "pair_norm", "scaled_add", "sink_gate", "small_product",
           "span_attention", "summary_mix"]

EPS = 1e-6
_KNOBS = {}


def _knobs(structure, rows, **key):
    """A structure's knobs for a shape key and row count: the build's tuning, else kernel.json's defaults."""
    if getattr(_crit_tuning, "SELECTED", None):
        return _crit_tuning.knobs(structure, key, rows)
    at = (structure, rows, *key.values())
    found = _KNOBS.get(at)
    if found is None:
        if len(_KNOBS) > 4096:
            _KNOBS.clear()
        found = _KNOBS[at] = _crit_tuning.knobs(structure, key, rows)
    return found


def _launch(kernel, structure, rows, call, **key):
    """Launch a kernel on the knobs for its shape key and row count: call(knobs) gives (grid, arguments, options)."""
    return _crit_tuning.launch(kernel, structure, key, _knobs(structure, rows, **key), call)


def _pow2(n):
    return 1 << max(0, int(n) - 1).bit_length()


def _cdiv(a, b):
    return -(-a // b)


def _cuda(x, what):
    if not x.is_cuda:
        raise ValueError(f"{what} is a CUDA tensor, not one on {x.device}")


def head_clip(x, threshold, scale=None, tau=None, round_bf16=False, out_dtype=None):
    """x [t][heads][d] (a token's heads contiguous; tokens at any stride) out in out_dtype (FP32 by default): each
    head vector (times scale [heads][d] FP32) shortened to threshold where its L2 norm (+ 1e-6) is longer; with tau
    [t] FP32, times its token's tau (rounded to BF16 first with round_bf16)."""
    _cuda(x, "x")
    t, heads, d = x.shape
    if x.stride(2) != 1 or x.stride(1) != d:
        x = x.contiguous()
    out = torch.empty(t, heads, d, dtype=out_dtype or torch.float32, device=x.device)
    if t:
        def call(k):
            return ((t * heads,), (x, scale if scale is not None else x, tau if tau is not None else x, out,
                                   x.stride(0), heads, d, float(threshold)),
                    dict(SCALE=scale is not None, TAU=tau is not None, ROUND=bool(round_bf16), BLOCK=_pow2(d),
                         num_warps=k["WARPS"]))
        _launch(_triton.head_clip_kernel, "head-clip", 1, call, D=d)   # order knobs: by the width
    return out


def small_product(r, proj):
    """[t][heads][E] FP32 (values of BF16): r [t][heads][R] FP32 rounded to BF16, times proj [R][E] FP32 (sums in
    index order), rounded to BF16."""
    _cuda(r, "r")
    t, heads, R = r.shape
    E = proj.shape[1]
    out = torch.empty(t, heads, E, dtype=torch.float32, device=r.device)
    if t:
        r = r.contiguous()

        def call(k):
            return ((t * heads, _cdiv(E, k["BLOCK"])), (r, proj, out, E),
                    dict(R=R, BLOCK=k["BLOCK"], num_warps=k["WARPS"]))
        _launch(_triton.small_product_kernel, "small-product", t * heads, call, R=R, E=E)
    return out


def sink_gate(a, lse, sink, gate):
    """[t][heads * d] BF16: a [t][heads][d] times sigmoid(lse - sink) ([t][heads]; a sink logit with no value) times
    sigmoid(gate) [t][heads * d] (rows at any stride)."""
    _cuda(a, "a")
    t, heads, d = a.shape
    out = torch.empty(t, heads * d, dtype=torch.bfloat16, device=a.device)
    if t:
        a, lse = a.contiguous(), lse.contiguous()

        def call(k):
            return ((t * heads,), (a, lse, sink, sink.stride(0), gate, gate.stride(0), out, heads, d),
                    dict(BLOCK=_pow2(d), num_warps=k["WARPS"]))
        _launch(_triton.sink_gate_kernel, "sink-gate", t, call, D=d)
    return out


def scaled_add(y, x, scale=1.0, out_dtype=None):
    """(y + x) * scale in out_dtype (x's by default): y and x any float dtype, of one shape."""
    _cuda(x, "x")
    y, x = y.contiguous(), x.contiguous()
    out = torch.empty(x.shape, dtype=out_dtype or x.dtype, device=x.device)
    n = x.numel()
    if n:
        def call(k):
            return ((_cdiv(n, k["BLOCK"]),), (y, x, out, n, float(scale)),
                    dict(BLOCK=k["BLOCK"], num_warps=k["WARPS"]))
        _launch(_triton.scaled_add_kernel, "scaled-add", x.shape[0] if x.dim() else 1, call)
    return out


def summary_mix(blocks, h, query):
    """[t][Q][H] FP32 of summaries blocks [t][N][H] (any float dtype, N may be 0), rows h [t][H] and queries query
    [Q][H] FP32: softmax over the N + 1 of query . rms(v) / sqrt(H), the weighted sum of the rms(v). Two launches
    over (token, chunk of the hidden values): the chunks' partial sums, then each chunk's output from their sum in
    chunk order. The result is a view whose query rows [:, q] are contiguous."""
    _cuda(h, "h")
    t, n, hidden = blocks.shape
    q = query.contiguous()
    Q = q.shape[0]
    QP = _pow2(Q)
    if QP != Q:
        q = torch.cat([q, q.new_zeros(QP - Q, hidden)])
    blocks = blocks if blocks.stride(-1) == 1 and (n == 0 or blocks.stride(1) == hidden) else blocks.contiguous()
    h = h if h.stride(-1) == 1 else h.contiguous()
    out = torch.empty(QP, t, hidden, dtype=torch.float32, device=h.device)
    if t:
        NP = _pow2(n + 1)
        chunk = _knobs("summary-mix", 1, H=hidden)["CHUNK"]   # the partials' layout: by the width alone
        chunks = _cdiv(hidden, chunk)
        stride_bt = blocks.stride(0) if n else 0
        parts = torch.empty(t, chunks, NP + QP * NP, dtype=torch.float32, device=h.device)

        def first(k):
            return ((t, chunks), (blocks, h, q, parts, stride_bt, hidden, h.stride(0), n, hidden, chunks),
                    dict(Q=QP, NP=NP, CHUNK=chunk, num_warps=k["WARPS"]))

        def second(k):
            return ((t, chunks), (blocks, h, parts, out, stride_bt, hidden, h.stride(0), t * hidden, n, hidden, chunks,
                                  hidden ** -0.5, EPS), dict(Q=QP, NP=NP, CHUNK=chunk, num_warps=k["WARPS"]))
        _launch(_triton.mix_parts_kernel, "summary-mix", 1, first, H=hidden)
        _launch(_triton.mix_kernel, "summary-mix", 1, second, H=hidden)
    return out[:Q].transpose(0, 1)


def pair_norm(x, message):
    """[t][2H] BF16: [rms(x), rms(message)], each normed without a weight (eps 1e-6); x and message [t][H], rows at
    any stride."""
    _cuda(x, "x")
    t, hidden = x.shape
    out = torch.empty(t, 2 * hidden, dtype=torch.bfloat16, device=x.device)
    if t:
        def call(k):
            return ((t,), (x, message, out, x.stride(0), message.stride(0), hidden, EPS),
                    dict(BLOCK=k["BLOCK"], num_warps=k["WARPS"]))
        _launch(_triton.pair_norm_kernel, "pair-norm", 1, call, H=hidden)
    return out


def gated_join(x, message, f, a):
    """[t][H] FP32: (1 - tanh(f)) * x + tanh(a) * message (f, a FP32 [t][H], rows of one stride: two halves of one
    product's rows, or two tensors)."""
    _cuda(x, "x")
    t, hidden = x.shape
    if f.stride(-1) != 1 or a.stride(-1) != 1 or f.stride(0) != a.stride(0):
        f, a = f.contiguous(), a.contiguous()
    out = torch.empty(t, hidden, dtype=torch.float32, device=x.device)
    if t:
        def call(k):
            return ((t, _cdiv(hidden, k["BLOCK"])), (x, message, f, a, out, x.stride(0), message.stride(0),
                                                     f.stride(0), hidden),
                    dict(BLOCK=k["BLOCK"], num_warps=k["WARPS"]))
        _launch(_triton.gated_join_kernel, "gated-join", t, call, H=hidden)
    return out


def low_rank_gates(x, message, down, up):
    """[t][G][hidden] FP32: G gated joins at once, gate i joining x[:, i] to message[:, i] (each [t][G][hidden], any
    float dtype, rows and gates by stride: an expanded x is one input for every gate):
        g = down_i . [rms(x), rms(m)] (BF16 inputs and g),  f, a = up_i . g (up_i: forget rows, then gain rows),
        (1 - tanh(f)) * x + tanh(a) * m
    down [G][rank][2 hidden], up [G][2 hidden][rank], BF16. Two launches for all G gates, each row's sums its own."""
    _cuda(x, "x")
    t, g, hidden = x.shape
    rank = down.shape[1]
    if x.stride(-1) != 1 or message.stride(-1) != 1:
        x, message = x.contiguous(), message.contiguous()
    low = torch.empty(g, t, rank, dtype=torch.float32, device=x.device)
    out = torch.empty(t, g, hidden, dtype=torch.float32, device=x.device)
    if t:

        def first(k):
            return ((g, _cdiv(rank, k["BR"]), t), (x, message, down, low, t, hidden, rank, x.stride(0), x.stride(1),
                                                   message.stride(0), message.stride(1), EPS),
                    dict(BR=k["BR"], BK=k["BK"], num_warps=k["WARPS"]))

        def second(k):
            return ((g, _cdiv(hidden, k["BH"]), t), (low, up, x, message, out, t, hidden, rank, x.stride(0),
                                                     x.stride(1), message.stride(0), message.stride(1)),
                    dict(G=g, BH=k["BH"], RP=_pow2(rank), num_warps=k["WARPS"]))
        _launch(_triton.gate_down_kernel, "low-rank-gates", 1, first, H=hidden, R=rank)
        _launch(_triton.gate_up_kernel, "low-rank-gates", 1, second, H=hidden, R=rank)
    return out


def span_attention(q, k, v, first, last):
    """[rows][kv heads][G][D] BF16: each row's queries q (G a kv head's, a power of two of at least 16) over its keys
    and values k, v [rows][kv heads][T][D] (BF16, contiguous) at positions first[r] .. last[r] (int tensors), every
    query seeing all of them (no causal mask); programs past a row's last position return at once."""
    _cuda(q, "q")
    rows, kvh, g, d = q.shape
    t = k.shape[2]
    knobs = _knobs("span-attention", 1, G=g, D=d)      # order knobs: by the shape alone
    chunk, step = knobs["CHUNK"], knobs["BN"]
    splits = _cdiv(t, chunk)
    acc = torch.empty(rows * kvh, splits, g, d, dtype=torch.float32, device=q.device)
    ml = torch.empty(rows * kvh, splits, g, 2, dtype=torch.float32, device=q.device)
    out = torch.empty(rows, kvh, g, d, dtype=torch.bfloat16, device=q.device)
    q = q.contiguous()

    def part(kn):
        return ((rows * kvh, splits), (q, k, v, first, last, acc, ml, t, kvh, splits, d ** -0.5),
                dict(G=g, D=d, CHUNK=chunk, BN=step, num_warps=kn["WARPS"]))

    def merge(kn):
        return ((rows * kvh,), (acc, ml, out, splits), dict(G=g, D=d, num_warps=kn["WARPS"]))
    _launch(_triton.span_part_kernel, "span-attention", 1, part, G=g, D=d)
    _launch(_triton.span_merge_kernel, "span-attention", 1, merge, G=g, D=d)
    return out
