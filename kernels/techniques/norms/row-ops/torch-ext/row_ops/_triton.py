"""The Triton kernels behind row_ops' functions. Every kernel computes in FP32 and rounds where its function says.

Specialization: the arguments that follow the request (an element count, the token count of the gates' guard, a
stride of a per-call buffer) are each kernel's do_not_specialize; every other runtime argument the model fixes (a
width, a weight's or a product's stride, a layer's summary count).

Row invariance: a program covers one row (a token, a token head, a row's KV head) or a block of one row's values, and
each row reduces in the order its width and the structure's order knobs give it, never the row count: the row count
enters only the grid and the masks. Block sizes and warp counts are tl.constexpr / launch parameters the package reads
from the build's tuning (kernel.json's knobs).
"""
import triton
import triton.language as tl


# -- head-clip ----------------------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=[])
def head_clip_kernel(x_ptr, scale_ptr, tau_ptr, out_ptr, stride_xt, heads, D, threshold, SCALE: tl.constexpr,
                     TAU: tl.constexpr, ROUND: tl.constexpr, BLOCK: tl.constexpr):
    # One token head: x (times its scale) shortened to threshold where its L2 norm is longer (norm + 1e-6); with TAU,
    # rounded to BF16 first (ROUND) and multiplied by its token's tau; stored in the output's dtype.
    r = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    dm = d < D
    x = tl.load(x_ptr + (r // heads) * stride_xt + (r % heads) * D + d, mask=dm, other=0.0).to(tl.float32)
    if SCALE:
        x = x * tl.load(scale_ptr + (r % heads) * D + d, mask=dm, other=0.0)
    n = tl.sqrt(tl.sum(x * x, axis=0))
    y = x * tl.minimum(threshold / (n + 1e-6), 1.0)
    if TAU:
        if ROUND:
            y = y.to(tl.bfloat16).to(tl.float32)
        y = y * tl.load(tau_ptr + r // heads)
    tl.store(out_ptr + r * D + d, y.to(out_ptr.dtype.element_ty), mask=dm)


# -- small-product ------------------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=[])
def small_product_kernel(r_ptr, proj_ptr, out_ptr, E, R: tl.constexpr, BLOCK: tl.constexpr):
    # One row's product with proj [R][E]: its R values (rounded to BF16 first) times proj in FP32, summed in index
    # order, the result rounded to BF16 (a BF16 product's output).
    row = tl.program_id(0)
    e = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    em = e < E
    acc = tl.zeros([BLOCK], tl.float32)
    for i in range(0, R):
        rv = tl.load(r_ptr + row * R + i).to(tl.bfloat16).to(tl.float32)
        acc += rv * tl.load(proj_ptr + i * E + e, mask=em, other=0.0)
    tl.store(out_ptr + row * E + e, acc.to(tl.bfloat16).to(tl.float32), mask=em)


# -- sink-gate ----------------------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=[])
def sink_gate_kernel(a_ptr, lse_ptr, sink_ptr, stride_sink, g_ptr, stride_g, out_ptr, heads, D, BLOCK: tl.constexpr):
    # One token head: a * sigmoid(lse - sink) * sigmoid(g), in BF16.
    r = tl.program_id(0)
    t = r // heads
    hd = r % heads
    d = tl.arange(0, BLOCK)
    dm = d < D
    a = tl.load(a_ptr + r * D + d, mask=dm, other=0.0).to(tl.float32)
    keep = 1.0 / (1.0 + tl.exp(tl.load(sink_ptr + t * stride_sink + hd).to(tl.float32) - tl.load(lse_ptr + r)))
    g = tl.load(g_ptr + t * stride_g + hd * D + d, mask=dm, other=0.0).to(tl.float32)
    tl.store(out_ptr + r * D + d, (a * keep * (1.0 / (1.0 + tl.exp(-g)))).to(tl.bfloat16), mask=dm)


# -- scaled-add ---------------------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=["n"])
def scaled_add_kernel(y_ptr, x_ptr, out_ptr, n, scale, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    y = tl.load(y_ptr + i, mask=m, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + i, mask=m, other=0.0).to(tl.float32)
    tl.store(out_ptr + i, ((y + x) * scale).to(out_ptr.dtype.element_ty), mask=m)


# -- summary-mix --------------------------------------------------------------------------------------------------------
@triton.jit
def _vectors(blocks_ptr, h_ptr, t, stride_bt, stride_bn, stride_ht, N, H, c, j):
    cm = c < H
    vb = tl.load(blocks_ptr + t * stride_bt + j[:, None] * stride_bn + c[None, :],
                 mask=(j[:, None] < N) & cm[None, :], other=0.0).to(tl.float32)
    vh = tl.load(h_ptr + t * stride_ht + c, mask=cm, other=0.0).to(tl.float32)
    return tl.where((j == N)[:, None], vh[None, :], vb)                         # [NP, CHUNK]


@triton.jit(do_not_specialize=[])
def mix_parts_kernel(blocks_ptr, h_ptr, q_ptr, part_ptr, stride_bt, stride_bn, stride_ht, N, H, CHUNKS,
                     Q: tl.constexpr, NP: tl.constexpr, CHUNK: tl.constexpr):
    # Program (token, chunk): its chunk's sums of squares of the N summaries and the row (index N) and their dot
    # products with each query, into the partials [t][CHUNKS][NP + Q * NP].
    t = tl.program_id(0)
    ch = tl.program_id(1)
    j = tl.arange(0, NP)
    k = tl.arange(0, Q)
    c = ch * CHUNK + tl.arange(0, CHUNK)
    v = _vectors(blocks_ptr, h_ptr, t, stride_bt, stride_bn, stride_ht, N, H, c, j)
    qv = tl.load(q_ptr + k[:, None] * H + c[None, :], mask=(c < H)[None, :], other=0.0)   # [Q, CHUNK]
    at = part_ptr + (t * CHUNKS + ch) * (NP + Q * NP)
    tl.store(at + j, tl.sum(v * v, axis=1))
    tl.store(at + NP + k[:, None] * NP + j[None, :], tl.sum(qv[:, None, :] * v[None, :, :], axis=2))


@triton.jit(do_not_specialize=["stride_oq"])
def mix_kernel(blocks_ptr, h_ptr, part_ptr, out_ptr, stride_bt, stride_bn, stride_ht, stride_oq, N, H, CHUNKS,
               scale, eps, Q: tl.constexpr, NP: tl.constexpr, CHUNK: tl.constexpr):
    # Program (token, chunk): the partials summed over the chunks in ascending order (the same sums in every program
    # of the token), each vector's RMS, the softmax over the N + 1 a query, and the chunk's weighted sum.
    t = tl.program_id(0)
    ch = tl.program_id(1)
    j = tl.arange(0, NP)
    k = tl.arange(0, Q)
    ss = tl.zeros([NP], tl.float32)
    dots = tl.zeros([Q, NP], tl.float32)
    for p in range(0, CHUNKS):
        at = part_ptr + (t * CHUNKS + p) * (NP + Q * NP)
        ss += tl.load(at + j)
        dots += tl.load(at + NP + k[:, None] * NP + j[None, :])
    inv = 1.0 / tl.sqrt(ss / H + eps)                                          # [NP]
    s = tl.where((j <= N)[None, :], dots * inv[None, :] * scale, float("-inf"))
    m = tl.max(s, axis=1)
    e = tl.exp(s - m[:, None])
    w = e / tl.sum(e, axis=1)[:, None] * inv[None, :]                          # [Q, NP]
    c = ch * CHUNK + tl.arange(0, CHUNK)
    v = _vectors(blocks_ptr, h_ptr, t, stride_bt, stride_bn, stride_ht, N, H, c, j)
    o = tl.sum(w[:, :, None] * v[None, :, :], axis=1)                           # [Q, CHUNK]
    tl.store(out_ptr + k[:, None] * stride_oq + t * H + c[None, :], o, mask=(c < H)[None, :])


# -- pair-norm ----------------------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=[])
def pair_norm_kernel(x_ptr, m_ptr, out_ptr, stride_x, stride_m, H, eps, BLOCK: tl.constexpr):
    # One row: [rms(x), rms(m)] in BF16, each normed without a weight; the sums of squares over BLOCK lanes.
    t = tl.program_id(0)
    sx = tl.zeros([BLOCK], tl.float32)
    sm = tl.zeros([BLOCK], tl.float32)
    for c0 in range(0, H, BLOCK):
        c = c0 + tl.arange(0, BLOCK)
        cm = c < H
        x = tl.load(x_ptr + t * stride_x + c, mask=cm, other=0.0).to(tl.float32)
        m = tl.load(m_ptr + t * stride_m + c, mask=cm, other=0.0).to(tl.float32)
        sx += x * x
        sm += m * m
    ix = 1.0 / tl.sqrt(tl.sum(sx, axis=0) / H + eps)
    im = 1.0 / tl.sqrt(tl.sum(sm, axis=0) / H + eps)
    for c0 in range(0, H, BLOCK):
        c = c0 + tl.arange(0, BLOCK)
        cm = c < H
        x = tl.load(x_ptr + t * stride_x + c, mask=cm, other=0.0).to(tl.float32)
        m = tl.load(m_ptr + t * stride_m + c, mask=cm, other=0.0).to(tl.float32)
        tl.store(out_ptr + t * 2 * H + c, (x * ix).to(tl.bfloat16), mask=cm)
        tl.store(out_ptr + t * 2 * H + H + c, (m * im).to(tl.bfloat16), mask=cm)


# -- gated-join ---------------------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=[])
def gated_join_kernel(x_ptr, m_ptr, f_ptr, a_ptr, out_ptr, stride_x, stride_m, stride_f, H, BLOCK: tl.constexpr):
    # (1 - tanh(f)) * x + tanh(a) * m, FP32.
    t = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    cm = c < H
    x = tl.load(x_ptr + t * stride_x + c, mask=cm, other=0.0).to(tl.float32)
    m = tl.load(m_ptr + t * stride_m + c, mask=cm, other=0.0).to(tl.float32)
    f = tl.load(f_ptr + t * stride_f + c, mask=cm, other=0.0)
    a = tl.load(a_ptr + t * stride_f + c, mask=cm, other=0.0)
    tf = 2.0 / (1.0 + tl.exp(-2.0 * f)) - 1.0
    ta = 2.0 / (1.0 + tl.exp(-2.0 * a)) - 1.0
    tl.store(out_ptr + t * H + c, (1.0 - tf) * x + ta * m, mask=cm)


# -- low-rank-gates -----------------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=["T"])
def gate_down_kernel(x_ptr, m_ptr, w_ptr, g_ptr, T, H, R, sx_t, sx_g, sm_t, sm_g, eps, BR: tl.constexpr,
                     BK: tl.constexpr):
    # Program (gate, block of the rank, row t): g = down . [rms(x), rms(m)] (the normed inputs in BF16, FP32 sums over
    # the inputs in BK lanes, g in BF16).
    gi = tl.program_id(0)
    r = tl.program_id(1) * BR + tl.arange(0, BR)
    rm = r < R
    w_at = w_ptr + gi * R * 2 * H + r[:, None] * 2 * H
    t = tl.program_id(2)
    if t < T:
        xa = x_ptr + t * sx_t + gi * sx_g
        ma = m_ptr + t * sm_t + gi * sm_g
        sx = tl.zeros([BK], tl.float32)
        sm = tl.zeros([BK], tl.float32)
        for c0 in range(0, H, BK):
            c = c0 + tl.arange(0, BK)
            cm = c < H
            xv = tl.load(xa + c, mask=cm, other=0.0).to(tl.float32)
            mv = tl.load(ma + c, mask=cm, other=0.0).to(tl.float32)
            sx += xv * xv
            sm += mv * mv
        ix = 1.0 / tl.sqrt(tl.sum(sx, axis=0) / H + eps)
        im = 1.0 / tl.sqrt(tl.sum(sm, axis=0) / H + eps)
        acc = tl.zeros([BR, BK], tl.float32)
        for c0 in range(0, H, BK):
            c = c0 + tl.arange(0, BK)
            cm = c < H
            xv = (tl.load(xa + c, mask=cm, other=0.0).to(tl.float32) * ix).to(tl.bfloat16).to(tl.float32)
            mv = (tl.load(ma + c, mask=cm, other=0.0).to(tl.float32) * im).to(tl.bfloat16).to(tl.float32)
            wx = tl.load(w_at + c[None, :], mask=rm[:, None] & cm[None, :], other=0.0).to(tl.float32)
            wm = tl.load(w_at + H + c[None, :], mask=rm[:, None] & cm[None, :], other=0.0).to(tl.float32)
            acc += wx * xv[None, :] + wm * mv[None, :]
        g = tl.sum(acc, axis=1).to(tl.bfloat16).to(tl.float32)
        tl.store(g_ptr + (gi * T + t) * R + r, g, mask=rm)


@triton.jit(do_not_specialize=["T"])
def gate_up_kernel(g_ptr, w_ptr, x_ptr, m_ptr, out_ptr, T, H, R, sx_t, sx_g, sm_t, sm_g, G: tl.constexpr,
                   BH: tl.constexpr, RP: tl.constexpr):
    # Program (gate, block of the hidden, row t): f, a = [forget; gain] . g, then (1 - tanh f) x + tanh(a) m; out
    # [t][gates][hidden] FP32.
    gi = tl.program_id(0)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    hm = h < H
    j = tl.arange(0, RP)
    jm = j < R
    w_at = w_ptr + gi * 2 * H * R
    wf = tl.load(w_at + h[:, None] * R + j[None, :], mask=hm[:, None] & jm[None, :], other=0.0).to(tl.float32)
    wa = tl.load(w_at + (H + h)[:, None] * R + j[None, :], mask=hm[:, None] & jm[None, :], other=0.0).to(tl.float32)
    t = tl.program_id(2)
    if t < T:
        gv = tl.load(g_ptr + (gi * T + t) * R + j, mask=jm, other=0.0)
        f = tl.sum(wf * gv[None, :], axis=1)
        a = tl.sum(wa * gv[None, :], axis=1)
        xv = tl.load(x_ptr + t * sx_t + gi * sx_g + h, mask=hm, other=0.0).to(tl.float32)
        mv = tl.load(m_ptr + t * sm_t + gi * sm_g + h, mask=hm, other=0.0).to(tl.float32)
        tf = 2.0 / (1.0 + tl.exp(-2.0 * f)) - 1.0
        ta = 2.0 / (1.0 + tl.exp(-2.0 * a)) - 1.0
        tl.store(out_ptr + (t * G + gi) * H + h, (1.0 - tf) * xv + ta * mv, mask=hm)


# -- span-attention -----------------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=[])
def span_part_kernel(q_ptr, k_ptr, v_ptr, first_ptr, last_ptr, acc_ptr, ml_ptr, T, KVH, SPLITS, scale, G: tl.constexpr,
                     D: tl.constexpr, CHUNK: tl.constexpr, BN: tl.constexpr):
    # Program (row's kv head, split): its queries [G][D] over the keys of positions first .. last within its split's
    # CHUNK positions, online softmax in FP32 in steps of BN keys; partial sums [G][D] and (max, sum) [G][2].
    rk = tl.program_id(0)
    sp = tl.program_id(1)
    r = rk // KVH
    lo = tl.maximum(sp * CHUNK, tl.load(first_ptr + r))
    hi = tl.minimum(sp * CHUNK + CHUNK, tl.load(last_ptr + r) + 1)
    g = tl.arange(0, G)
    d = tl.arange(0, D)
    m_i = tl.full([G], float("-inf"), tl.float32)
    l_i = tl.zeros([G], tl.float32)
    acc = tl.zeros([G, D], tl.float32)
    if lo < hi:
        q = tl.load(q_ptr + rk * G * D + g[:, None] * D + d[None, :])
        base = rk.to(tl.int64) * T * D
        for start in range(lo, hi, BN):
            n = start + tl.arange(0, BN)
            nm = n < hi
            kt = tl.load(k_ptr + base + n[None, :] * D + d[:, None], mask=nm[None, :], other=0.0)
            sc = tl.dot(q, kt) * scale
            sc = tl.where(nm[None, :], sc, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(sc, 1))
            p = tl.exp(sc - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            vt = tl.load(v_ptr + base + n[:, None] * D + d[None, :], mask=nm[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vt)
            m_i = m_new
    at = rk * SPLITS + sp
    tl.store(acc_ptr + at * G * D + g[:, None] * D + d[None, :], acc)
    tl.store(ml_ptr + at * G * 2 + g * 2, m_i)
    tl.store(ml_ptr + at * G * 2 + g * 2 + 1, l_i)


@triton.jit(do_not_specialize=[])
def span_merge_kernel(acc_ptr, ml_ptr, out_ptr, SPLITS, G: tl.constexpr, D: tl.constexpr):
    # One row's kv head: its splits' partials merged in split order, BF16 out.
    rk = tl.program_id(0)
    g = tl.arange(0, G)
    d = tl.arange(0, D)
    m = tl.full([G], float("-inf"), tl.float32)
    for sp in range(0, SPLITS):
        m = tl.maximum(m, tl.load(ml_ptr + (rk * SPLITS + sp) * G * 2 + g * 2))
    l = tl.zeros([G], tl.float32)
    acc = tl.zeros([G, D], tl.float32)
    for sp in range(0, SPLITS):
        at = rk * SPLITS + sp
        w = tl.exp(tl.load(ml_ptr + at * G * 2 + g * 2) - m)
        l += w * tl.load(ml_ptr + at * G * 2 + g * 2 + 1)
        acc += w[:, None] * tl.load(acc_ptr + at * G * D + g[:, None] * D + d[None, :])
    tl.store(out_ptr + rk * G * D + g[:, None] * D + d[None, :], (acc / l[:, None]).to(tl.bfloat16))
