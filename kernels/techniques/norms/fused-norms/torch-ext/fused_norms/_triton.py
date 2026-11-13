"""The Triton kernels behind fused_norms' functions. Every kernel computes in FP32 and rounds once, on store.

Row invariance: a program covers HB rows of BD values (HB and BD and the warp count follow from the row width and the
knobs tuned for that width, never from the row count), and each row reduces in the order that layout gives it, so a
row has the same bits in a call of one row and in a call of thousands. The row count enters only the grid and the row
mask (never specialized on). Block sizes and warp counts are tl.constexpr / launch parameters the package reads from
the build's tuning (kernel.json's knobs).
"""
import triton
import triton.language as tl
from triton.language.extra import libdevice

try:   # programmatic dependent launch (sm_90 and newer): griddepcontrol.wait / launch_dependents
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
    HAS_GDC = True
except ImportError:   # an older Triton: the kernels launch without it (PDL 0)
    HAS_GDC = False

    @triton.jit
    def gdc_wait():
        pass

    @triton.jit
    def gdc_launch_dependents():
        pass

from crit_kcore import epilogue   # the kernel core's sigmoid and activations (kernels/core)

# NORM: 0 none, 1 RMS (mean of squares), 2 L2 (sum of squares). GATE: 0 none, 1 silu, 2 sigmoid.
# ROPE: 0 none, 1 half (pairs j, j + R / 2), 2 interleaved (pairs 2j, 2j + 1). ACT: 0 silu, 1 gelu, 2 gelu_tanh,
# 3 sigmoid, 4 none (the gate itself): crit_kcore.epilogue's act codes.


@triton.jit
def _rows_body(pid, x_ptr, sxa, sxb, r_ptr, sra, srb, z_ptr, sza, szb, w_ptr, groups, swg, y_ptr, sya, syb,
               pos_ptr, f_ptr, sf, n_table, n_rows, B, n_blocks, D, R, eps, offset, scale, xw_ptr, sxwa, sxwb,
               NORM: tl.constexpr, HAS_RES: tl.constexpr, HAS_W: tl.constexpr, GROUPED: tl.constexpr,
               GATE: tl.constexpr, ROPE: tl.constexpr, TABLE: tl.constexpr, IN_PLACE: tl.constexpr,
               HAS_XW: tl.constexpr, HB: tl.constexpr, BD: tl.constexpr):
    """Rows of D values at x[a * sxa + b * sxb + d]: without rope, rows r = a * B + b of n_rows, HB consecutive rows a
    program; with rope, a program is token a's heads b of one block of HB (n_blocks blocks a token). GROUPED: the
    weight has one row (swg apart) per group, row r (head b with rope) taking row r % groups (b % groups). IN_PLACE
    (rope writing into x): every thread's loads, its partners' included, finish before any store. HAS_XW (with a
    residual): x of each row times its own weight xw[a * sxwa + b * sxwb] before the add (a row added into every
    stream: sxb 0)."""
    if ROPE != 0:
        a = (pid // n_blocks).to(tl.int64)
        b = (pid % n_blocks) * HB + tl.arange(0, HB)
        rm = b < B
        b = b.to(tl.int64)
        a_rows = a + tl.zeros([HB], dtype=tl.int64)
    else:
        r = pid * HB + tl.arange(0, HB)
        rm = r < n_rows
        a_rows = (r // B).to(tl.int64)
        b = (r % B).to(tl.int64)
    if GROUPED:
        if ROPE != 0:
            w_rows = (b % groups) * swg
        else:
            w_rows = (r % groups).to(tl.int64) * swg
    d = tl.arange(0, BD)
    dm = d < D
    m = rm[:, None] & dm[None, :]
    x_rows = a_rows * sxa + b * sxb
    x = tl.load(x_ptr + x_rows[:, None] + d[None, :], mask=m, other=0.0).to(tl.float32)
    if HAS_XW:
        x = x * tl.load(xw_ptr + a_rows * sxwa + b * sxwb, mask=rm, other=0.0).to(tl.float32)[:, None]
    if HAS_RES:
        r_rows = a_rows * sra + b * srb
        s = (tl.load(r_ptr + r_rows[:, None] + d[None, :], mask=m, other=0.0).to(tl.float32) + x)
        s = s.to(r_ptr.dtype.element_ty)
        tl.store(r_ptr + r_rows[:, None] + d[None, :], s, mask=m)
        x = s.to(tl.float32)
    if NORM == 1:
        inv = tl.math.rsqrt(tl.sum(x * x, axis=1) / D + eps)
    elif NORM == 2:
        inv = tl.math.rsqrt(tl.sum(x * x, axis=1) + eps)
    else:
        inv = tl.full([HB], 1.0, tl.float32)
    y = x * inv[:, None]
    if HAS_W:
        if GROUPED:
            y = y * (tl.load(w_ptr + w_rows[:, None] + d[None, :], mask=m, other=0.0).to(tl.float32) + offset)
        else:
            y = y * (tl.load(w_ptr + d, mask=dm, other=0.0).to(tl.float32) + offset)[None, :]
    if ROPE != 0:
        half = R // 2
        if ROPE == 1:
            low = d < half
            p = tl.where(low, d + half, d - half)
            j = tl.where(low, d, d - half)
            sign = tl.where(low, -1.0, 1.0)
        else:
            p = d ^ 1
            j = d // 2
            sign = tl.where((d % 2) == 0, -1.0, 1.0)
        turn = d < R
        p = tl.where(turn, p, d)
        j = tl.where(turn, j, 0)
        xp = tl.load(x_ptr + x_rows[:, None] + p[None, :], mask=m & turn[None, :], other=0.0).to(tl.float32)
        yp = xp * inv[:, None]
        if HAS_W:
            if GROUPED:
                yp = yp * (tl.load(w_ptr + w_rows[:, None] + p[None, :], mask=m & turn[None, :],
                                   other=0.0).to(tl.float32) + offset)
            else:
                yp = yp * (tl.load(w_ptr + p, mask=dm & turn, other=0.0).to(tl.float32) + offset)[None, :]
        position = tl.load(pos_ptr + a)
        if TABLE:
            position = tl.minimum(tl.maximum(position, 0), n_table - 1)
            c = tl.load(f_ptr + position * sf + j, mask=turn, other=1.0)
            s = tl.load(f_ptr + position * sf + half + j, mask=turn, other=0.0)
        else:
            angle = position.to(tl.float32) * tl.load(f_ptr + j, mask=turn, other=0.0)
            c = tl.cos(angle)
            s = tl.sin(angle)
        c = c * scale
        s = s * sign * scale
        y = tl.where(turn[None, :], y * c[None, :] + yp * s[None, :], y)
    if GATE != 0:
        z_rows = a_rows * sza + b * szb
        g = tl.load(z_ptr + z_rows[:, None] + d[None, :], mask=m, other=0.0).to(tl.float32)
        if GATE == 1:
            y = y * (g * epilogue.sigmoid(g))
        else:
            y = y * epilogue.sigmoid(g)
    y_rows = a_rows * sya + b * syb
    if IN_PLACE:
        tl.debug_barrier()
    tl.store(y_ptr + y_rows[:, None] + d[None, :], y.to(y_ptr.dtype.element_ty), mask=m)


@triton.jit(do_not_specialize=["n_rows"])
def rows_kernel(x_ptr, sxa, sxb, r_ptr, sra, srb, z_ptr, sza, szb, w_ptr, groups, swg, y_ptr, sya, syb,
                pos_ptr, f_ptr, sf, n_table, n_rows, B, n_blocks, D, R, eps, offset, scale, xw_ptr, sxwa, sxwb,
                NORM: tl.constexpr, HAS_RES: tl.constexpr, HAS_W: tl.constexpr, GROUPED: tl.constexpr,
                GATE: tl.constexpr, ROPE: tl.constexpr, TABLE: tl.constexpr, IN_PLACE: tl.constexpr,
                HAS_XW: tl.constexpr, HB: tl.constexpr, BD: tl.constexpr):
    """Every row of one tensor (_rows_body)."""
    _rows_body(tl.program_id(0), x_ptr, sxa, sxb, r_ptr, sra, srb, z_ptr, sza, szb, w_ptr, groups, swg, y_ptr, sya,
               syb, pos_ptr, f_ptr, sf, n_table, n_rows, B, n_blocks, D, R, eps, offset, scale, xw_ptr, sxwa, sxwb,
               NORM, HAS_RES, HAS_W, GROUPED, GATE, ROPE, TABLE, IN_PLACE, HAS_XW, HB, BD)


@triton.jit(do_not_specialize=["tokens"])
def pair_kernel(x_ptr, sxa, sxb, w_ptr, groups, swg, y_ptr, sya, syb, B, n_blocks,
                x2_ptr, sx2a, sx2b, w2_ptr, groups2, swg2, y2_ptr, sy2a, sy2b, B2, n_blocks2, tokens,
                pos_ptr, f_ptr, sf, n_table, D, R, eps, offset, scale,
                NORM: tl.constexpr, HAS_W: tl.constexpr, GROUPED: tl.constexpr, GROUPED2: tl.constexpr,
                ROPE: tl.constexpr, TABLE: tl.constexpr, IN_PLACE: tl.constexpr, IN_PLACE2: tl.constexpr,
                HB: tl.constexpr, BD: tl.constexpr):
    """Two head tensors of the same tokens (queries and keys) in one launch: the first tokens * n_blocks programs the
    first tensor's heads, the rest the second's, each the body rope rows run (_rows_body), its own weight."""
    pid = tl.program_id(0)
    first = tokens * n_blocks
    if pid < first:
        _rows_body(pid, x_ptr, sxa, sxb, x_ptr, 0, 0, x_ptr, 0, 0, w_ptr, groups, swg, y_ptr, sya, syb,
                   pos_ptr, f_ptr, sf, n_table, 0, B, n_blocks, D, R, eps, offset, scale, x_ptr, 0, 0,
                   NORM, False, HAS_W, GROUPED, 0, ROPE, TABLE, IN_PLACE, False, HB, BD)
    else:
        _rows_body(pid - first, x2_ptr, sx2a, sx2b, x2_ptr, 0, 0, x2_ptr, 0, 0, w2_ptr, groups2, swg2, y2_ptr, sy2a,
                   sy2b, pos_ptr, f_ptr, sf, n_table, 0, B2, n_blocks2, D, R, eps, offset, scale, x2_ptr, 0, 0,
                   NORM, False, HAS_W, GROUPED2, 0, ROPE, TABLE, IN_PLACE2, False, HB, BD)


@triton.jit
def softplus_decay_kernel(x_ptr, sxa, b_ptr, rate_ptr, g_ptr, sga, y_ptr, sya, beta_ptr, sba, N, n_blocks,
                          HAS_GATE: tl.constexpr, BLOCK: tl.constexpr):
    """y[t][i] = -rate[i] * softplus(x[t][i] + bias[i]) (softplus(v) = v past 20, log1p(exp(v)) below), and with a
    gate beta[t][i] = sigmoid(g[t][i]), BLOCK values of one row a program; FP32 inside, with the CUDA math library's
    exp and log1p and an IEEE division (the values torch's own softplus and sigmoid give on CUDA)."""
    pid = tl.program_id(0)
    t = (pid // n_blocks).to(tl.int64)
    i = (pid % n_blocks) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    v = tl.load(x_ptr + t * sxa + i, mask=m, other=0.0).to(tl.float32)
    v = v + tl.load(b_ptr + i, mask=m, other=0.0).to(tl.float32)
    soft = tl.where(v > 20.0, v, libdevice.log1p(libdevice.exp(v)))       # exp overflowing past 88 is not chosen
    rate = tl.load(rate_ptr + i, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + t * sya + i, (-rate * soft).to(y_ptr.dtype.element_ty), mask=m)
    if HAS_GATE:
        g = tl.load(g_ptr + t * sga + i, mask=m, other=0.0).to(tl.float32)
        beta = tl.math.div_rn(tl.full([BLOCK], 1.0, tl.float32), 1.0 + libdevice.exp(-g))
        tl.store(beta_ptr + t * sba + i, beta.to(beta_ptr.dtype.element_ty), mask=m)


@triton.jit
def long_rows_kernel(x_ptr, sxa, sxb, r_ptr, sra, srb, z_ptr, sza, szb, w_ptr, groups, swg, y_ptr, sya, syb, B, D, eps,
                     offset, NORM: tl.constexpr, HAS_RES: tl.constexpr, HAS_W: tl.constexpr, GATE: tl.constexpr,
                     CHUNK: tl.constexpr, ONE_PASS: tl.constexpr = False, PDL: tl.constexpr = 0):
    """One row wider than one block per program: ONE_PASS (CHUNK at least the row's width), the row held in registers
    and read once, its sum of squares one reduction over CHUNK lanes; else two passes of CHUNK values: the sums of
    squares (adding the residual and storing it), then the normalized row. A fixed order whatever the row count. PDL
    as act_mul_kernel's (programmatic dependent launch: speed only)."""
    r = tl.program_id(0)
    a = (r // B).to(tl.int64)
    b = (r % B).to(tl.int64)
    x_row = x_ptr + a * sxa + b * sxb
    r_row = r_ptr + a * sra + b * srb
    if PDL != 0:
        gdc_wait()
    if ONE_PASS:
        d = tl.arange(0, CHUNK)
        dm = d < D
        x = tl.load(x_row + d, mask=dm, other=0.0).to(tl.float32)
        if HAS_RES:
            s = (tl.load(r_row + d, mask=dm, other=0.0).to(tl.float32) + x).to(r_ptr.dtype.element_ty)
            tl.store(r_row + d, s, mask=dm)
            x = s.to(tl.float32)
        if PDL == 1:
            gdc_launch_dependents()
        if NORM == 1:
            inv = tl.math.rsqrt(tl.sum(x * x, axis=0) / D + eps)
        elif NORM == 2:
            inv = tl.math.rsqrt(tl.sum(x * x, axis=0) + eps)
        else:
            inv = 1.0
        y = x * inv
        if HAS_W:
            y = y * (tl.load(w_ptr + (r % groups).to(tl.int64) * swg + d, mask=dm, other=0.0).to(tl.float32) + offset)
        if GATE != 0:
            g = tl.load(z_ptr + a * sza + b * szb + d, mask=dm, other=0.0).to(tl.float32)
            if GATE == 1:
                y = y * (g * epilogue.sigmoid(g))
            else:
                y = y * epilogue.sigmoid(g)
        tl.store(y_ptr + a * sya + b * syb + d, y.to(y_ptr.dtype.element_ty), mask=dm)
        if PDL == 2:
            gdc_launch_dependents()
        return
    acc = tl.zeros([CHUNK], dtype=tl.float32)
    for start in range(0, D, CHUNK):
        d = start + tl.arange(0, CHUNK)
        dm = d < D
        x = tl.load(x_row + d, mask=dm, other=0.0).to(tl.float32)
        if HAS_RES:
            s = (tl.load(r_row + d, mask=dm, other=0.0).to(tl.float32) + x).to(r_ptr.dtype.element_ty)
            tl.store(r_row + d, s, mask=dm)
            x = s.to(tl.float32)
        acc += x * x
    if NORM == 1:
        inv = tl.math.rsqrt(tl.sum(acc, axis=0) / D + eps)
    elif NORM == 2:
        inv = tl.math.rsqrt(tl.sum(acc, axis=0) + eps)
    else:
        inv = 1.0
    z_row = z_ptr + a * sza + b * szb
    y_row = y_ptr + a * sya + b * syb
    w_row = w_ptr + (r % groups).to(tl.int64) * swg
    for start in range(0, D, CHUNK):
        d = start + tl.arange(0, CHUNK)
        dm = d < D
        if HAS_RES:
            x = tl.load(r_row + d, mask=dm, other=0.0).to(tl.float32)
        else:
            x = tl.load(x_row + d, mask=dm, other=0.0).to(tl.float32)
        y = x * inv
        if HAS_W:
            y = y * (tl.load(w_row + d, mask=dm, other=0.0).to(tl.float32) + offset)
        if GATE != 0:
            g = tl.load(z_row + d, mask=dm, other=0.0).to(tl.float32)
            if GATE == 1:
                y = y * (g * epilogue.sigmoid(g))
            else:
                y = y * epilogue.sigmoid(g)
        tl.store(y_row + d, y.to(y_ptr.dtype.element_ty), mask=dm)


@triton.jit(do_not_specialize=["n_rows"])
def act_mul_kernel(g_ptr, sga, sgb, u_ptr, sua, sub, y_ptr, sya, syb, n_rows, B, N, n_blocks,
                   ACT: tl.constexpr, INTERLEAVED: tl.constexpr, BLOCK: tl.constexpr, PDL: tl.constexpr = 0):
    """y[row][i] = act(g[row][i]) * u[row][i] for BLOCK columns of one row a program; INTERLEAVED: g and u alternate
    in one row of x (g_ptr), read as pairs. PDL (launched with programmatic dependent launch): 0 none; 1, 2, 3 wait
    for the kernel before (griddepcontrol.wait: its writes visible) before any load, so the program's start overlaps
    its end, and let the kernel after start once this program has loaded its inputs (1), stored its outputs (2), or
    when it exits (3). Speed only: the values computed are the same."""
    pid = tl.program_id(0)
    row = pid // n_blocks
    a = (row // B).to(tl.int64)
    b = (row % B).to(tl.int64)
    i = (pid % n_blocks) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    if PDL != 0:
        gdc_wait()
    if INTERLEAVED:
        pair = tl.arange(0, 2)
        both = tl.load(g_ptr + a * sga + b * sgb + (2 * i)[:, None] + pair[None, :], mask=m[:, None],
                       other=0.0).to(tl.float32)
        g, u = tl.split(both)
    else:
        g = tl.load(g_ptr + a * sga + b * sgb + i, mask=m, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + a * sua + b * sub + i, mask=m, other=0.0).to(tl.float32)
    if PDL == 1:
        gdc_launch_dependents()
    y = epilogue.act(g, ACT) * u
    tl.store(y_ptr + a * sya + b * syb + i, y.to(y_ptr.dtype.element_ty), mask=m)
    if PDL == 2:
        gdc_launch_dependents()


@triton.jit
def stream_sum_kernel(x_ptr, sxt, sxs, g_ptr, sgt, sgs, sgd, y_ptr, syt, S, D, n_blocks, scale,
                      HAS_GATE: tl.constexpr, ACT: tl.constexpr, BLOCK: tl.constexpr):
    """y[t][d] = scale * the sum over s = 0 .. S - 1, in that order, of act(g[t][s][d]) * x[t][s][d] (g one value per
    stream when sgd is 0; without a gate the plain sum), BLOCK values of one token a program."""
    pid = tl.program_id(0)
    t = (pid // n_blocks).to(tl.int64)
    d = (pid % n_blocks) * BLOCK + tl.arange(0, BLOCK)
    m = d < D
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for s in range(0, S):
        x = tl.load(x_ptr + t * sxt + s * sxs + d, mask=m, other=0.0).to(tl.float32)
        if HAS_GATE:
            g = tl.load(g_ptr + t * sgt + s * sgs + d * sgd, mask=m, other=0.0).to(tl.float32)
            x = epilogue.act(g, ACT) * x
        acc += x
    tl.store(y_ptr + t * syt + d, (acc * scale).to(y_ptr.dtype.element_ty), mask=m)


@triton.jit
def stream_add_kernel(h_ptr, sht, shs, x_ptr, sxt, w_ptr, swt, sws, S, D, n_blocks,
                      HAS_W: tl.constexpr, BLOCK: tl.constexpr):
    """h[t][s][d] += w[t][s] * x[t][d] for every stream s (w 1 without weights), in place, BLOCK values of one token a
    program; FP32 inside, stored in h's dtype."""
    pid = tl.program_id(0)
    t = (pid // n_blocks).to(tl.int64)
    d = (pid % n_blocks) * BLOCK + tl.arange(0, BLOCK)
    m = d < D
    x = tl.load(x_ptr + t * sxt + d, mask=m, other=0.0).to(tl.float32)
    for s in range(0, S):
        at = h_ptr + t * sht + s * shs + d
        if HAS_W:
            add = tl.load(w_ptr + t * swt + s * sws).to(tl.float32) * x
        else:
            add = x
        tl.store(at, (tl.load(at, mask=m, other=0.0).to(tl.float32) + add).to(h_ptr.dtype.element_ty), mask=m)


# QUANT: 1 FP8 rows (E4M3 codes, one FP32 scale a row: amax / 448, at least 1e-12), 2 NVFP4 (E2M1 codes, one E4M3 scale
# a 16-value block under the input scale g, in cuBLASLt's 128 x 4 tiled layout). The quantizers' arithmetic is that of
# the projections' quantizers (mma-rows quantize_fp8_rows and quantize_nvfp4), on the normalized row rounded to BF16
# as rms_norm would write it, so the codes and scales are those of rms_norm followed by the quantizer.

@triton.jit
def _e2m1_codes(v):
    """The nearest E2M1 code of each v (already over its block's scale), ties to the even code, saturating at 6,
    bit 3 set where v < 0; NaN gives 0."""
    a = tl.abs(v)
    code = ((a > 0.25).to(tl.int32) + (a >= 0.75).to(tl.int32) + (a > 1.25).to(tl.int32)
            + (a >= 1.75).to(tl.int32) + (a > 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32) + (a > 5.0).to(tl.int32))
    return code | tl.where(v < 0, 8, 0)


@triton.jit(do_not_specialize=["n_rows"])
def rows_quant_kernel(x_ptr, sxa, sxb, r_ptr, sra, srb, w_ptr, q_ptr, sq, s_ptr, n_rows, B, D, eps, offset, g, g_rcp,
                      ct, HAS_RES: tl.constexpr, HAS_W: tl.constexpr, QUANT: tl.constexpr, HB: tl.constexpr,
                      BD: tl.constexpr):
    """rms_norm's rows (_rows_body without rope, gates or groups: the same loads, sums and products in the same
    layout, so the same bits) quantized for a projection: codes row r at q[r * sq], scales at s."""
    pid = tl.program_id(0)
    r = pid * HB + tl.arange(0, HB)
    rm = r < n_rows
    a_rows = (r // B).to(tl.int64)
    b = (r % B).to(tl.int64)
    d = tl.arange(0, BD)
    dm = d < D
    m = rm[:, None] & dm[None, :]
    x_rows = a_rows * sxa + b * sxb
    x = tl.load(x_ptr + x_rows[:, None] + d[None, :], mask=m, other=0.0).to(tl.float32)
    if HAS_RES:
        r_rows = a_rows * sra + b * srb
        s = (tl.load(r_ptr + r_rows[:, None] + d[None, :], mask=m, other=0.0).to(tl.float32) + x)
        s = s.to(r_ptr.dtype.element_ty)
        tl.store(r_ptr + r_rows[:, None] + d[None, :], s, mask=m)
        x = s.to(tl.float32)
    inv = tl.math.rsqrt(tl.sum(x * x, axis=1) / D + eps)
    y = x * inv[:, None]
    if HAS_W:
        y = y * (tl.load(w_ptr + d, mask=dm, other=0.0).to(tl.float32) + offset)[None, :]
    y = tl.where(m, y.to(tl.bfloat16).to(tl.float32), 0.0)   # the BF16 row the quantizer would read
    rows = r.to(tl.int64)
    if QUANT == 1:
        nan = tl.max(tl.where(y != y, 1, 0), axis=1)
        amax = tl.where(nan > 0, float("nan"), tl.max(tl.abs(y), axis=1))
        sc = amax * (1.0 / 448.0)
        sc = tl.where(sc < 1e-12, 1e-12, sc)
        v = tl.math.div_rn(y, sc[:, None])
        v = tl.where(v > 448.0, 448.0, tl.where(v < -448.0, -448.0, v))
        codes = v.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        tl.store(q_ptr + rows[:, None] * sq + d[None, :], codes, mask=m)
        tl.store(s_ptr + rows, sc, mask=rm)
    else:
        blocks = tl.reshape(y, (HB, BD // 16, 16))
        nan = tl.max(tl.where(blocks != blocks, 1, 0), axis=2)
        amax = tl.where(nan > 0, float("nan"), tl.max(tl.abs(blocks), axis=2))
        bs = (amax * (1.0 / 6.0)) * g
        bs = tl.where(bs > 448.0, 448.0, bs)
        sb = bs.to(tl.float8e4nv)
        unit = sb.to(tl.float32) * g_rcp
        positive = unit > 0.0   # a zero (or NaN) scale gives code 0 for its block
        unit = tl.where(unit < 1e-30, 1e-30, unit)
        q = tl.where(positive[:, :, None], tl.math.div_rn(blocks, unit[:, :, None]), 0.0)
        pairs = tl.reshape(_e2m1_codes(q), (HB, BD // 2, 2))
        lo, hi = tl.split(pairs)
        byte = (lo | (hi << 4)).to(tl.uint8)
        h = tl.arange(0, BD // 2)
        tl.store(q_ptr + rows[:, None] * sq + h[None, :], byte, mask=rm[:, None] & (h < D // 2)[None, :])
        j = tl.arange(0, BD // 16)
        rr = (r & 127)[:, None]
        at = (((r >> 7)[:, None] * ct + (j >> 2)[None, :]).to(tl.int64) * 512 + (rr & 31) * 16 + (rr >> 5) * 4
              + (j & 3)[None, :])
        tl.store(s_ptr + at, sb.to(tl.uint8, bitcast=True), mask=rm[:, None] & (j < D // 16)[None, :])
