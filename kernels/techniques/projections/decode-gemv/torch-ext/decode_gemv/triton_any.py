"""decode-gemv's format-generic Triton GEMV (kernel.json's structure gemv-triton): 1 to 16 rows of 16-bit activations
times a matrix in any weight format gemv-cuda takes (the key's weights: 0 NVFP4, 1 FP8 row scales, 2 MXFP8, 3 BF16,
4 INT4 with a zero point a group), the format a decoder over the kernel core's (crit_kcore.formats), the multiply one
of two paths (MUL, an order knob):

- fma (1): a program takes BLOCK_N matrix rows and BLOCK_K values a step; the decoder gives each 16-value group's
  even and odd elements' exact FP32 values and the factor its sum takes (NVFP4's block scale, MXFP8's power of two,
  INT4's scale); one FP32 sum per row and group carried over K, the groups added once at the end, the row factor after.
  Eight rows a program; more run as further programs over the next rows, each row's sum its own.
- dot (2): a program multiplies a [16 rows x BK] block of the activations by a [BN x BK] block of the weights decoded
  exactly to the dot's 16-bit type (BF16, or FP16 for FP16 activations where the values are exact in it; INT4 with a
  zero point as two dots, the value's BF16 rounding and its rest) with tl.dot, FP32 sums over its part of K; SPLIT parts
  write FP32 partial sums a second kernel adds in part order, then the row factor.

The program's shape never depends on the call's row count, so a row's bits are the same alone and beside up to 15
others. Every knob but MUL, BN (the dot's), BK and SPLIT only picks which rows a program takes and how it runs (the fma
path's BLOCK_N included: tests check its tiles' bits); a tuning keeps one value of each order knob a matrix.

The older per-format Triton GEMVs are this kernel: gemv_triton (structure gemv-nvfp4-triton) is the fma path on NVFP4,
gemv_fp8_dot (gemv-fp8-dot) the dot path on FP8 (BF16 operands for either activation type, as it had), the same
operations and so the same bits as before. A tuning made before gemv-triton still applies: for NVFP4 and FP8 an alias's
table wins where it names a configuration other than its defaults, where a sweep selected one, or where the tuning
tunes the alias and not gemv-triton.

Untuned (every knob 0) the kernel picks from the format, the shape and the device: the dot path where K is a multiple
of 128 and the format's values are exact in the dot's type (MXFP8, whose power of two can leave BF16's range, and BF16
weights against FP16 activations take fma), its tile by the shape and the multiprocessor count (dot_rule, and
dot_rule_nibbles for two codes a byte); the fma path's tile by
Triton's autotuner on the device, BLOCK_K 512.
"""
import torch
import triton
import triton.language as tl

from crit_kcore import formats

from . import _crit_tuning

MAX_ROWS = 16
FMA_ROWS = 8         # rows an fma program takes
MUL_RULE, MUL_FMA, MUL_DOT = 0, 1, 2
WEIGHTS = {"nvfp4": formats.NVFP4, "fp8": formats.FP8, "mxfp8": formats.MXFP8, "bf16": formats.BF16,
           "int4_zp": formats.INT4_ZP}
BLOCK_K = 512        # the fma path's values a step where nothing names another
# Tiles (rows a program, warps) the fma path's autotuner measures once per matrix shape: never per row count, and they
# choose which rows a program computes, never a row's order.
GEMV_TILES = [triton.Config({"BLOCK_N": n, "BLOCK_K": BLOCK_K}, num_warps=w)
              for n, w in ((2, 2), (4, 2), (4, 4), (8, 4), (8, 8), (16, 4), (16, 8), (32, 8))]

_HW = {}


def hardware(device):
    """Whether the fma path decodes E2M1 by the hardware conversion on `device` (compute capability 10 and later;
    Triton builds for its architecture-specific target there). Tests set _HW[device] to compare both decodes."""
    device = torch.device(device)
    if device not in _HW:
        _HW[device] = torch.cuda.get_device_capability(device)[0] >= 10
    return _HW[device]


_FP8 = {}


def hardware_fp8(device):
    """Whether `device` converts E4M3 in hardware (compute capability 8.9 on); elsewhere the integer decode."""
    device = torch.device(device)
    if device not in _FP8:
        _FP8[device] = torch.cuda.get_device_capability(device) >= (8, 9)
    return _FP8[device]


_SMS = {}


def multiprocessors(device):
    device = torch.device(device)
    if device not in _SMS:
        _SMS[device] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SMS[device]


# ---- the fma path ----

@triton.jit
def _part(lo, hi, x_ptr, k0, j2, kmask, GROUPS: tl.constexpr):
    xx = tl.load(x_ptr + k0 + j2, mask=kmask, other=0.0).to(tl.float32)
    xe, xo = tl.split(tl.reshape(xx, (GROUPS, 8, 2)))
    return tl.sum(lo * xe[None, :, :] + hi * xo[None, :, :], axis=2)


@triton.jit
def _pairs(WF: tl.constexpr, w_ptr, s_ptr, z_ptr, rows, rmask, k0, K, GROUP, BLOCK_N: tl.constexpr,
           BLOCK_K: tl.constexpr, HW: tl.constexpr, HW_FP8: tl.constexpr):
    """The decoder of the fma path: (even, odd, factor) for a step of BLOCK_K values of the rows: the exact FP32 values
    of elements 2i and 2i + 1 of each 16-value group [BLOCK_N, GROUPS, 8], and the factor each group's sum takes
    [BLOCK_N, GROUPS] (NVFP4: the E4M3 block scale; MXFP8: the E8M0 power of two; INT4: the FP16 scale; FP8 and BF16
    have none: zeros, never multiplied, _add)."""
    HALF: tl.constexpr = BLOCK_K // 2
    GROUPS: tl.constexpr = BLOCK_K // 16
    g = tl.arange(0, GROUPS)
    gmask = rmask[:, None] & (k0 + 16 * g < K)[None, :]
    if WF == 0 or WF == 4:   # 4-bit codes, element 2i in the low nibble of byte i
        j = tl.arange(0, HALF)
        packed = tl.load(w_ptr + rows[:, None] * (K // 2) + (k0 // 2 + j)[None, :],
                         mask=rmask[:, None] & (k0 + 2 * j < K)[None, :], other=0)
        if WF == 0:
            factor = formats.e4m3_to(tl.load(s_ptr + rows[:, None] * (K // 16) + (k0 // 16 + g)[None, :], mask=gmask,
                                             other=0), tl.float32, HW_FP8)
            if HW:
                lo16, hi16 = formats.e2m1_pair_cvt(packed)
                even = tl.reshape(lo16.to(tl.float32), (BLOCK_N, GROUPS, 8))
                odd = tl.reshape(hi16.to(tl.float32), (BLOCK_N, GROUPS, 8))
            else:
                even = tl.reshape(formats.e2m1(packed), (BLOCK_N, GROUPS, 8))
                odd = tl.reshape(formats.e2m1(packed >> 4), (BLOCK_N, GROUPS, 8))
        else:
            at = rows[:, None] * (K // GROUP) + ((k0 + 16 * g) // GROUP)[None, :]
            factor = formats.f16_values(tl.load(s_ptr + at, mask=gmask, other=0))
            zero = tl.load(z_ptr + at, mask=gmask, other=0)[:, :, None]
            even = formats.int4_zp_offsets(tl.reshape(packed, (BLOCK_N, GROUPS, 8)), zero)
            odd = formats.int4_zp_offsets(tl.reshape(packed >> 4, (BLOCK_N, GROUPS, 8)), zero)
    else:   # one value a byte or a 16-bit word
        j2 = tl.arange(0, BLOCK_K)
        raw = tl.load(w_ptr + rows[:, None] * K + (k0 + j2)[None, :], mask=rmask[:, None] & (k0 + j2 < K)[None, :],
                      other=0)
        if WF == 3:
            v = formats.bf16_values(raw)
        else:
            v = formats.e4m3_to(raw, tl.float32, HW_FP8)
        even, odd = tl.split(tl.reshape(v, (BLOCK_N, GROUPS, 8, 2)))
        if WF == 2:
            factor = formats.e8m0_values(tl.load(s_ptr + rows[:, None] * (K // 32) + ((k0 + 16 * g) // 32)[None, :],
                                                 mask=gmask, other=0))
        else:
            factor = tl.zeros([BLOCK_N, GROUPS], tl.float32)
    return even, odd, factor


@triton.jit
def _add(acc, lo, hi, factor, x_ptr, k0, j2, kmask, GROUPS: tl.constexpr, WF: tl.constexpr):
    if WF == 0 or WF == 2 or WF == 4:
        acc += _part(lo, hi, x_ptr, k0, j2, kmask, GROUPS) * factor
    else:
        acc += _part(lo, hi, x_ptr, k0, j2, kmask, GROUPS)
    return acc


@triton.jit
def _store(out_ptr, rows, rmask, acc, f, HAS_F: tl.constexpr):
    if HAS_F:
        tl.store(out_ptr + rows, (tl.sum(acc, axis=1) * f).to(out_ptr.dtype.element_ty), mask=rmask)
    else:
        tl.store(out_ptr + rows, tl.sum(acc, axis=1).to(out_ptr.dtype.element_ty), mask=rmask)


@triton.jit
def _fma_body(x_ptr, w_ptr, s_ptr, z_ptr, f_ptr, out_ptr, N, K, GROUP, stride_xm, stride_om, M: tl.constexpr,
              BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, WF: tl.constexpr, HW: tl.constexpr,
              HW_FP8: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    GROUPS: tl.constexpr = BLOCK_K // 16
    j2 = tl.arange(0, BLOCK_K)
    # One FP32 sum per row and 16-value group of the step, carried over the K loop: the groups' order across lanes is
    # reduced once, after it.
    acc0 = tl.zeros([BLOCK_N, GROUPS], tl.float32)
    acc1 = tl.zeros([BLOCK_N, GROUPS], tl.float32)
    acc2 = tl.zeros([BLOCK_N, GROUPS], tl.float32)
    acc3 = tl.zeros([BLOCK_N, GROUPS], tl.float32)
    acc4 = tl.zeros([BLOCK_N, GROUPS], tl.float32)
    acc5 = tl.zeros([BLOCK_N, GROUPS], tl.float32)
    acc6 = tl.zeros([BLOCK_N, GROUPS], tl.float32)
    acc7 = tl.zeros([BLOCK_N, GROUPS], tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kmask2 = k0 + j2 < K
        lo, hi, factor = _pairs(WF, w_ptr, s_ptr, z_ptr, rows, rmask, k0, K, GROUP, BLOCK_N, BLOCK_K, HW, HW_FP8)
        acc0 = _add(acc0, lo, hi, factor, x_ptr, k0, j2, kmask2, GROUPS, WF)
        if M > 1:
            acc1 = _add(acc1, lo, hi, factor, x_ptr + stride_xm, k0, j2, kmask2, GROUPS, WF)
        if M > 2:
            acc2 = _add(acc2, lo, hi, factor, x_ptr + 2 * stride_xm, k0, j2, kmask2, GROUPS, WF)
        if M > 3:
            acc3 = _add(acc3, lo, hi, factor, x_ptr + 3 * stride_xm, k0, j2, kmask2, GROUPS, WF)
        if M > 4:
            acc4 = _add(acc4, lo, hi, factor, x_ptr + 4 * stride_xm, k0, j2, kmask2, GROUPS, WF)
        if M > 5:
            acc5 = _add(acc5, lo, hi, factor, x_ptr + 5 * stride_xm, k0, j2, kmask2, GROUPS, WF)
        if M > 6:
            acc6 = _add(acc6, lo, hi, factor, x_ptr + 6 * stride_xm, k0, j2, kmask2, GROUPS, WF)
        if M > 7:
            acc7 = _add(acc7, lo, hi, factor, x_ptr + 7 * stride_xm, k0, j2, kmask2, GROUPS, WF)
    # The row factor after the sum: NVFP4's inverse global scale (times 2^14 on the bit-pattern decode), FP8's scale.
    HAS_F: tl.constexpr = WF == 0 or WF == 1
    if WF == 0:
        f = tl.load(f_ptr + rows, mask=rmask, other=0.0) * (1.0 if HW else 16384.0)
    elif WF == 1:
        f = tl.load(f_ptr + rows, mask=rmask, other=0.0)
    else:
        f = 0.0
    _store(out_ptr, rows, rmask, acc0, f, HAS_F)
    if M > 1:
        _store(out_ptr + stride_om, rows, rmask, acc1, f, HAS_F)
    if M > 2:
        _store(out_ptr + 2 * stride_om, rows, rmask, acc2, f, HAS_F)
    if M > 3:
        _store(out_ptr + 3 * stride_om, rows, rmask, acc3, f, HAS_F)
    if M > 4:
        _store(out_ptr + 4 * stride_om, rows, rmask, acc4, f, HAS_F)
    if M > 5:
        _store(out_ptr + 5 * stride_om, rows, rmask, acc5, f, HAS_F)
    if M > 6:
        _store(out_ptr + 6 * stride_om, rows, rmask, acc6, f, HAS_F)
    if M > 7:
        _store(out_ptr + 7 * stride_om, rows, rmask, acc7, f, HAS_F)


@triton.autotune(configs=GEMV_TILES, key=["N", "K", "HW", "WF"])
@triton.jit
def _fma_kernel(x_ptr, w_ptr, s_ptr, z_ptr, f_ptr, out_ptr, N, K, GROUP, stride_xm, stride_om, M: tl.constexpr,
                BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, WF: tl.constexpr, HW: tl.constexpr,
                HW_FP8: tl.constexpr):
    _fma_body(x_ptr, w_ptr, s_ptr, z_ptr, f_ptr, out_ptr, N, K, GROUP, stride_xm, stride_om, M, BLOCK_N, BLOCK_K, WF,
              HW, HW_FP8)


def fma(weights, x, w, scales, zeros, factor, out, group, block_n, block_k, warps, stages):
    """The fma path on x [M][K] (M <= 16; rows past 8 as further launches over the next rows): block_n 0 takes the
    autotuner's tile (block_k then 512)."""
    M, K = x.shape
    N = out.shape[1]
    dummy = w
    hw, hw8 = hardware(x.device), hardware_fp8(x.device)
    for r0 in range(0, M, FMA_ROWS):
        m = min(FMA_ROWS, M - r0)
        xs, os = x[r0:r0 + m], out[r0:r0 + m]
        args = (xs, w, dummy if scales is None else scales, dummy if zeros is None else zeros,
                dummy if factor is None else factor, os, N, K, max(int(group), 1), xs.stride(0), os.stride(0))
        if block_n == 0:
            _fma_kernel[lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)](*args, M=m, WF=weights, HW=hw, HW_FP8=hw8)
        else:
            _fma_kernel.fn[(triton.cdiv(N, block_n),)](*args, M=m, BLOCK_N=block_n, BLOCK_K=block_k, WF=weights,
                                                       HW=hw, HW_FP8=hw8, num_warps=warps, num_stages=stages)
    return out


# ---- the dot path ----

@triton.jit
def _dot_tile(WF: tl.constexpr, w_ptr, s_ptr, nn, k0, K, BN: tl.constexpr, BK: tl.constexpr, DOT: tl.constexpr,
              HW_FP8: tl.constexpr):
    """The dot path's decoder for one value a byte or a word (FP8, MXFP8, BF16): the weights [BN][BK] of rows nn from
    column k0 in K order, exactly in DOT."""
    k = k0 + tl.arange(0, BK)
    base = nn[:, None].to(tl.int64)
    if WF == 1:
        raw = tl.load(w_ptr + base * K + k[None, :])
        if HW_FP8:
            w = raw.to(tl.float8e4nv, bitcast=True).to(DOT)
        else:
            w = formats.e4m3_values(raw).to(DOT)
    elif WF == 2:
        v = formats.e4m3_to(tl.load(w_ptr + base * K + k[None, :]), tl.float32, HW_FP8)
        e = formats.e8m0_values(tl.load(s_ptr + base * (K // 32) + (k0 // 32 + tl.arange(0, BK // 32))[None, :]))
        w = tl.reshape(tl.reshape(v, (BN, BK // 32, 32)) * e[:, :, None], (BN, BK)).to(DOT)
    else:
        w = tl.load(w_ptr + base * K + k[None, :]).to(DOT)
    return w


@triton.jit
def _dot_nibbles(WF: tl.constexpr, w_ptr, s_ptr, z_ptr, nn, k0, K, GROUP, BN: tl.constexpr, BK: tl.constexpr,
                 DOT: tl.constexpr, HW: tl.constexpr, HW_FP8: tl.constexpr):
    """The dot path's decoder for two codes a byte (NVFP4, INT4 with a zero point): the even and the odd elements of
    rows nn from column k0, [BN][BK / 2] each, exactly in DOT: (even, even rest, odd, odd rest), the rests INT4's
    (its value's DOT rounding and the remainder: hi + lo exact), NVFP4's the same tensors again (unused)."""
    j = tl.arange(0, BK // 2)
    base = nn[:, None].to(tl.int64)
    packed = tl.load(w_ptr + base * (K // 2) + (k0 // 2 + j)[None, :])
    if WF == 0:
        scale = formats.e4m3_to(tl.load(s_ptr + base * (K // 16) + (k0 // 16 + tl.arange(0, BK // 16))[None, :]),
                                tl.float32, HW_FP8)
        if HW:
            lo16, hi16 = formats.e2m1_pair_cvt(packed)
            lo = lo16.to(tl.float32)
            hi = hi16.to(tl.float32)
        else:
            lo = formats.e2m1(packed)
            hi = formats.e2m1(packed >> 4)
            scale = scale * 16384.0   # the bit-pattern decode's unit folded in, exactly
        even = tl.reshape(tl.reshape(lo, (BN, BK // 16, 8)) * scale[:, :, None], (BN, BK // 2)).to(DOT)
        odd = tl.reshape(tl.reshape(hi, (BN, BK // 16, 8)) * scale[:, :, None], (BN, BK // 2)).to(DOT)
        return even, even, odd, odd
    else:
        # A group is a multiple of 32 inputs: one scale and zero a 32-value block of the step (16 bytes).
        at = base * (K // GROUP) + ((k0 + 32 * tl.arange(0, BK // 32)) // GROUP)[None, :]
        scale = formats.f16_values(tl.load(s_ptr + at))[:, :, None]
        zero = tl.load(z_ptr + at)[:, :, None]
        ve = tl.reshape(formats.int4_zp_values(tl.reshape(packed, (BN, BK // 32, 16)), zero, scale), (BN, BK // 2))
        vo = tl.reshape(formats.int4_zp_values(tl.reshape(packed >> 4, (BN, BK // 32, 16)), zero, scale),
                        (BN, BK // 2))
        eh = ve.to(DOT)
        oh = vo.to(DOT)
        return eh, (ve - eh.to(tl.float32)).to(DOT), oh, (vo - oh.to(tl.float32)).to(DOT)


@triton.jit
def _dot_kernel(x_ptr, w_ptr, s_ptr, z_ptr, f_ptr, out_ptr, T, N, K, KS, GROUP, x_stride, out_stride,
                BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr, WF: tl.constexpr, DOT: tl.constexpr,
                HW: tl.constexpr, HW_FP8: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    rows = tl.arange(0, 16)
    n = pid_n * BN + tl.arange(0, BN)
    nmask = n < N
    nn = tl.minimum(n, N - 1)   # a row past N reads row N - 1 and writes nothing
    acc = tl.zeros([16, BN], dtype=tl.float32)
    xmask = rows[:, None] < T
    for k0 in range(pid_s * KS, pid_s * KS + KS, BK):
        k = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + rows[:, None] * x_stride + k[None, :], mask=xmask, other=0.)
        if WF == 0 or WF == 4:
            # Two codes a byte: the even and the odd inputs as two dots of BK / 2 (no interleave of the weights)
            xe, xo = tl.split(tl.reshape(x, (16, BK // 2, 2)))
            eh, el, oh, ol = _dot_nibbles(WF, w_ptr, s_ptr, z_ptr, nn, k0, K, GROUP, BN, BK, DOT, HW, HW_FP8)
            acc = tl.dot(xe.to(DOT), tl.trans(eh), acc)
            acc = tl.dot(xo.to(DOT), tl.trans(oh), acc)
            if WF == 4:
                acc = tl.dot(xe.to(DOT), tl.trans(el), acc)
                acc = tl.dot(xo.to(DOT), tl.trans(ol), acc)
        else:
            acc = tl.dot(x.to(DOT), tl.trans(_dot_tile(WF, w_ptr, s_ptr, nn, k0, K, BN, BK, DOT, HW_FP8)), acc)
    omask = xmask & nmask[None, :]
    if SPLIT == 1:
        if WF == 0 or WF == 1:
            y = acc * tl.load(f_ptr + nn)[None, :]
        else:
            y = acc
        tl.store(out_ptr + rows[:, None] * out_stride + n[None, :], y.to(out_ptr.dtype.element_ty), mask=omask)
    else:
        tl.store(out_ptr + pid_s * T * N + rows[:, None] * N + n[None, :], acc, mask=omask)


@triton.jit
def _reduce_kernel(part_ptr, f_ptr, out_ptr, T, N, out_stride, SPLIT: tl.constexpr, BN: tl.constexpr,
                   HAS_F: tl.constexpr):
    t = tl.program_id(1)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    nmask = n < N
    acc = tl.load(part_ptr + t * N + n, mask=nmask, other=0.)
    for s in tl.static_range(1, SPLIT):
        acc += tl.load(part_ptr + s * T * N + t * N + n, mask=nmask, other=0.)
    if HAS_F:
        y = acc * tl.load(f_ptr + n, mask=nmask, other=0.)
    else:
        y = acc
    tl.store(out_ptr + t * out_stride + n, y.to(out_ptr.dtype.element_ty), mask=nmask)


def partial_elements(rows: int, n: int, split: int) -> int:
    """FP32 elements the dot path's partial sums of a call take (0 for one part)."""
    return split * rows * n if split > 1 else 0


def dot(weights, x, w, scales, zeros, factor, out, group, plan, partial=None, dot_dtype=None):
    """The dot path on x [M][K] (M <= 16), plan (BN, BK, warps, split) with K a multiple of split * BK (BF16 weights:
    BK at most 16384 / BN, a stage's block within shared memory; a larger one runs that); dot_dtype the operands'
    type (None: x's). partial: FP32 with partial_elements(M, N, split) elements or more when split > 1 (None
    allocates them)."""
    bn, bk, warps, split = plan
    if weights == formats.BF16 and bn * bk > 16384:
        bk = max(128, 16384 // bn)   # a [BN x BK] BF16 block a stage within shared memory: the order of that BK
    M, K = x.shape
    N = out.shape[1]
    hw, hw8 = hardware(x.device), hardware_fp8(x.device)
    dtype = tl.float16 if (dot_dtype or x.dtype) == torch.float16 else tl.bfloat16
    s = w if scales is None else scales
    z = w if zeros is None else zeros
    f = w if factor is None else factor
    grid = (triton.cdiv(N, bn), split)
    if split == 1:
        _dot_kernel[grid](x, w, s, z, f, out, M, N, K, K, max(int(group), 1), x.stride(0), out.stride(0), BN=bn, BK=bk,
                          SPLIT=1, WF=weights, DOT=dtype, HW=hw, HW_FP8=hw8, num_warps=warps)
        return out
    need = partial_elements(M, N, split)
    if partial is None:
        partial = torch.empty(need, dtype=torch.float32, device=x.device)
    elif partial.numel() < need or partial.dtype != torch.float32:
        raise ValueError(f"{split} parts of K take an FP32 partial buffer of {need} elements or more")
    _dot_kernel[grid](x, w, s, z, f, partial, M, N, K, K // split, max(int(group), 1), x.stride(0), out.stride(0),
                      BN=bn, BK=bk, SPLIT=split, WF=weights, DOT=dtype, HW=hw, HW_FP8=hw8, num_warps=warps)
    _reduce_kernel[(triton.cdiv(N, 1024), M)](partial, f, out, M, N, out.stride(0), SPLIT=split, BN=1024,
                                              HAS_F=weights in (0, 1), num_warps=4)
    return out


def dot_rule(n: int, k: int, multiprocessors: int):
    """(BN, BK, warps, split) from the matrix's shape and the multiprocessor count alone: 64 rows and 8 warps a program
    from 16384 rows on, else 32 and 4; BK 256 (512 from 65536 rows on, 128 where 256 does not divide K); K split in two,
    four, ... while the programs are fewer than the multiprocessors and each part keeps at least 4 BK. (Measured on an
    RTX 5090 over ten FP8 matrices of three models, 640 to 248320 rows, at 1 to 16 rows: within 1-6% of the best tile
    of a sweep.)"""
    if k % 128:
        raise ValueError(f"the dot path takes K a multiple of 128, not {k}")
    bn, warps = (64, 8) if n >= 16384 else (32, 4)
    bk = 512 if n >= 65536 and k % 512 == 0 else 256 if k % 256 == 0 else 128
    split = 1
    while triton.cdiv(n, bn) * split < multiprocessors and k % (2 * split * bk) == 0 and k // (2 * split) >= 4 * bk:
        split *= 2
    return bn, bk, warps, split


def dot_rule_nibbles(n: int, k: int, multiprocessors: int, int4: bool = False):
    """(BN, BK, warps, split) for a matrix of two codes a byte (NVFP4, INT4 with a zero point), from its shape and the
    multiprocessor count: their decode, not the loads, bounds a program, so 64 rows and 4 warps a program, BK 256 (INT4,
    which decodes twice the operands: 128; 128 where 256 does not divide K), K split in two, four, ... while the
    programs are fewer than four a multiprocessor and each part keeps at least 4 BK. (Measured on an RTX 5090 over
    34816 x 5120, 5120 x 17408 and 12288 x 5120 at 1 and 16 rows: the rule's plan is the best or within 10% of it.)"""
    if k % 128:
        raise ValueError(f"the dot path takes K a multiple of 128, not {k}")
    bn, warps = 64, 4
    bk = 128 if int4 or k % 256 else 256
    split = 1
    while triton.cdiv(n, bn) * split < 4 * multiprocessors and k % (2 * split * bk) == 0 and k // (2 * split) >= 4 * bk:
        split *= 2
    return bn, bk, warps, split


# ---- the structure gemv-triton: knobs from the tuning, an alias's where it applies ----

def _lookup(structure, key, rows):
    """(configuration index, knobs) the build's tuning gives `structure` for the key and row count (_crit_tuning's
    knobs(), with the index)."""
    t = _crit_tuning.TUNING[structure]
    if structure in _crit_tuning.SELECTED:
        i = _crit_tuning.SELECTED[structure]
        return i, dict(t["configs"][i])
    fields = _crit_tuning.KEYS[structure]
    for e in t["table"]:
        k = e["key"]
        if (k is None or all(int(k[f]) == int(key[f]) for f in fields)) and e["rows"][0] <= rows <= e["rows"][1]:
            return e["config"], dict(t["configs"][e["config"]])
    return 0, dict(t["configs"][0])


def _alias_takes(alias, key, rows):
    """An alias's configuration for this call, or None: where a sweep selected one, where its table names one other
    than its defaults, or where the tuning tunes the alias and not gemv-triton (kernel.json's alias rule)."""
    t = _crit_tuning.TUNING.get(alias)
    if t is None:
        return None
    index, config = _lookup(alias, key, rows)
    if alias in _crit_tuning.SELECTED or index != 0 or (t["table"] and not _crit_tuning.TUNING["gemv-triton"]["table"]):
        return config
    return None


def rule_mul(weights, k, x_dtype):
    """The multiply path an untuned build takes: dot where K is a multiple of 128 and the format's values are exact in
    the dot's type (x's: BF16 for NVFP4, FP8, BF16 and INT4's two parts; FP16 for NVFP4 and FP8), fma elsewhere
    (MXFP8 always: its power of two per 32 can leave BF16's range; a tuning may still pick the dot path for it)."""
    if k % 128:
        return MUL_FMA
    if x_dtype == torch.bfloat16:
        return MUL_DOT if weights in (formats.NVFP4, formats.FP8, formats.BF16, formats.INT4_ZP) else MUL_FMA
    return MUL_DOT if weights in (formats.NVFP4, formats.FP8) else MUL_FMA


def plan(weights, n, k, rows, x_dtype=torch.bfloat16, device=None):
    """("fma", BLOCK_N, BLOCK_K, warps, stages) or ("dot", (BN, BK, warps, split)) for a call: gemv-triton's tuning
    for (weights, N, K) and the row count, an alias's where it applies (NVFP4: gemv-nvfp4-triton, FP8:
    gemv-fp8-dot), the rule for knobs left 0. A plan the matrix cannot take (the dot path's K not a multiple of
    split * BK) falls back to the rule; K not a multiple of 128 always takes fma."""
    key = {"weights": weights, "N": n, "K": k}
    config = None
    if weights == formats.NVFP4:
        old = _alias_takes("gemv-nvfp4-triton", key, rows)
        if old is not None:
            config = {"MUL": MUL_FMA, "BN": old["BLOCK_N"], "BK": old["BLOCK_K"], "WARPS": old["WARPS"],
                      "STAGES": old["STAGES"], "SPLIT": 0}
    elif weights == formats.FP8:
        old = _alias_takes("gemv-fp8-dot", key, 1)
        if old is not None:
            config = {"MUL": MUL_DOT, "BN": old["BN"], "BK": old["BK"], "WARPS": old["WARPS"], "STAGES": 0,
                      "SPLIT": old["SPLIT"]}
    if config is None:
        config = _lookup("gemv-triton", key, rows)[1]
    mul = config["MUL"] or rule_mul(weights, k, x_dtype)
    if mul == MUL_DOT and k % 128 == 0 and not (weights == formats.BF16 and x_dtype != torch.bfloat16):
        p = (config["BN"], config["BK"], config["WARPS"], config["SPLIT"])
        if 0 in p or p[0] < 16 or k % (p[1] * p[3]):
            sms = multiprocessors(device or "cuda")
            p = (dot_rule_nibbles(n, k, sms, weights == formats.INT4_ZP)
                 if weights in (formats.NVFP4, formats.INT4_ZP) else dot_rule(n, k, sms))
        return "dot", p
    bn = config["BN"] if config["BN"] <= 32 else 0
    bk = config["BK"] if bn and config["BK"] else BLOCK_K
    return "fma", bn, bk, config["WARPS"] or 4, config["STAGES"] or 3


def _check(weights, x, codes, out, scales, factor, zeros, group):
    if weights not in WEIGHTS.values():
        raise ValueError(f"gemv_any: weights is one of {sorted(WEIGHTS.values())}, not {weights}")
    if x.dim() != 2 or not 1 <= x.shape[0] <= MAX_ROWS or x.stride(-1) != 1 or out.stride(-1) != 1:
        raise ValueError(f"gemv_any: x [1..{MAX_ROWS}][K] with unit column strides, not {tuple(x.shape)}")
    M, K = x.shape
    N = codes.shape[0]
    per = 2 if weights in (formats.NVFP4, formats.INT4_ZP) else 1
    if codes.shape[1] * per != K or tuple(out.shape) != (M, N):
        raise ValueError(f"gemv_any: codes [N][K{' / 2' if per == 2 else ''}] and out [M][N] for x {tuple(x.shape)}, "
                         f"not {tuple(codes.shape)} and {tuple(out.shape)}")
    if weights == formats.NVFP4 and (K % 16 or scales is None or factor is None):
        raise ValueError("gemv_any: NVFP4 takes K a multiple of 16, E4M3 scales [N][K / 16] and a row factor [N]")
    if weights == formats.FP8 and (K % 16 or factor is None):
        raise ValueError("gemv_any: FP8 takes K a multiple of 16 and FP32 row scales [N]")
    if weights == formats.MXFP8 and (K % 32 or scales is None):
        raise ValueError("gemv_any: MXFP8 takes K a multiple of 32 and E8M0 scales [N][K / 32]")
    if weights == formats.BF16 and (K % 8 or codes.dtype != torch.bfloat16):
        raise ValueError("gemv_any: BF16 weights are a BF16 tensor, K a multiple of 8")
    if weights == formats.INT4_ZP and (group < 32 or group % 32 or K % group or scales is None or zeros is None):
        raise ValueError("gemv_any: INT4 takes a group a multiple of 32 dividing K, FP16 scales and zero points")


def gemv_any(x, weights, codes, out, scales=None, factor=None, zeros=None, group=0, partial=None):
    """out [M][N] (any float dtype, unit column stride) = x [M][K] (M 1..16, BF16 or FP16, unit column stride) times
    the matrix in format `weights` (WEIGHTS; gemv-cuda's codes) on the Triton GEMV (structure gemv-triton):
    0 NVFP4: codes [N][K / 2] uint8, scales [N][K / 16] E4M3, factor [N] FP32 (the inverse global scale);
    1 FP8: codes [N][K] E4M3 (float8_e4m3fn or uint8), factor [N] FP32 row scales;
    2 MXFP8: codes [N][K] E4M3, scales [N][K / 32] E8M0 bytes;
    3 BF16: codes [N][K] BF16 (BF16 activations on the dot path);
    4 INT4 with a zero point: codes [N][K / 2] uint8 (repack_awq / repack_gptq), scales FP16 and zeros uint8
    [N][K / group], group a multiple of 32 dividing K.
    partial: the dot path's FP32 partial sums where it splits K (dot_partial_elements; None allocates them)."""
    _check(weights, x, codes, out, scales, factor, zeros, group)
    M, K = x.shape
    N = codes.shape[0]
    w = codes if weights == formats.BF16 else codes.view(torch.uint8)
    s = None if scales is None else scales.view(torch.int16) if weights == formats.INT4_ZP else scales.view(torch.uint8)
    chosen = plan(weights, N, K, M, x.dtype, x.device)
    if chosen[0] == "dot":
        return dot(weights, x, w, s, zeros, factor, out, group, chosen[1], partial)
    return fma(weights, x, w, s, zeros, factor, out, group, *chosen[1:])
