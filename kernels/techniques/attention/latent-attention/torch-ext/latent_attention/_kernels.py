"""Triton kernels of latent attention over a paged cache of compressed rows (one row a position: the latent, rank R,
then the rotary key, width P), every query head of a token over the same rows, the latent its value too.

The window kernel reads a tile of query rows (a row's columns times its heads, BLOCK_M of them) over pieces b,
b + n_parts, ... of PIECE positions each, up to the tile's last column; each piece's latents are loaded once and
serve the scores and the values. Its state (running maximum, sum, unnormalized latent output) goes into the partials,
which the merge folds in partition order (the kernel core's attention_softmax). The row sums run on the tensor cores
(a dot with ones): a fixed order whatever the tile, so a token's bits depend only on its own query and positions, the
piece and the partition count, not on the rows or columns beside it."""
import triton
import triton.language as tl

from crit_kcore import attention_softmax as asm
from crit_kcore import formats


@triton.jit
def _window_kernel(q_ptr, c_ptr, cs_ptr, tab_ptr, trow_ptr, start_ptr, count_ptr, first_ptr, po_ptr, pm_ptr, pl_ptr,
                   stride_qt, stride_qh, tab_stride, tab_width, n_parts, sm_scale, count_all,
                   H: tl.constexpr, R: tl.constexpr, P: tl.constexpr, COLS: tl.constexpr, BLOCK_M: tl.constexpr,
                   PIECE: tl.constexpr, PAGE: tl.constexpr, COUNTS: tl.constexpr, VSPLIT: tl.constexpr,
                   FP8: tl.constexpr, HW_FP8: tl.constexpr, SCALES: tl.constexpr):
    """Program (query tile, partition b, row): tile rows j -> column j // H, head j % H; the partition's pieces over
    the row's positions up to each column's own, the state into partial row (r * COLS + column, head, b). VSPLIT
    programs share a tile, each computing its scores and R / VSPLIT of the latent output columns (the same bits). FP8:
    the cache holds E4M3 codes (as bytes) with SCALES FP32 scales a position (cs_ptr: 2, the latent's and the rotary
    key's; 1, one for the whole row, a KV pool's fp8:row pages),
    converted exactly to the queries' dtype for the dots; each part's dot takes its scale per position, the values
    the latent's (folded into the probabilities, as paged-kv-attention's fp8:row pages)."""
    mt = tl.program_id(0) // VSPLIT
    vs = tl.program_id(0) % VSPLIT
    DVP: tl.constexpr = R // VSPLIT
    b = tl.program_id(1)
    r = tl.program_id(2)
    if COUNTS:
        count = tl.minimum(tl.load(count_ptr + r), COLS)
    else:
        count = count_all
    j0 = mt * BLOCK_M
    first = tl.load(first_ptr + r).to(tl.int64)
    c_hi = tl.minimum((j0 + BLOCK_M - 1) // H, count - 1)
    n_keys = first + c_hi + 1
    n_pieces = tl.cdiv(n_keys, PIECE).to(tl.int32)
    if (j0 // H < count) & (b < n_pieces):
        start = tl.load(start_ptr + r).to(tl.int64)
        trow = tl.load(trow_ptr + r).to(tl.int64)
        j = j0 + tl.arange(0, BLOCK_M)
        c = (j // H).to(tl.int64)
        h = j % H
        live = c < count
        rr = tl.arange(0, R)
        pp = tl.arange(0, P)
        qrow = q_ptr + (start + c) * stride_qt + h * stride_qh
        ql = tl.load(qrow[:, None] + rr[None, :], mask=live[:, None], other=0.0)
        qr = tl.load(qrow[:, None] + R + pp[None, :], mask=live[:, None], other=0.0)
        lim = first + c
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, DVP], tl.float32)
        vv = vs * DVP + tl.arange(0, DVP)
        ones = tl.full([PIECE, 16], 1.0, q_ptr.dtype.element_ty)
        for i in range(b, n_pieces, n_parts):
            n = i * PIECE + tl.arange(0, PIECE)
            nm = n < n_keys
            pg = tl.load(tab_ptr + trow * tab_stride + tl.minimum(n // PAGE, tab_width - 1), mask=nm, other=0)
            slot = pg.to(tl.int64) * PAGE + n % PAGE
            row = c_ptr + slot * (R + P)
            kl = tl.load(row[:, None] + rr[None, :], mask=nm[:, None], other=0)
            kr = tl.load(row[:, None] + R + pp[None, :], mask=nm[:, None], other=0)
            if FP8:
                kl = formats.e4m3_to(kl, ql.dtype, HW_FP8)
                kr = formats.e4m3_to(kr, ql.dtype, HW_FP8)
                sl = tl.load(cs_ptr + slot * SCALES, mask=nm, other=0.0)
                sr = tl.load(cs_ptr + slot * SCALES + (SCALES - 1), mask=nm, other=0.0)
                s = (tl.dot(ql, tl.trans(kl)) * sl[None, :] + tl.dot(qr, tl.trans(kr)) * sr[None, :]) * sm_scale
            else:
                s = tl.dot(qr, tl.trans(kr), tl.dot(ql, tl.trans(kl))) * sm_scale
            s = tl.where(nm[None, :] & (n[None, :] <= lim[:, None]), s, float("-inf"))
            p, alpha, m_new = asm.step(s, m_i, False)
            if VSPLIT == 1:
                kv = kl
            else:
                kv = tl.load(row[:, None] + vv[None, :], mask=nm[:, None], other=0)
                if FP8:
                    kv = formats.e4m3_to(kv, ql.dtype, HW_FP8)
            if FP8:
                pv = (p * sl[None, :]).to(ql.dtype)
            else:
                pv = p.to(ql.dtype)
            acc = acc * alpha[:, None] + tl.dot(pv, kv)
            l_i = l_i * alpha + asm.row_sum(p, ones, True)
            m_i = m_new
        idx = ((r.to(tl.int64) * COLS + c) * H + h) * n_parts + b
        tl.store(po_ptr + idx[:, None] * R + vv[None, :], acc, mask=live[:, None])
        tl.store(pm_ptr + idx, m_i, mask=live & (vs == 0))
        tl.store(pl_ptr + idx, l_i, mask=live & (vs == 0))


@triton.jit(do_not_specialize=["pos0", "T"])
def _prompt_kernel(q_ptr, c_ptr, cs_ptr, tab_ptr, pos_ptr, out_ptr, lse_ptr, sink_ptr, T, pos0, stride_qt, stride_qh,
                   stride_ot, stride_oh, sm_scale, H: tl.constexpr, R: tl.constexpr, P: tl.constexpr,
                   BLOCK_M: tl.constexpr, PIECE: tl.constexpr, PAGE: tl.constexpr, VSPLIT: tl.constexpr,
                   FP8: tl.constexpr, HW_FP8: tl.constexpr, SCALES: tl.constexpr, POS_T: tl.constexpr,
                   LSE: tl.constexpr, SINKS: tl.constexpr):
    """Program (query tile, value part): a prompt chunk of one sequence, T tokens at positions pos0 .. (or with POS_T
    pos_ptr[0], read on the device) over its page table, causal; tile rows j -> token j // H, head j % H (the chunk's
    tokens times their heads packed into the tile, each piece of cached rows loaded once for all of them); the pieces
    of PIECE positions in order up to the tile's last token in one online softmax, the head's sink last, the output
    normalized and written (no partitions, no partials). A token's bits do not depend on the chunk's length."""
    mt = tl.program_id(0) // VSPLIT
    vs = tl.program_id(0) % VSPLIT
    DVP: tl.constexpr = R // VSPLIT
    if POS_T:
        pos0 = tl.load(pos_ptr).to(tl.int32)
    j = mt * BLOCK_M + tl.arange(0, BLOCK_M)
    t = j // H
    h = j % H
    live = t < T
    t_hi = tl.minimum((mt * BLOCK_M + BLOCK_M - 1) // H, T - 1)
    n_keys = pos0 + t_hi + 1
    rr = tl.arange(0, R)
    pp = tl.arange(0, P)
    vv = vs * DVP + tl.arange(0, DVP)
    qrow = q_ptr + t.to(tl.int64) * stride_qt + h * stride_qh
    ql = tl.load(qrow[:, None] + rr[None, :], mask=live[:, None], other=0.0)
    qr = tl.load(qrow[:, None] + R + pp[None, :], mask=live[:, None], other=0.0)
    lim = pos0 + t
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, DVP], tl.float32)
    ones = tl.full([PIECE, 16], 1.0, q_ptr.dtype.element_ty)
    for n0 in range(0, n_keys, PIECE):
        n = n0 + tl.arange(0, PIECE)
        nm = n < n_keys
        pg = tl.load(tab_ptr + n // PAGE, mask=nm, other=0)
        slot = pg.to(tl.int64) * PAGE + n % PAGE
        row = c_ptr + slot * (R + P)
        kl = tl.load(row[:, None] + rr[None, :], mask=nm[:, None], other=0)
        kr = tl.load(row[:, None] + R + pp[None, :], mask=nm[:, None], other=0)
        if FP8:
            kl = formats.e4m3_to(kl, ql.dtype, HW_FP8)
            kr = formats.e4m3_to(kr, ql.dtype, HW_FP8)
            sl = tl.load(cs_ptr + slot * SCALES, mask=nm, other=0.0)
            sr = tl.load(cs_ptr + slot * SCALES + (SCALES - 1), mask=nm, other=0.0)
            s = (tl.dot(ql, tl.trans(kl)) * sl[None, :] + tl.dot(qr, tl.trans(kr)) * sr[None, :]) * sm_scale
        else:
            s = tl.dot(qr, tl.trans(kr), tl.dot(ql, tl.trans(kl))) * sm_scale
        s = tl.where(nm[None, :] & (n[None, :] <= lim[:, None]), s, float("-inf"))
        p, alpha, m_new = asm.step(s, m_i, False)
        if VSPLIT == 1:
            kv = kl
        else:
            kv = tl.load(row[:, None] + vv[None, :], mask=nm[:, None], other=0)
            if FP8:
                kv = formats.e4m3_to(kv, ql.dtype, HW_FP8)
        if FP8:
            pv = (p * sl[None, :]).to(ql.dtype)
        else:
            pv = p.to(ql.dtype)
        acc = acc * alpha[:, None] + tl.dot(pv, kv)
        l_i = l_i * alpha + asm.row_sum(p, ones, True)
        m_i = m_new
    if SINKS:
        m_i, l_i, keep = asm.sink(m_i, l_i, tl.load(sink_ptr + h, mask=live, other=float("-inf")), False)
        acc = acc * keep[:, None]
    ok = l_i != 0.0   # a token with no position is 0; a NaN sum (a NaN query) stays NaN
    o = tl.where(ok[:, None], acc / tl.where(ok, l_i, 1.0)[:, None], 0.0)
    orow = out_ptr + t.to(tl.int64) * stride_ot + h * stride_oh
    tl.store(orow[:, None] + vv[None, :], o.to(out_ptr.dtype.element_ty), mask=live[:, None])
    if LSE:
        tl.store(lse_ptr + t.to(tl.int64) * H + h, asm.lse(m_i, l_i, False), mask=live & (vs == 0))


@triton.jit
def _merge_kernel(po_ptr, pm_ptr, pl_ptr, out_ptr, lse_ptr, sink_ptr, start_ptr, count_ptr, first_ptr, n_parts,
                  stride_ot, stride_oh, count_all, H: tl.constexpr, R: tl.constexpr, COLS: tl.constexpr,
                  PIECE: tl.constexpr, COUNTS: tl.constexpr, LSE: tl.constexpr, SINKS: tl.constexpr,
                  CHUNK_P: tl.constexpr):
    """Program (column, head, row): the partials of the partitions holding the column's positions folded in partition
    order, the head's sink (SINKS) last, into the output row in the latent space (and its log-sum-exp)."""
    c = tl.program_id(0)
    h = tl.program_id(1)
    r = tl.program_id(2)
    if COUNTS:
        count = tl.minimum(tl.load(count_ptr + r), COLS)
    else:
        count = count_all
    if c < count:
        start = tl.load(start_ptr + r).to(tl.int64)
        first = tl.load(first_ptr + r).to(tl.int64)
        nb = tl.minimum(n_parts, tl.cdiv(first + c + 1, PIECE)).to(tl.int32)
        base = ((r.to(tl.int64) * COLS + c) * H + h) * n_parts
        rr = tl.arange(0, R)
        m_run = tl.max(tl.full([CHUNK_P], float("-inf"), tl.float32), axis=0)
        l_run = tl.sum(tl.zeros([CHUNK_P], tl.float32), axis=0)
        acc = tl.zeros([R], tl.float32)
        for b0 in range(0, nb, CHUNK_P):
            for k in tl.static_range(CHUNK_P):
                use = b0 + k < nb
                at = base + b0 + k
                m = tl.load(pm_ptr + at, mask=use, other=float("-inf"))
                l = tl.load(pl_ptr + at, mask=use, other=0.0)
                o = tl.load(po_ptr + at * R + rr, mask=use, other=0.0)
                m_new, l_new, keep, take = asm.fold(m_run, l_run, m, l, False)
                m_run = tl.where(use, m_new, m_run)
                l_run = tl.where(use, l_new, l_run)
                acc = tl.where(use, tl.fma(acc, keep, o * take), acc)
        if SINKS:
            m_run, l_run, keep = asm.sink(m_run, l_run, tl.load(sink_ptr + h), False)
            acc = acc * keep
        ok = l_run != 0.0   # a row with no position is 0; a NaN sum (a NaN query) stays NaN
        o = tl.where(ok, acc / tl.where(ok, l_run, 1.0), 0.0)
        tl.store(out_ptr + (start + c) * stride_ot + h * stride_oh + rr, o.to(out_ptr.dtype.element_ty))
        if LSE:
            tl.store(lse_ptr + (start + c) * H + h, asm.lse(m_run, l_run, False))
