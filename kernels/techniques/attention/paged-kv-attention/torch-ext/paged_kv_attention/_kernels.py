"""The Triton kernels behind paged_kv_attention's functions (see __init__.py for what each computes).

Page kinds (KIND): 0 16-bit pages (BF16 or FP16), 1 FP8 E4M3 codes with one multiplier per layer for the keys and one
for the values, 2 FP8 E4M3 codes with an FP32 scale per position and KV head, 3 INT8 codes with an FP32 scale per
position and KV head. A position's slot is page * PAGE + offset, its keys at (slot * kv_heads + head) * dim.

The online softmax's steps, row sums and folds are the kernel core's (crit_kcore.attention_softmax, natural
exponents: BASE2 False), the E4M3 conversions too (crit_kcore.formats).
"""
import triton
import triton.language as tl
from triton.language.extra import libdevice

from crit_kcore import attention_softmax as asm
from crit_kcore import formats


@triton.jit
def _values(x, KIND: tl.constexpr, HW_FP8: tl.constexpr, OUT: tl.constexpr):
    """Codes or 16-bit values as loaded from a page to the dot product's dtype, exactly (E4M3 codes load as bytes)."""
    if KIND == 1 or KIND == 2:
        y = formats.e4m3_to(x, OUT, HW_FP8)
    else:
        y = x.to(OUT)
    return y


# -- writes -----------------------------------------------------------------------------------------------------------
@triton.jit
def _write_kernel(k_ptr, v_ptr, slot_ptr, kp_ptr, vp_ptr, ks_ptr, vs_ptr, stride_kt, stride_kh, stride_vt, stride_vh,
                  k_div, v_div, HKV: tl.constexpr, D: tl.constexpr, DV: tl.constexpr, BD: tl.constexpr,
                  KIND: tl.constexpr):
    """One token's KV head (axis 1: keys 0..HKV-1, values HKV..2HKV-1) into its slot; a slot below 0 is skipped."""
    t = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(slot_ptr + t).to(tl.int64)
    d = tl.arange(0, BD)
    if slot >= 0:
        if head < HKV:
            h = head
            dm = d < D
            x = tl.load(k_ptr + t * stride_kt + h * stride_kh + d, mask=dm, other=0.0).to(tl.float32)
            dst = kp_ptr + (slot * HKV + h) * D + d
            sdst = ks_ptr + slot * HKV + h
            div = k_div
        else:
            h = head - HKV
            dm = d < DV
            x = tl.load(v_ptr + t * stride_vt + h * stride_vh + d, mask=dm, other=0.0).to(tl.float32)
            dst = vp_ptr + (slot * HKV + h) * DV + d
            sdst = vs_ptr + slot * HKV + h
            div = v_div
        if KIND == 0:
            tl.store(dst, x.to(dst.dtype.element_ty), mask=dm)
        elif KIND == 1:
            q = tl.minimum(tl.maximum(tl.div_rn(x, div), -448.0), 448.0)
            tl.store(dst, formats.e4m3_codes(q), mask=dm)
        elif KIND == 2:
            scale = tl.maximum(tl.div_rn(tl.max(tl.abs(x), axis=0), 448.0), 1e-30)
            q = tl.minimum(tl.maximum(tl.div_rn(x, scale), -448.0), 448.0)
            tl.store(dst, formats.e4m3_codes(q), mask=dm)
            tl.store(sdst, scale)
        else:
            scale = tl.maximum(tl.div_rn(tl.max(tl.abs(x), axis=0), 127.0), 1e-30)
            code = tl.minimum(tl.maximum(libdevice.rint(tl.div_rn(x, scale)), -127.0), 127.0)
            tl.store(dst, code.to(tl.int8), mask=dm)
            tl.store(sdst, scale)


# -- decode and verify windows, split over KV partitions -------------------------------------------------------------
@triton.jit
def _ancestors(par_ptr, start, c, live, COLS: tl.constexpr):
    """Each column's ancestors and itself within its row's window, as a 64-bit mask (bit j: column j)."""
    one = tl.full(c.shape, 1, tl.int64)
    anc = tl.where(live, one << c, 0)
    par = tl.load(par_ptr + start + c, mask=live, other=-1).to(tl.int64)
    for _ in range(COLS):
        has = par >= 0
        at = tl.where(has, par, 0)
        anc = anc | tl.where(has, one << at, 0)
        par = tl.where(has, tl.load(par_ptr + start + at, mask=has, other=-1).to(tl.int64), -1)
    return anc


@triton.jit
def _piece_scores(qrow, live, krow, nm, BLOCK_M: tl.constexpr, PIECE: tl.constexpr, D: tl.constexpr,
                  DC: tl.constexpr, KIND: tl.constexpr, HW_FP8: tl.constexpr):
    """The query rows' dot products with a piece's keys, the head dimension in chunks of DC (the chained dots keep a
    full dot's k-step order, so the bits are those of one dot over D; the chunks keep the tiles out of the spills)."""
    dc = tl.arange(0, DC)
    s = tl.zeros([BLOCK_M, PIECE], tl.float32)
    for ci in tl.static_range(D // DC):
        qc = tl.load(qrow[:, None] + (ci * DC + dc)[None, :], mask=live[:, None], other=0.0)
        kc = tl.load(krow[:, None] + (ci * DC + dc)[None, :], mask=nm[:, None], other=0)
        s = tl.dot(qc, tl.trans(_values(kc, KIND, HW_FP8, qc.dtype)), s)
    return s


@triton.jit
def _piece_values(acc0, acc1, acc2, acc3, alpha, pv, vrow, nm, DV: tl.constexpr, DC: tl.constexpr,
                  KIND: tl.constexpr, HW_FP8: tl.constexpr):
    """The value accumulators (up to four chunks of DC of the value dimension) rescaled by alpha and given a piece's
    probabilities times its values."""
    dc = tl.arange(0, DC)
    v0 = tl.load(vrow[:, None] + dc[None, :], mask=nm[:, None], other=0)
    acc0 = acc0 * alpha[:, None] + tl.dot(pv, _values(v0, KIND, HW_FP8, pv.dtype))
    if DV // DC > 1:
        v1 = tl.load(vrow[:, None] + (DC + dc)[None, :], mask=nm[:, None], other=0)
        acc1 = acc1 * alpha[:, None] + tl.dot(pv, _values(v1, KIND, HW_FP8, pv.dtype))
    if DV // DC > 2:
        v2 = tl.load(vrow[:, None] + (2 * DC + dc)[None, :], mask=nm[:, None], other=0)
        acc2 = acc2 * alpha[:, None] + tl.dot(pv, _values(v2, KIND, HW_FP8, pv.dtype))
    if DV // DC > 3:
        v3 = tl.load(vrow[:, None] + (3 * DC + dc)[None, :], mask=nm[:, None], other=0)
        acc3 = acc3 * alpha[:, None] + tl.dot(pv, _values(v3, KIND, HW_FP8, pv.dtype))
    return acc0, acc1, acc2, acc3


@triton.jit
def _partial_row(off_ptr, r, COLS: tl.constexpr, OFFSETS: tl.constexpr):
    """Row r's first partial row in its launch's partials: with OFFSETS its offset past the launch's first row's (the
    rows' tokens packed one after another), else r * COLS."""
    if OFFSETS:
        return (tl.load(off_ptr + r) - tl.load(off_ptr)).to(tl.int64)
    return r.to(tl.int64) * COLS


@triton.jit
def _store_partials(po_ptr, idx, live, acc0, acc1, acc2, acc3, DV: tl.constexpr, DC: tl.constexpr):
    dc = tl.arange(0, DC)
    row = po_ptr + idx[:, None] * DV
    tl.store(row + dc[None, :], acc0, mask=live[:, None])
    if DV // DC > 1:
        tl.store(row + (DC + dc)[None, :], acc1, mask=live[:, None])
    if DV // DC > 2:
        tl.store(row + (2 * DC + dc)[None, :], acc2, mask=live[:, None])
    if DV // DC > 3:
        tl.store(row + (3 * DC + dc)[None, :], acc3, mask=live[:, None])


@triton.jit
def _piece_rows(rows, frows, onp, isf, cols, KIND: tl.constexpr, HW_FP8: tl.constexpr, OUT: tl.constexpr):
    """Columns `cols` of a piece's key or value rows in the dot product's dtype: from the pages where onp (codes or
    16-bit values, exactly), from the fresh 16-bit rows where isf."""
    pg = tl.load(rows[:, None] + cols[None, :], mask=onp[:, None], other=0)
    fr = tl.load(frows[:, None] + cols[None, :], mask=isf[:, None], other=0.0)
    return tl.where(isf[:, None], fr.to(OUT), _values(pg, KIND, HW_FP8, OUT))


@triton.jit
def _fresh_scores(qrow, live, krow, frow, onp, isf, BLOCK_M: tl.constexpr, PIECE: tl.constexpr, D: tl.constexpr,
                  DC: tl.constexpr, KIND: tl.constexpr, HW_FP8: tl.constexpr):
    """_piece_scores with some of the piece's keys from fresh rows: a column's dot products are its own key's, so a
    paged position's score has the bits it has without fresh rows."""
    dc = tl.arange(0, DC)
    s = tl.zeros([BLOCK_M, PIECE], tl.float32)
    for ci in tl.static_range(D // DC):
        qc = tl.load(qrow[:, None] + (ci * DC + dc)[None, :], mask=live[:, None], other=0.0)
        s = tl.dot(qc, tl.trans(_piece_rows(krow, frow, onp, isf, ci * DC + dc, KIND, HW_FP8, qc.dtype)), s)
    return s


@triton.jit
def _fresh_values(acc0, acc1, acc2, acc3, alpha, pv, vrow, frow, onp, isf, DV: tl.constexpr, DC: tl.constexpr,
                  KIND: tl.constexpr, HW_FP8: tl.constexpr):
    """_piece_values with some of the piece's values from fresh rows."""
    dc = tl.arange(0, DC)
    acc0 = acc0 * alpha[:, None] + tl.dot(pv, _piece_rows(vrow, frow, onp, isf, dc, KIND, HW_FP8, pv.dtype))
    if DV // DC > 1:
        acc1 = acc1 * alpha[:, None] + tl.dot(pv, _piece_rows(vrow, frow, onp, isf, DC + dc, KIND, HW_FP8, pv.dtype))
    if DV // DC > 2:
        acc2 = acc2 * alpha[:, None] + tl.dot(pv, _piece_rows(vrow, frow, onp, isf, 2 * DC + dc, KIND, HW_FP8,
                                                              pv.dtype))
    if DV // DC > 3:
        acc3 = acc3 * alpha[:, None] + tl.dot(pv, _piece_rows(vrow, frow, onp, isf, 3 * DC + dc, KIND, HW_FP8,
                                                              pv.dtype))
    return acc0, acc1, acc2, acc3


@triton.jit
def _partition(b, n_pieces, n_parts, n_keys, qrow, live, lim, anc, first, count, k_ptr, v_ptr, ks_ptr, vs_ptr,
               tab_ptr, trow, tab_stride, tab_width, kvh, sm_scale, k_mult, v_mult, fk_ptr, fv_ptr, stride_fk,
               stride_fv, fshift, ffirst, ones, vofs, win, wlo, BLOCK_M: tl.constexpr, PIECE: tl.constexpr, PAGE: tl.constexpr,
               HKV: tl.constexpr, D: tl.constexpr, DV: tl.constexpr, DVP: tl.constexpr, DC: tl.constexpr,
               KIND: tl.constexpr, HW_FP8: tl.constexpr, TREE: tl.constexpr, FRESH: tl.constexpr):
    """Partition b of a tile's query rows: pieces b, b + n_parts, ... of the row's keys (up to n_keys) in one online
    softmax -> (running maximum, sum, unnormalized output of the DVP value columns from vofs, in up to four DC-wide
    chunks). With FRESH the positions from ffirst on read their keys and values from the fresh rows (row = position -
    fshift; 16-bit, no page scales). win > 0: each row sees the win positions up to its own, and the pieces below
    wlo (the tile's lowest such position) are skipped, a piece no row sees changing no bit."""
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc0 = tl.zeros([BLOCK_M, DC], tl.float32)
    acc1 = tl.zeros([BLOCK_M, DC], tl.float32)
    acc2 = tl.zeros([BLOCK_M, DC], tl.float32)
    acc3 = tl.zeros([BLOCK_M, DC], tl.float32)
    i0 = b
    if win > 0:
        below = (tl.maximum(wlo, 0) // PIECE).to(tl.int32) - b    # this partition's pieces wholly below the window
        i0 = b + tl.cdiv(tl.maximum(below, 0), n_parts) * n_parts
    for i in range(i0, n_pieces, n_parts):
        n = i * PIECE + tl.arange(0, PIECE)
        nm = n < n_keys
        pg = tl.load(tab_ptr + trow * tab_stride + tl.minimum(n // PAGE, tab_width - 1), mask=nm, other=0)
        slot = pg.to(tl.int64) * PAGE + n % PAGE
        if FRESH:
            isf = nm & (n >= ffirst)
            onp = nm & (n < ffirst)
            t = (n - fshift).to(tl.int64)
            s = _fresh_scores(qrow, live, k_ptr + (slot * HKV + kvh) * D, fk_ptr + t * stride_fk + kvh * D, onp, isf,
                              BLOCK_M, PIECE, D, DC, KIND, HW_FP8)
        else:
            s = _piece_scores(qrow, live, k_ptr + (slot * HKV + kvh) * D, nm, BLOCK_M, PIECE, D, DC, KIND, HW_FP8)
        if KIND >= 2:
            ksc = tl.load(ks_ptr + slot * HKV + kvh, mask=nm, other=0.0)
            if FRESH:
                ksc = tl.where(isf, 1.0, ksc)
            s = s * (ksc * sm_scale)[None, :]
        elif FRESH:
            s = s * (tl.where(isf, 1.0, k_mult) * sm_scale)[None, :]
        else:
            s = s * (k_mult * sm_scale)
        if TREE:
            w = n - first
            inw = (w >= 0) & (w < count)
            bit = (anc[:, None] >> tl.where(inw, w, 0)[None, :]) & 1
            vis = nm[None, :] & ((n < first)[None, :] | (inw[None, :] & (bit != 0)))
        else:
            vis = nm[None, :] & (n[None, :] <= lim[:, None])
        if win > 0:
            vis = vis & (n[None, :] > (lim - win)[:, None])
        s = tl.where(vis, s, float("-inf"))
        p, alpha, m_new = asm.step(s, m_i, False)
        if KIND >= 2:
            vsc = tl.load(vs_ptr + slot * HKV + kvh, mask=nm, other=0.0)
            if FRESH:
                vsc = tl.where(isf, 1.0, vsc)
            pv = (p * vsc[None, :]).to(ones.dtype)
        elif FRESH:
            pv = (p * tl.where(isf, 1.0, v_mult)[None, :]).to(ones.dtype)
        else:
            pv = (p * v_mult).to(ones.dtype)
        if FRESH:
            acc0, acc1, acc2, acc3 = _fresh_values(acc0, acc1, acc2, acc3, alpha, pv,
                                                   v_ptr + (slot * HKV + kvh) * DV + vofs,
                                                   fv_ptr + t * stride_fv + kvh * DV + vofs, onp, isf, DVP, DC, KIND,
                                                   HW_FP8)
        else:
            acc0, acc1, acc2, acc3 = _piece_values(acc0, acc1, acc2, acc3, alpha, pv,
                                                   v_ptr + (slot * HKV + kvh) * DV + vofs, nm, DVP, DC, KIND, HW_FP8)
        # The row sums on the tensor cores too: a fixed summation order, whatever the tile's layout.
        l_i = l_i * alpha + asm.row_sum(p, ones, True)
        m_i = m_new
    return m_i, l_i, acc0, acc1, acc2, acc3


@triton.jit
def _window_kernel(q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, tab_ptr, trow_ptr, start_ptr, count_ptr, first_ptr, par_ptr,
                   off_ptr, po_ptr, pm_ptr, pl_ptr, out_ptr, lse_ptr, fk_ptr, fv_ptr, fshift_ptr, ffirst_ptr,
                   sink_ptr, stride_qt, stride_qh, stride_ot, stride_oh, stride_fk, stride_fv, stride_fr, tab_stride,
                   tab_width, n_parts, sm_scale, k_mult, v_mult, count_all, win, COLS: tl.constexpr, CPT: tl.constexpr,
                   HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
                   BLOCK_M: tl.constexpr, PIECE: tl.constexpr, PAGE: tl.constexpr, KIND: tl.constexpr,
                   HW_FP8: tl.constexpr, TREE: tl.constexpr, COUNTS: tl.constexpr, OFFSETS: tl.constexpr,
                   DC: tl.constexpr, FUSED: tl.constexpr, LSE: tl.constexpr, FRESH: tl.constexpr,
                   DVP: tl.constexpr, SINKS: tl.constexpr):
    """Program (column tile, partition b, row x KV head): the tile's query rows (its columns x the KV head's query
    heads) over pieces b, b + n_parts, ... of the row's keys, up to its last column's position; writes each row's
    unnormalized output, running maximum and sum into the partials. FUSED (grid (tiles, DV / DVP, rows x KV heads)):
    the program reads every partition in turn and folds them as the merge does (_fold), writing the output (and the
    log-sum-exps) itself: the same bits, no partials; program b holds value columns b * DVP .. b * DVP + DVP - 1 (its
    scores computed by each such program, the same in each). FRESH: per row, the positions from ffirst on from the
    fresh rows (fshift, ffirst: int64 per row, stride_fr apart, 0 for one for every row). SINKS (fused): each query
    head's sink logit (sink_ptr, FP32 [HQ]) folded into its rows' final states, as the merge folds it."""
    mt = tl.program_id(0)
    b = tl.program_id(1)
    rk = tl.program_id(2)
    r = rk // HKV
    kvh = rk % HKV
    if COUNTS:
        count = tl.minimum(tl.load(count_ptr + r), COLS)
    else:
        count = count_all
    c0 = mt * CPT
    first = tl.load(first_ptr + r).to(tl.int64)
    n_keys = first + tl.minimum(count, c0 + CPT)
    n_pieces = tl.cdiv(n_keys, PIECE).to(tl.int32)
    if FUSED:
        part0 = 0
        vofs = b * DVP
    else:
        part0 = b
        vofs = 0
    if (c0 < count) & (part0 < n_pieces):
        start = tl.load(start_ptr + r).to(tl.int64)
        trow = tl.load(trow_ptr + r).to(tl.int64)
        j = tl.arange(0, BLOCK_M)
        c = (c0 + j // G).to(tl.int64)
        g = j % G
        live = (j < CPT * G) & (c < count)
        qrow = q_ptr + (start + c) * stride_qt + (kvh * G + g) * stride_qh
        lim = first + c
        anc = c
        if TREE:
            anc = _ancestors(par_ptr, start, c, live, COLS)
        fshift = first
        ffirst = first
        if FRESH:
            fshift = tl.load(fshift_ptr + r * stride_fr)
            ffirst = tl.load(ffirst_ptr + r * stride_fr)
        ones = tl.full([PIECE, 16], 1.0, q_ptr.dtype.element_ty)
        if FUSED:
            m_run = tl.full([BLOCK_M], float("-inf"), tl.float32)
            l_run = tl.zeros([BLOCK_M], tl.float32)
            out0 = tl.zeros([BLOCK_M, DC], tl.float32)
            out1 = tl.zeros([BLOCK_M, DC], tl.float32)
            out2 = tl.zeros([BLOCK_M, DC], tl.float32)
            out3 = tl.zeros([BLOCK_M, DC], tl.float32)
            # A column folds the partitions the merge folds for it: those that hold its positions.
            held = tl.minimum(n_parts, tl.cdiv(lim + 1, PIECE))
            for part in range(0, tl.minimum(n_parts, n_pieces)):
                m_i, l_i, acc0, acc1, acc2, acc3 = _partition(
                    part, n_pieces, n_parts, n_keys, qrow, live, lim, anc, first, count, k_ptr, v_ptr, ks_ptr, vs_ptr,
                    tab_ptr, trow, tab_stride, tab_width, kvh, sm_scale, k_mult, v_mult, fk_ptr, fv_ptr, stride_fk,
                    stride_fv, fshift, ffirst, ones, vofs, win, first + c0 - win + 1, BLOCK_M, PIECE, PAGE, HKV, D,
                    DV, DVP, DC, KIND, HW_FP8, TREE, FRESH)
                # One partition's state into the running one, in partition order (as the merge folds them).
                m_new, l_new, keep, take = asm.fold(m_run, l_run, m_i, l_i, False)
                use = part < held
                m_run = tl.where(use, m_new, m_run)
                l_run = tl.where(use, l_new, l_run)
                out0 = tl.where(use[:, None], tl.fma(out0, keep[:, None], acc0 * take[:, None]), out0)
                if DVP // DC > 1:
                    out1 = tl.where(use[:, None], tl.fma(out1, keep[:, None], acc1 * take[:, None]), out1)
                if DVP // DC > 2:
                    out2 = tl.where(use[:, None], tl.fma(out2, keep[:, None], acc2 * take[:, None]), out2)
                if DVP // DC > 3:
                    out3 = tl.where(use[:, None], tl.fma(out3, keep[:, None], acc3 * take[:, None]), out3)
            if SINKS:
                sk = tl.load(sink_ptr + kvh * G + g, mask=live, other=float("-inf"))
                m_run, l_run, keep = asm.sink(m_run, l_run, sk, False)
                out0 = out0 * keep[:, None]
                if DVP // DC > 1:
                    out1 = out1 * keep[:, None]
                if DVP // DC > 2:
                    out2 = out2 * keep[:, None]
                if DVP // DC > 3:
                    out3 = out3 * keep[:, None]
            dc = tl.arange(0, DC)
            row = out_ptr + (start + c)[:, None] * stride_ot + (kvh * G + g)[:, None] * stride_oh + vofs
            ok = l_run != 0   # 0 for a row with no position; a NaN sum (a NaN query) stays NaN
            dtype = out_ptr.dtype.element_ty
            tl.store(row + dc[None, :], tl.where(ok[:, None], out0 / l_run[:, None], 0.0).to(dtype), mask=live[:, None])
            if DVP // DC > 1:
                tl.store(row + (DC + dc)[None, :], tl.where(ok[:, None], out1 / l_run[:, None], 0.0).to(dtype),
                         mask=live[:, None])
            if DVP // DC > 2:
                tl.store(row + (2 * DC + dc)[None, :], tl.where(ok[:, None], out2 / l_run[:, None], 0.0).to(dtype),
                         mask=live[:, None])
            if DVP // DC > 3:
                tl.store(row + (3 * DC + dc)[None, :], tl.where(ok[:, None], out3 / l_run[:, None], 0.0).to(dtype),
                         mask=live[:, None])
            if LSE:
                tl.store(lse_ptr + (start + c) * HQ + kvh * G + g, asm.lse(m_run, l_run, False), mask=live & (b == 0))
        else:
            m_i, l_i, acc0, acc1, acc2, acc3 = _partition(
                b, n_pieces, n_parts, n_keys, qrow, live, lim, anc, first, count, k_ptr, v_ptr, ks_ptr, vs_ptr,
                tab_ptr, trow, tab_stride, tab_width, kvh, sm_scale, k_mult, v_mult, fk_ptr, fv_ptr, stride_fk,
                stride_fv, fshift, ffirst, ones, vofs, win, first + c0 - win + 1, BLOCK_M, PIECE, PAGE, HKV, D, DV,
                DVP, DC, KIND, HW_FP8, TREE, FRESH)
            idx = ((_partial_row(off_ptr, r, COLS, OFFSETS) + c) * HQ + kvh * G + g) * n_parts + b
            _store_partials(po_ptr, idx, live, acc0, acc1, acc2, acc3, DV, DC)
            tl.store(pm_ptr + idx, m_i, mask=live)
            tl.store(pl_ptr + idx, l_i, mask=live)


@triton.jit
def _sparse_kernel(q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, tab_ptr, trow_ptr, start_ptr, count_ptr, first_ptr, par_ptr,
                   off_ptr, blk_ptr, blk_stride, n_blocks, block_size, po_ptr, pm_ptr, pl_ptr, stride_qt, stride_qh,
                   tab_stride, tab_width, n_parts, sm_scale, k_mult, v_mult, count_all, COLS: tl.constexpr,
                   HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
                   BLOCK_M: tl.constexpr, PIECE: tl.constexpr, PAGE: tl.constexpr, KIND: tl.constexpr,
                   HW_FP8: tl.constexpr, TREE: tl.constexpr, COUNTS: tl.constexpr, OFFSETS: tl.constexpr,
                   DC: tl.constexpr):
    """Program (partition b, KV head, row x column): one token's query heads of the KV head over the positions of its
    listed KV blocks (blocks[token][0 .. n_blocks - 1], -1 for none), read as one run of n_blocks * block_size
    positions in list order, its tiles b, b + n_parts, ...; positions past the token's own (or outside its tree) are
    not seen. Writes partials for every b below the tile count, empty ones included."""
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    rc = tl.program_id(2)
    r = rc // COLS
    c = (rc % COLS).to(tl.int64)
    if COUNTS:
        count = tl.minimum(tl.load(count_ptr + r), COLS)
    else:
        count = count_all
    n_tiles = tl.cdiv(n_blocks * block_size, PIECE)
    if (c < count) & (b < n_tiles):
        start = tl.load(start_ptr + r).to(tl.int64)
        first = tl.load(first_ptr + r).to(tl.int64)
        trow = tl.load(trow_ptr + r).to(tl.int64)
        tok = start + c
        g = tl.arange(0, BLOCK_M)
        live = g < G
        qrow = q_ptr + tok * stride_qt + (kvh * G + g) * stride_qh
        if TREE:
            # Every row is the same column: its mask (identical on every row) taken as a scalar.
            anc = tl.max(_ancestors(par_ptr, start, tl.zeros([BLOCK_M], tl.int64) + c, g >= 0, COLS), axis=0)
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc0 = tl.zeros([BLOCK_M, DC], tl.float32)
        acc1 = tl.zeros([BLOCK_M, DC], tl.float32)
        acc2 = tl.zeros([BLOCK_M, DC], tl.float32)
        acc3 = tl.zeros([BLOCK_M, DC], tl.float32)
        ones = tl.full([PIECE, 16], 1.0, q_ptr.dtype.element_ty)
        for i in range(b, n_tiles, n_parts):
            u = i * PIECE + tl.arange(0, PIECE)
            e = u // block_size
            blk = tl.load(blk_ptr + tok * blk_stride + e, mask=e < n_blocks, other=-1).to(tl.int64)
            n = blk * block_size + u % block_size
            if TREE:
                w = n - first
                inw = (w >= 0) & (w < count)
                seen = (n < first) | (inw & (((anc >> tl.where(inw, w, 0)) & 1) != 0))
            else:
                seen = n <= first + c
            nm = (blk >= 0) & (n < first + count) & seen
            pg = tl.load(tab_ptr + trow * tab_stride + tl.minimum(n // PAGE, tab_width - 1), mask=nm, other=0)
            slot = pg.to(tl.int64) * PAGE + n % PAGE
            s = _piece_scores(qrow, live, k_ptr + (slot * HKV + kvh) * D, nm, BLOCK_M, PIECE, D, DC, KIND, HW_FP8)
            if KIND >= 2:
                ksc = tl.load(ks_ptr + slot * HKV + kvh, mask=nm, other=0.0)
                s = s * (ksc * sm_scale)[None, :]
            else:
                s = s * (k_mult * sm_scale)
            s = tl.where(nm[None, :], s, float("-inf"))
            p, alpha, m_new = asm.step(s, m_i, False)
            if KIND >= 2:
                vsc = tl.load(vs_ptr + slot * HKV + kvh, mask=nm, other=0.0)
                pv = (p * vsc[None, :]).to(ones.dtype)
            else:
                pv = (p * v_mult).to(ones.dtype)
            acc0, acc1, acc2, acc3 = _piece_values(acc0, acc1, acc2, acc3, alpha, pv, v_ptr + (slot * HKV + kvh) * DV,
                                                   nm, DV, DC, KIND, HW_FP8)
            l_i = l_i * alpha + asm.row_sum(p, ones, True)
            m_i = m_new
        idx = ((_partial_row(off_ptr, r, COLS, OFFSETS) + c) * HQ + kvh * G + g) * n_parts + b
        _store_partials(po_ptr, idx, live, acc0, acc1, acc2, acc3, DV, DC)
        tl.store(pm_ptr + idx, m_i, mask=live)
        tl.store(pl_ptr + idx, l_i, mask=live)


@triton.jit
def _merge_kernel(po_ptr, pm_ptr, pl_ptr, out_ptr, lse_ptr, start_ptr, count_ptr, first_ptr, off_ptr, sink_ptr, n_parts,
                  n_fixed, stride_ot, stride_oh, count_all, COLS: tl.constexpr, HQ: tl.constexpr, DV: tl.constexpr,
                  PIECE: tl.constexpr, COUNTS: tl.constexpr, OFFSETS: tl.constexpr, LSE: tl.constexpr,
                  BLOCK_P: tl.constexpr, CHUNK_P: tl.constexpr, SINKS: tl.constexpr):
    """Program (column, query head, row): the partials of the partitions that hold the column's keys folded in
    partition order (asm.fold: a running maximum, the sum and output rescaled to it), into the output row (and its
    log-sum-exp). The loop takes CHUNK_P partitions a turn (their loads issued together), each folded in order, so a
    column's bits are the fused window kernel's too. SINKS: the head's sink logit folded in last."""
    c = tl.program_id(0)
    h = tl.program_id(1)
    r = tl.program_id(2)
    if COUNTS:
        count = tl.minimum(tl.load(count_ptr + r), COLS)
    else:
        count = count_all
    if c < count:
        start = tl.load(start_ptr + r).to(tl.int64)
        if n_fixed > 0:
            nb = tl.minimum(n_parts, n_fixed)
        else:
            first = tl.load(first_ptr + r).to(tl.int64)
            nb = tl.minimum(n_parts, tl.cdiv(first + c + 1, PIECE)).to(tl.int32)
        base = ((_partial_row(off_ptr, r, COLS, OFFSETS) + c) * HQ + h) * n_parts
        dv = tl.arange(0, DV)
        m_run = tl.max(tl.full([CHUNK_P], float("-inf"), tl.float32), axis=0)
        l_run = tl.sum(tl.zeros([CHUNK_P], tl.float32), axis=0)
        acc = tl.zeros([DV], tl.float32)
        for b0 in range(0, nb, CHUNK_P):
            for k in tl.static_range(CHUNK_P):
                use = b0 + k < nb
                at = base + b0 + k
                m = tl.load(pm_ptr + at, mask=use, other=float("-inf"))
                l = tl.load(pl_ptr + at, mask=use, other=0.0)
                o = tl.load(po_ptr + at * DV + dv, mask=(dv < DV) & use, other=0.0)
                m_new, l_new, keep, take = asm.fold(m_run, l_run, m, l, False)
                m_run = tl.where(use, m_new, m_run)
                l_run = tl.where(use, l_new, l_run)
                acc = tl.where(use, tl.fma(acc, keep, o * take), acc)
        if SINKS:
            m_run, l_run, keep = asm.sink(m_run, l_run, tl.load(sink_ptr + h), False)
            acc = acc * keep
        ok = l_run != 0   # 0 for a row with no position; a NaN sum (a NaN query) stays NaN
        o = tl.where(ok, acc / l_run, 0.0)
        tl.store(out_ptr + (start + c) * stride_ot + h * stride_oh + dv, o.to(out_ptr.dtype.element_ty))
        if LSE:
            tl.store(lse_ptr + (start + c) * HQ + h, asm.lse(m_run, l_run, False))


# -- prompt chunks ----------------------------------------------------------------------------------------------------
@triton.jit
def _prefill_kernel(q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, tab_ptr, out_ptr, lse_ptr, pos_ptr, fk_ptr, fv_ptr, first_ptr,
                    sink_ptr, win, T, pos0, first0, sm_scale, k_mult, v_mult, stride_qt, stride_qh, stride_ot, stride_oh, stride_fk,
                    stride_fv,
                    HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr,
                    BLOCK_N: tl.constexpr, PAGE: tl.constexpr, KIND: tl.constexpr, HW_FP8: tl.constexpr,
                    INT8_QK: tl.constexpr, LSE: tl.constexpr, POS_T: tl.constexpr, FRESH: tl.constexpr,
                    FIRST_T: tl.constexpr, SINKS: tl.constexpr):
    """Program (query block, query head): a chunk of one sequence's queries at positions pos0.. over its positions
    [0, own], causal: through its page table, or with FRESH the positions before `first` through it and the chunk's
    own from `first` on from its fresh keys and values (fk_ptr, fv_ptr: [T][HKV][D] in the queries' dtype, as the
    projections gave them; first: first0, or with FIRST_T read from first_ptr on the device, at least pos0); with POS_T
    pos0 is read from pos_ptr on the device. SINKS: the head's sink logit folded into every row's final state. win > 0:
    each query sees the win positions up to its own, the blocks below the tile's window skipped."""
    if POS_T:
        pos0 = tl.load(pos_ptr).to(tl.int32)
    if FIRST_T:
        first0 = tl.load(first_ptr).to(tl.int32)
    mb = tl.program_id(0)
    head = tl.program_id(1)
    kvh = head // G
    rows = mb * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < T
    d = tl.arange(0, D)
    q = tl.load(q_ptr + rows[:, None] * stride_qt + head * stride_qh + d[None, :], mask=rmask[:, None], other=0.0)
    if INT8_QK:
        # Each query row to INT8 at its own scale (amax / 127, rounded half away from zero), in registers.
        qf = q.to(tl.float32)
        qs = tl.maximum(tl.max(tl.abs(qf), axis=1) / 127.0, 1e-30)
        qf = qf / qs[:, None]
        qi = tl.where(qf >= 0, qf + 0.5, qf - 0.5).to(tl.int8)
        qs = qs * sm_scale
    qpos = pos0 + rows
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    end = tl.minimum(pos0 + (mb + 1) * BLOCK_M, pos0 + T)
    paged = tl.minimum(tl.maximum(first0, pos0), end) if FRESH else end      # the positions read from the pages
    lo = 0
    if win > 0:   # the tile's lowest position seen, rounded down to a block
        lo = tl.maximum(pos0 + mb * BLOCK_M - win + 1, 0) // BLOCK_N * BLOCK_N
    for n0 in range(lo, paged, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        nm = n < paged
        pg = tl.load(tab_ptr + n // PAGE, mask=nm, other=0)
        slot = pg.to(tl.int64) * PAGE + n % PAGE
        kc = tl.load(k_ptr + (slot * HKV + kvh)[:, None] * D + d[None, :], mask=nm[:, None], other=0)
        if INT8_QK:
            ksc = tl.load(ks_ptr + slot * HKV + kvh, mask=nm, other=0.0)
            s = tl.dot(qi, tl.trans(kc)).to(tl.float32) * qs[:, None] * ksc[None, :]
        elif KIND >= 2:
            ksc = tl.load(ks_ptr + slot * HKV + kvh, mask=nm, other=0.0)
            s = tl.dot(q, tl.trans(_values(kc, KIND, HW_FP8, q.dtype))) * (ksc * sm_scale)[None, :]
        else:
            s = tl.dot(q, tl.trans(_values(kc, KIND, HW_FP8, q.dtype))) * (k_mult * sm_scale)
        vis = (n[None, :] <= qpos[:, None]) & nm[None, :]
        if win > 0:
            vis = vis & (n[None, :] > (qpos - win)[:, None])
        s = tl.where(vis, s, float("-inf"))
        p, alpha, m_new = asm.step(s, m_i, False)
        l_i = l_i * alpha + asm.row_sum(p, p, False)
        vc = tl.load(v_ptr + (slot * HKV + kvh)[:, None] * D + d[None, :], mask=nm[:, None], other=0)
        if KIND >= 2:
            vsc = tl.load(vs_ptr + slot * HKV + kvh, mask=nm, other=0.0)
            pv = (p * vsc[None, :]).to(q.dtype)
        else:
            pv = (p * v_mult).to(q.dtype)
        acc = acc * alpha[:, None] + tl.dot(pv, _values(vc, KIND, HW_FP8, q.dtype))
        m_i = m_new
    if FRESH:
        # The chunk's own positions from `first` on from its fresh 16-bit keys and values, no page scales.
        for n0 in range(tl.maximum(paged, lo), end, BLOCK_N):
            n = n0 + tl.arange(0, BLOCK_N)
            nm = n < end
            t = (n - pos0).to(tl.int64)
            kc = tl.load(fk_ptr + t[:, None] * stride_fk + kvh * D + d[None, :], mask=nm[:, None], other=0.0)
            s = tl.dot(q, tl.trans(kc.to(q.dtype))) * sm_scale
            vis = (n[None, :] <= qpos[:, None]) & nm[None, :]
            if win > 0:
                vis = vis & (n[None, :] > (qpos - win)[:, None])
            s = tl.where(vis, s, float("-inf"))
            p, alpha, m_new = asm.step(s, m_i, False)
            l_i = l_i * alpha + asm.row_sum(p, p, False)
            vc = tl.load(fv_ptr + t[:, None] * stride_fv + kvh * D + d[None, :], mask=nm[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), vc.to(q.dtype))
            m_i = m_new
    if SINKS:
        m_i, l_i, keep = asm.sink(m_i, l_i, tl.load(sink_ptr + head), False)
        acc = acc * keep[:, None]
    o = acc / l_i[:, None]
    tl.store(out_ptr + rows[:, None] * stride_ot + head * stride_oh + d[None, :], o.to(out_ptr.dtype.element_ty),
             mask=rmask[:, None])
    if LSE:
        tl.store(lse_ptr + rows * HQ + head, m_i + tl.log(l_i), mask=rmask)
