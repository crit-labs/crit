"""The flash structures' Triton kernels (see __init__.py: attend(..., flash=True), prefill(..., flash=True)).

One kernel reads windows (decode tokens, verify windows, prompt pieces) and prompt chunks over the pages: a program's
query tile packs a row's columns times a KV head's query heads (row j of the tile: column j // G, head j % G, so no
tile row is left empty between columns and every key block is read once for all the heads that share it); the row's
keys split into contiguous ranges of `chunk` positions, each walked in blocks of BLOCK_N positions with the online
softmax of the kernel core (crit_kcore.attention_softmax, base 2: the softmax scale carries log2(e)) and FP32 row sums;
whole blocks every row of the tile sees skip the causal mask. Split launches write each range's partial state for the
merge (_flash_merge_kernel), fused ones fold the ranges in order in the program (the same rule, the same bits) and
write the output; ONE_SPLIT launches (a prompt chunk read in one range) write it directly.

A row's bits: the scores and the products are mma.sync dots (each element in the instruction's k-order, whatever the
tile), the maximum is exact, and a row's sum is the tile layout's reduction, the same for every row of every tile of
one shape (BLOCK_M rows in num_warps warps): so launches that share the tile, the block and the chunk give a row the
same bits wherever it sits (window launches take the cache's tile, flash_tile(), and mask every block: SKIP False).

Mechanisms after FlashInfer's batch prefill over paged KV (Apache-2.0): grouped query heads packed into the query
tile's rows, contiguous KV splits merged by their states, the mask only where the causal edge crosses a block, the
softmax scale folded with log2(e) into one multiply before exp2, the longest tiles first.
"""
import triton
import triton.language as tl

from crit_kcore import attention_softmax as asm

from ._kernels import _ancestors, _values


@triton.jit
def _flash_range(acc, m_i, l_i, q, qi, qs, lo, hi, lim, anc, first, count, kvh, k_ptr, v_ptr, ks_ptr, vs_ptr, tab_ptr,
                 trow, tab_stride, tab_width, fk_ptr, fv_ptr, stride_fk, stride_fv, fshift, qk_scale, k_mult, v_mult,
                 vofs, win, BLOCK_N: tl.constexpr, PAGE: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                 DV: tl.constexpr, DVP: tl.constexpr, KIND: tl.constexpr, HW_FP8: tl.constexpr, TREE: tl.constexpr,
                 MASKED: tl.constexpr, FRESH_SRC: tl.constexpr, INT8_QK: tl.constexpr):
    """Positions lo .. hi - 1 in blocks of BLOCK_N from lo, into the running (acc, m_i, l_i). MASKED: each row sees
    positions up to its own (lim) or, with TREE, those before the window and its ancestors' (anc); else every block is
    seen by every row in full. FRESH_SRC: keys and values from the fresh 16-bit rows (row = position - fshift), else
    through the page table at the page kind's precision. win > 0 (masked blocks): each row sees the win positions up
    to its own."""
    d = tl.arange(0, D)
    dv = vofs + tl.arange(0, DVP)
    for n0 in range(lo, hi, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        nm = n < hi
        if FRESH_SRC:
            t = (n - fshift).to(tl.int64)
            kc = tl.load(fk_ptr + t[:, None] * stride_fk + kvh * D + d[None, :], mask=nm[:, None], other=0.0)
            s = tl.dot(q, tl.trans(kc.to(q.dtype))) * qk_scale
        else:
            pg = tl.load(tab_ptr + trow * tab_stride + tl.minimum(n // PAGE, tab_width - 1), mask=nm, other=0)
            slot = pg.to(tl.int64) * PAGE + n % PAGE
            kc = tl.load(k_ptr + (slot * HKV + kvh)[:, None] * D + d[None, :], mask=nm[:, None], other=0)
            if INT8_QK:
                ksc = tl.load(ks_ptr + slot * HKV + kvh, mask=nm, other=0.0)
                s = tl.dot(qi, tl.trans(kc)).to(tl.float32) * qs[:, None] * ksc[None, :]
            elif KIND >= 2:
                ksc = tl.load(ks_ptr + slot * HKV + kvh, mask=nm, other=0.0)
                s = tl.dot(q, tl.trans(_values(kc, KIND, HW_FP8, q.dtype))) * (ksc * qk_scale)[None, :]
            else:
                s = tl.dot(q, tl.trans(_values(kc, KIND, HW_FP8, q.dtype))) * (k_mult * qk_scale)
        if MASKED:
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
        p, alpha, m_new = asm.step(s, m_i, True)
        if FRESH_SRC:
            vc = tl.load(fv_ptr + t[:, None] * stride_fv + kvh * DV + dv[None, :], mask=nm[:, None], other=0.0)
            pv = p.to(q.dtype)
            vv = vc.to(q.dtype)
        else:
            vc = tl.load(v_ptr + (slot * HKV + kvh)[:, None] * DV + dv[None, :], mask=nm[:, None], other=0)
            if KIND >= 2:
                vsc = tl.load(vs_ptr + slot * HKV + kvh, mask=nm, other=0.0)
                pv = (p * vsc[None, :]).to(q.dtype)
            else:
                pv = (p * v_mult).to(q.dtype)
            vv = _values(vc, KIND, HW_FP8, q.dtype)
        acc = acc * alpha[:, None] + tl.dot(pv, vv)
        l_i = l_i * alpha + asm.row_sum(p, None, False)
        m_i = m_new
    return acc, m_i, l_i


@triton.jit
def _aligned(lo, edge, hi, BLOCK_N: tl.constexpr):
    """The end of the whole blocks from lo that lie below min(edge, hi): lo + a multiple of BLOCK_N."""
    top = tl.minimum(edge, hi)
    return tl.where(top > lo, lo + ((top - lo) // BLOCK_N) * BLOCK_N, lo)


@triton.jit
def _flash_split(lo, hi, nfree, ffirst, q, qi, qs, lim, anc, first, count, kvh, k_ptr, v_ptr, ks_ptr, vs_ptr, tab_ptr,
                 trow, tab_stride, tab_width, fk_ptr, fv_ptr, stride_fk, stride_fv, fshift, qk_scale, k_mult, v_mult,
                 vofs, win, wlo, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, PAGE: tl.constexpr, HKV: tl.constexpr,
                 D: tl.constexpr, DV: tl.constexpr, DVP: tl.constexpr, KIND: tl.constexpr, HW_FP8: tl.constexpr,
                 TREE: tl.constexpr, FRESH: tl.constexpr, INT8_QK: tl.constexpr, SKIP: tl.constexpr):
    """One split, positions lo .. hi - 1 -> its state (acc, m, l): with SKIP the whole blocks every row sees (below
    nfree) without the mask and the rest masked, else every block masked (one loop, fewer registers); with FRESH the
    positions from ffirst on from the fresh rows. SKIP changes bits: without a mask between them the scale's multiply
    and the maximum's subtraction may contract into one FMA, so a block read unmasked rounds otherwise than masked;
    launches whose rows must agree (every window launch) mask every block. win > 0: each row sees the win positions up
    to its own; the blocks wholly below wlo (the tile's lowest such position) are skipped and every block is masked."""
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    skip = tl.where(win > 0, wlo, lo)   # the blocks wholly below it are skipped
    start = tl.minimum(hi, tl.where(skip > lo, lo + (skip - lo) // BLOCK_N * BLOCK_N, lo))
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, DVP], tl.float32)
    if FRESH:
        paged = tl.minimum(hi, tl.maximum(ffirst, start))
    else:
        paged = hi
    edge = start
    if SKIP:
        edge = tl.where(win > 0, start, _aligned(start, nfree, paged, BLOCK_N))
        acc, m_i, l_i = _flash_range(acc, m_i, l_i, q, qi, qs, start, edge, lim, anc, first, count, kvh, k_ptr, v_ptr,
                                     ks_ptr, vs_ptr, tab_ptr, trow, tab_stride, tab_width, fk_ptr, fv_ptr, stride_fk,
                                     stride_fv, fshift, qk_scale, k_mult, v_mult, vofs, win, BLOCK_N, PAGE, HKV, D,
                                     DV, DVP, KIND, HW_FP8, TREE, False, False, INT8_QK)
    acc, m_i, l_i = _flash_range(acc, m_i, l_i, q, qi, qs, edge, paged, lim, anc, first, count, kvh, k_ptr, v_ptr,
                                 ks_ptr, vs_ptr, tab_ptr, trow, tab_stride, tab_width, fk_ptr, fv_ptr, stride_fk,
                                 stride_fv, fshift, qk_scale, k_mult, v_mult, vofs, win, BLOCK_N, PAGE, HKV, D, DV,
                                 DVP, KIND, HW_FP8, TREE, True, False, INT8_QK)
    if FRESH:
        edge = paged
        if SKIP:
            edge = tl.where(win > 0, paged, _aligned(paged, nfree, hi, BLOCK_N))
            acc, m_i, l_i = _flash_range(acc, m_i, l_i, q, qi, qs, paged, edge, lim, anc, first, count, kvh, k_ptr,
                                         v_ptr, ks_ptr, vs_ptr, tab_ptr, trow, tab_stride, tab_width, fk_ptr, fv_ptr,
                                         stride_fk, stride_fv, fshift, qk_scale, k_mult, v_mult, vofs, win, BLOCK_N,
                                         PAGE, HKV, D, DV, DVP, KIND, HW_FP8, TREE, False, True, False)
        acc, m_i, l_i = _flash_range(acc, m_i, l_i, q, qi, qs, edge, hi, lim, anc, first, count, kvh, k_ptr, v_ptr,
                                     ks_ptr, vs_ptr, tab_ptr, trow, tab_stride, tab_width, fk_ptr, fv_ptr, stride_fk,
                                     stride_fv, fshift, qk_scale, k_mult, v_mult, vofs, win, BLOCK_N, PAGE, HKV, D,
                                     DV, DVP, KIND, HW_FP8, TREE, True, True, False)
    return acc, m_i, l_i


@triton.jit
def _row_chunk(first, count, n_parts, chunk, BLOCK_N: tl.constexpr, DEV_CHUNK: tl.constexpr):
    """Positions a split of this row: `chunk`, or with DEV_CHUNK the row's positions over n_parts splits in whole
    blocks (read on the device: one launch for every position)."""
    found = chunk
    if DEV_CHUNK:
        found = (tl.cdiv(tl.cdiv(first + count, n_parts), BLOCK_N) * BLOCK_N).to(tl.int32)
    return found


@triton.jit(do_not_specialize=["first0", "ffirst0"])
def _flash_kernel(q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, tab_ptr, trow_ptr, start_ptr, count_ptr, first_ptr, par_ptr,
                  off_ptr, po_ptr, pm_ptr, pl_ptr, out_ptr, lse_ptr, fk_ptr, fv_ptr, fshift_ptr, ffirst_ptr,
                  sink_ptr, stride_qt, stride_qh, stride_ot, stride_oh, stride_fk, stride_fv, stride_fr, tab_stride,
                  tab_width, n_parts, chunk, qk_scale, k_mult, v_mult, cols, first0, ffirst0, win,
                  HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
                  DVP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, PAGE: tl.constexpr,
                  KIND: tl.constexpr, HW_FP8: tl.constexpr, TREE: tl.constexpr, TREE_COLS: tl.constexpr,
                  COUNTS: tl.constexpr, OFFSETS: tl.constexpr, FUSED: tl.constexpr, ONE_SPLIT: tl.constexpr,
                  LSE: tl.constexpr, FRESH: tl.constexpr, ONE_ROW: tl.constexpr, POS_T: tl.constexpr,
                  FIRST_T: tl.constexpr, DEV_CHUNK: tl.constexpr, INT8_QK: tl.constexpr, SKIP: tl.constexpr,
                  SINKS: tl.constexpr):
    """Program (query tile, split b or value part b, row x KV head). Tile rows j: packed index mt * BLOCK_M + j ->
    column // G, head % G of the row's KV head. Split launches: split b's positions (b * chunk ..) up to the tile's
    last column, its state into the partials; FUSED: every split in turn folded in split order (a column folds those
    holding its positions, as the merge does), value columns b * DVP .., the output (and lse) written; ONE_SPLIT: the
    positions in one range, the output written. ONE_ROW: one sequence's chunk (prefill): row 0, token 0 on, table row
    0, its first position first0 (or with POS_T first_ptr[0] on the device), `cols` columns; FRESH: positions from
    ffirst on from the fresh rows (row = position - fshift: per row from fshift_ptr / ffirst_ptr stride_fr apart, or
    with ONE_ROW fshift = first, ffirst = ffirst0 or with FIRST_T ffirst_ptr[0]). SINKS (fused, one split): each
    query head's sink logit (sink_ptr, FP32 [HQ]) folded into its rows' final states, as the merge folds it."""
    # The last tiles (the most positions under a causal edge) first: the GPU's last wave holds the short ones.
    mt = tl.num_programs(0) - 1 - tl.program_id(0)
    b = tl.program_id(1)
    rk = tl.program_id(2)
    r = rk // HKV
    kvh = rk % HKV
    if ONE_ROW:
        count = cols
        if POS_T:
            first = tl.load(first_ptr).to(tl.int64)
        else:
            first = first0.to(tl.int64)
    else:
        if COUNTS:
            count = tl.minimum(tl.load(count_ptr + r), cols)
        else:
            count = cols
        first = tl.load(first_ptr + r).to(tl.int64)
    j0 = mt * BLOCK_M
    c_lo = j0 // G
    c_hi = tl.minimum((j0 + BLOCK_M - 1) // G, count - 1)
    n_keys = first + c_hi + 1
    chunk = _row_chunk(first, count, n_parts, chunk, BLOCK_N, DEV_CHUNK)
    if FUSED | ONE_SPLIT:
        go = c_lo < count
        vofs = b * DVP
    else:
        go = (c_lo < count) & (b * chunk < n_keys)
        vofs = 0
    if go:
        if ONE_ROW:
            start = 0
            trow = 0
        else:
            start = tl.load(start_ptr + r).to(tl.int64)
            trow = tl.load(trow_ptr + r).to(tl.int64)
        j = j0 + tl.arange(0, BLOCK_M)
        c = (j // G).to(tl.int64)
        g = j % G
        live = c < count
        d = tl.arange(0, D)
        q = tl.load(q_ptr + ((start + c) * stride_qt + (kvh * G + g) * stride_qh)[:, None] + d[None, :],
                    mask=live[:, None], other=0.0)
        qi = q
        qs = tl.zeros([BLOCK_M], tl.float32)
        if INT8_QK:
            # Each query row to INT8 at its own scale (amax / 127, rounded half away from zero), in registers.
            qf = q.to(tl.float32)
            qs = tl.maximum(tl.max(tl.abs(qf), axis=1) / 127.0, 1e-30)
            qf = qf / qs[:, None]
            qi = tl.where(qf >= 0, qf + 0.5, qf - 0.5).to(tl.int8)
            qs = qs * qk_scale
        lim = first + c
        anc = c
        if TREE:
            anc = _ancestors(par_ptr, start, c, live, TREE_COLS)
            nfree = first
        else:
            nfree = first + c_lo + 1
        fshift = first
        ffirst = first
        if FRESH:
            if ONE_ROW:
                if FIRST_T:
                    ffirst = tl.load(ffirst_ptr).to(tl.int64)
                else:
                    ffirst = ffirst0.to(tl.int64)
            else:
                fshift = tl.load(fshift_ptr + r * stride_fr)
                ffirst = tl.load(ffirst_ptr + r * stride_fr)
        if FUSED | ONE_SPLIT:
            if ONE_SPLIT:
                out, m_run, l_run = _flash_split(0, n_keys, nfree, ffirst, q, qi, qs, lim, anc, first, count, kvh,
                                                 k_ptr, v_ptr, ks_ptr, vs_ptr, tab_ptr, trow, tab_stride, tab_width,
                                                 fk_ptr, fv_ptr, stride_fk, stride_fv, fshift, qk_scale, k_mult,
                                                 v_mult, vofs, win, first + c_lo - win + 1, BLOCK_M, BLOCK_N, PAGE, HKV, D, DV, DVP, KIND, HW_FP8,
                                                 TREE, FRESH, INT8_QK, SKIP)
            else:
                m_run = tl.full([BLOCK_M], float("-inf"), tl.float32)
                l_run = tl.zeros([BLOCK_M], tl.float32)
                out = tl.zeros([BLOCK_M, DVP], tl.float32)
                # Every split of the tile in order: a split past a column's own position leaves it no position seen
                # (maximum -inf, sum and output 0), and folding that state changes no bit of the running one, so each
                # column gets the fold of the splits holding its positions, as the merge folds them.
                for part in range(0, tl.cdiv(n_keys, chunk)):
                    lo = part * chunk
                    acc, m_i, l_i = _flash_split(lo, tl.minimum(lo + chunk, n_keys), nfree, ffirst, q, qi, qs, lim,
                                                 anc, first, count, kvh, k_ptr, v_ptr, ks_ptr, vs_ptr, tab_ptr, trow,
                                                 tab_stride, tab_width, fk_ptr, fv_ptr, stride_fk, stride_fv, fshift,
                                                 qk_scale, k_mult, v_mult, vofs, win, first + c_lo - win + 1, BLOCK_M, BLOCK_N, PAGE, HKV, D, DV,
                                                 DVP, KIND, HW_FP8, TREE, FRESH, INT8_QK, SKIP)
                    m_run, l_run, keep, take = asm.fold(m_run, l_run, m_i, l_i, True)
                    out = tl.fma(out, keep[:, None], acc * take[:, None])
            if SINKS:
                sk = tl.load(sink_ptr + kvh * G + g, mask=live, other=float("-inf"))
                m_run, l_run, keep = asm.sink(m_run, l_run, sk, True)
                out = out * keep[:, None]
            dvp = tl.arange(0, DVP)
            row = ((start + c) * stride_ot + (kvh * G + g) * stride_oh + vofs)[:, None] + dvp[None, :]
            tl.store(out_ptr + row, asm.finish(out, l_run).to(out_ptr.dtype.element_ty), mask=live[:, None])
            if LSE:
                tl.store(lse_ptr + (start + c) * HQ + kvh * G + g, asm.lse(m_run, l_run, True), mask=live & (b == 0))
        else:
            lo = b * chunk
            acc, m_i, l_i = _flash_split(lo, tl.minimum(lo + chunk, n_keys), nfree, ffirst, q, qi, qs, lim, anc, first,
                                         count, kvh, k_ptr, v_ptr, ks_ptr, vs_ptr, tab_ptr, trow, tab_stride,
                                         tab_width, fk_ptr, fv_ptr, stride_fk, stride_fv, fshift, qk_scale, k_mult,
                                         v_mult, vofs, win, first + c_lo - win + 1, BLOCK_M, BLOCK_N, PAGE, HKV, D, DV, DVP, KIND, HW_FP8, TREE,
                                         FRESH, INT8_QK, SKIP)
            if OFFSETS:
                prow = (tl.load(off_ptr + r) - tl.load(off_ptr)).to(tl.int64)
            else:
                prow = r.to(tl.int64) * cols
            idx = ((prow + c) * HQ + kvh * G + g) * n_parts + b
            dvp = tl.arange(0, DVP)
            tl.store(po_ptr + idx[:, None] * DV + dvp[None, :], acc, mask=live[:, None])
            tl.store(pm_ptr + idx, m_i, mask=live)
            tl.store(pl_ptr + idx, l_i, mask=live)


@triton.jit(do_not_specialize=["first0"])
def _flash_merge_kernel(po_ptr, pm_ptr, pl_ptr, out_ptr, lse_ptr, start_ptr, count_ptr, first_ptr, off_ptr, sink_ptr,
                        n_parts, chunk, cols, first0, stride_ot, stride_oh, HQ: tl.constexpr, DV: tl.constexpr,
                        BLOCK_N: tl.constexpr, COUNTS: tl.constexpr, OFFSETS: tl.constexpr, LSE: tl.constexpr,
                        ONE_ROW: tl.constexpr, POS_T: tl.constexpr, DEV_CHUNK: tl.constexpr, CHUNK_P: tl.constexpr,
                        SINKS: tl.constexpr):
    """Program (column, query head, row): the partial states of the splits holding the column's positions folded in
    split order (attention_softmax.fold, base 2: a fused launch's rule), into the output row (and its log-sum-exp);
    CHUNK_P partials loaded a turn, each folded in order. SINKS: the head's sink logit folded in last."""
    c = tl.program_id(0)
    h = tl.program_id(1)
    r = tl.program_id(2)
    if ONE_ROW:
        count = cols
        if POS_T:
            first = tl.load(first_ptr).to(tl.int64)
        else:
            first = first0.to(tl.int64)
    else:
        if COUNTS:
            count = tl.minimum(tl.load(count_ptr + r), cols)
        else:
            count = cols
        first = tl.load(first_ptr + r).to(tl.int64)
    if c < count:
        if ONE_ROW:
            start = 0
        else:
            start = tl.load(start_ptr + r).to(tl.int64)
        chunk = _row_chunk(first, count, n_parts, chunk, BLOCK_N, DEV_CHUNK)
        nb = tl.minimum(n_parts, tl.cdiv(first + c + 1, chunk)).to(tl.int32)
        if OFFSETS:
            prow = (tl.load(off_ptr + r) - tl.load(off_ptr)).to(tl.int64)
        else:
            prow = r.to(tl.int64) * cols
        base = ((prow + c) * HQ + h) * n_parts
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
                o = tl.load(po_ptr + at * DV + dv, mask=use, other=0.0)
                m_new, l_new, keep, take = asm.fold(m_run, l_run, m, l, True)
                m_run = tl.where(use, m_new, m_run)
                l_run = tl.where(use, l_new, l_run)
                acc = tl.where(use, tl.fma(acc, keep, o * take), acc)
        if SINKS:
            m_run, l_run, keep = asm.sink(m_run, l_run, tl.load(sink_ptr + h), True)
            acc = acc * keep
        ok = l_run != 0   # 0 for a row with no position; a NaN sum (a NaN query) stays NaN
        o = tl.where(ok, acc / tl.where(ok, l_run, 1.0), 0.0)
        tl.store(out_ptr + (start + c) * stride_ot + h * stride_oh + dv, o.to(out_ptr.dtype.element_ty))
        if LSE:
            tl.store(lse_ptr + (start + c) * HQ + h, asm.lse(m_run, l_run, True))
