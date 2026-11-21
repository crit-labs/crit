"""Triton kernels of the sparse indexer: a token's index score over each compressed key, the heads' weighted ReLU sum
of their query-key products, then the top k keys of each token.

score_kernel: program (a block of TM tokens, a block of BN keys). Its tile rows are the tokens' heads (row j: token
j // H, head j % H), TM x H rows of D values times BN keys of D values on the tensor cores (FP32 sums from 16-bit
operands, or E4M3 keys converted exactly), then ReLU, times the head's factor (its weight times the scale), summed
over the heads in FP32 (an explicit tree: heads 2 i and 2 i + 1, then the pairs' sums) and written; keys at or past
the token's end get -inf. A token's score depends on its own query, weights and keys: never on the tokens beside it,
nor on TM, BN or the warps. With SLOTS the keys are read where a pool stores them: key j of the program's tokens is
row slots[group][j] of the store (their group: GROUP tokens share a sequence's slots, and a program's tokens are of
one group), so no gathered copy of a sequence's keys is made; the arithmetic and its bits are the gathered keys'.

topk_kernel: program (one token). The scores' order-preserving 32-bit keys (a NaN below every number, -inf included)
over the token's valid keys; four passes of 8-bit digit histograms find the k-th largest key and how many of its ties
to keep, then one pass writes the chosen keys' positions in ascending order (every key above the threshold, the
threshold's ties from the lowest position), -1 past the token's valid keys."""
import triton
import triton.language as tl

from crit_kcore import radix   # the keys: a NaN as 0, below every number


@triton.jit(do_not_specialize=["T", "N", "GROUP", "stride_ot", "stride_sg"],
            do_not_specialize_on_alignment=["end_ptr"])   # ends: a slice of a request's, any offset
def score_kernel(q_ptr, k_ptr, ks_ptr, w_ptr, end_ptr, out_ptr, slot_ptr, T, N, GROUP, scale, stride_qt, stride_qh,
                 stride_kn, stride_wt, stride_ot, stride_sg, H: tl.constexpr, D: tl.constexpr, TM: tl.constexpr,
                 BN: tl.constexpr, FP8: tl.constexpr, HW_FP8: tl.constexpr, LOG_H: tl.constexpr, SLOTS: tl.constexpr):
    tb = tl.program_id(0)
    nb = tl.program_id(1)
    rows = tb * TM * H + tl.arange(0, TM * H)        # (token, head) pairs
    tok = rows // H
    head = rows % H
    live = tok < T
    d = tl.arange(0, D)
    q = tl.load(q_ptr + tok[:, None].to(tl.int64) * stride_qt + head[:, None] * stride_qh + d[None, :],
                mask=live[:, None], other=0.0)
    n = nb * BN + tl.arange(0, BN)
    nm = n < N
    if SLOTS:   # the keys in place: the store's rows at the sequence's slots
        at = tl.load(slot_ptr + ((tb * TM) // GROUP).to(tl.int64) * stride_sg + n, mask=nm, other=0).to(tl.int64)
    else:
        at = n.to(tl.int64)
    kc = tl.load(k_ptr + at[:, None] * stride_kn + d[None, :], mask=nm[:, None], other=0)
    if FP8:
        if HW_FP8:
            kv = kc.to(tl.float8e4nv, bitcast=True).to(q.dtype)
        else:
            kv = (kc.to(tl.float8e4nv, bitcast=True).to(tl.float32)).to(q.dtype)
    else:
        kv = kc
    s = tl.dot(q, tl.trans(kv))                        # [TM * H, BN] FP32
    if FP8:
        s = s * tl.load(ks_ptr + at, mask=nm, other=0.0)[None, :]
    f = tl.load(w_ptr + tok.to(tl.int64) * stride_wt + head, mask=live, other=0.0) * scale
    s = tl.maximum(s, 0.0) * f[:, None]
    # The head sum as an explicit tree: heads 2 i and 2 i + 1 added, then the pairs' sums, and so on (tl.sum's tree
    # follows the tile's register layout, so TM, BN and the warps changed a token's bits; this order is the source's).
    tree = tl.permute(tl.reshape(s, (TM, H, BN)), (0, 2, 1))   # [TM, BN, H]
    for level in tl.static_range(LOG_H):
        left, right = tl.split(tl.reshape(tree, (TM, BN, H >> (level + 1), 2)))
        tree = left + right
    total = tl.reshape(tree, (TM, BN))
    t2 = tb * TM + tl.arange(0, TM)
    tm = t2 < T
    ends = tl.load(end_ptr + t2, mask=tm, other=0)
    total = tl.where(n[None, :] < ends[:, None], total, float("-inf"))
    tl.store(out_ptr + t2[:, None].to(tl.int64) * stride_ot + n[None, :], total, mask=tm[:, None] & nm[None, :])


@triton.jit(do_not_specialize=["N", "SLICE", "stride_st", "stride_ot"],
            do_not_specialize_on_alignment=["end_ptr"])   # ends: a slice of a request's, any offset
def topk_kernel(s_ptr, pos_ptr, end_ptr, out_ptr, sout_ptr, N, K, SLICE, stride_st, stride_ot, BLOCK: tl.constexpr,
                KB: tl.constexpr, POS: tl.constexpr, ENDS: tl.constexpr, SOUT: tl.constexpr):
    """Program (token t, slice p): the k best of the token's valid scores in positions p x SLICE .. (p + 1) x SLICE
    - 1, written at out[t][p x K ..] (ascending, then -1). POS: the scores are candidates whose positions pos_ptr holds
    ([tokens][N], -1 for none): a candidate's position is written, not its index, and its order among equal scores is
    its index's (candidates come in position order). SOUT: the chosen scores too, beside their positions."""
    t = tl.program_id(0)
    p = tl.program_id(1)
    end = N
    if ENDS:
        end = tl.maximum(tl.minimum(tl.load(end_ptr + t), N), 0)
    lo = p * SLICE
    hi = tl.minimum(end, lo + SLICE)
    row = s_ptr + t.to(tl.int64) * stride_st
    prow = pos_ptr + t.to(tl.int64) * stride_st
    orow = out_ptr + t.to(tl.int64) * stride_ot + p * K
    srow = sout_ptr + t.to(tl.int64) * stride_ot + p * K
    need = K
    prefix = tl.zeros((), tl.uint32)
    bins = tl.arange(0, 512)
    # Four 8-bit passes, the highest digit first: the k-th largest key's digit among the keys matching the prefix.
    for d in tl.static_range(4):
        shift = 24 - 8 * d
        hist = tl.zeros((512,), tl.int32)
        for n0 in range(lo, hi, BLOCK):
            n = n0 + tl.arange(0, BLOCK)
            nm = n < hi
            if POS:
                nm = nm & (tl.load(prow + n, mask=nm, other=-1) >= 0)
            key = radix.radix_key(tl.load(row + n, mask=nm, other=0.0))
            if d == 0:
                match = nm
            else:
                match = nm & ((key >> (shift + 8)) == (prefix >> (shift + 8)))
            digit = tl.where(match, (key >> shift) & 0xFF, 256).to(tl.int32)
            hist += tl.histogram(digit, 512)
        hist = tl.where(bins < 256, hist, 0)
        total = tl.sum(hist, 0)
        if d == 0:
            need = tl.minimum(need, total)                 # every valid key where there are k or fewer
        ge = total - tl.cumsum(hist, 0) + hist           # keys of this digit or above
        pick = tl.sum(((ge >= need) & (bins < 256)).to(tl.int32), 0) - 1
        pick = tl.maximum(pick, 0)
        need = need - tl.sum(tl.where(bins > pick, hist, 0), 0)
        prefix = prefix | (pick.to(tl.uint32) << shift)
    # The threshold key `prefix`: every key above it and `need` of its ties (lowest positions first), ascending.
    written = tl.zeros((), tl.int32)
    ties = tl.zeros((), tl.int32)
    for n0 in range(lo, hi, BLOCK):
        n = n0 + tl.arange(0, BLOCK)
        nm = n < hi
        where = n
        if POS:
            where = tl.load(prow + n, mask=nm, other=-1)
            nm = nm & (where >= 0)
        x = tl.load(row + n, mask=nm, other=0.0)
        key = radix.radix_key(x)
        tie = nm & (key == prefix)
        tie_rank = ties + tl.cumsum(tie.to(tl.int32), 0) - 1
        take = nm & ((key > prefix) | (tie & (tie_rank < need)))
        at = written + tl.cumsum(take.to(tl.int32), 0) - 1
        tl.store(orow + at, where, mask=take)
        if SOUT:
            tl.store(srow + at, x, mask=take)
        written += tl.sum(take.to(tl.int32), 0)
        ties += tl.sum(tie.to(tl.int32), 0)
    for k0 in range(0, K, KB):
        j = k0 + tl.arange(0, KB)
        tl.store(orow + j, tl.full((KB,), -1, tl.int32), mask=(j >= written) & (j < K))
