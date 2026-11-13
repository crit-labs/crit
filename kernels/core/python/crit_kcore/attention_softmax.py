"""Online softmax on tensor-core tiles, for Triton kernels: the steps a flash-style attention loop is made of, with no
attention algorithm in them (which positions a program reads, in what order, and where its state goes are the
kernel's).

A program holds, per query row, a running maximum `m`, a running sum `l` and an unnormalized output `acc`; each block
of scores updates them (step, row_sum), partial states from different key ranges combine in a fixed order (fold) and
the output is normalized at the end (finish, lse).

BASE2 (a constexpr): scores already multiplied by log2(e) (folded into the softmax scale, one multiply per score
saved) take exp2 and keep `m` in base-2 units; lse() turns them back to natural log. BASE2 False: natural exponents,
tl.exp as written (the bits of kernels that predate the core).

Row sums: row_sum(p, ones, True) sums each row of a block on the tensor cores (a dot with a column of ones), so its
summation order is the MMA's k-order, the same whatever the tile's rows, warps and layout (a row's bits then do not
depend on the rows beside it); False sums with tl.sum in FP32 (the layout's reduction tree: fastest, the order may
follow the tile).

Mechanisms after FlashAttention-2 and FlashInfer (Apache-2.0): exp2 with the scale folded in, a -inf-safe maximum,
state merges by a running maximum.
"""
import triton
import triton.language as tl

LOG2E = 1.4426950408889634   # log2(e): natural-log scores times this are base-2 scores
LN2 = 0.6931471805599453     # ln(2): base-2 log-sum-exps times this are natural ones


@triton.jit
def _exp(x, BASE2: tl.constexpr):
    if BASE2:
        y = tl.math.exp2(x)
    else:
        y = tl.exp(x)
    return y


@triton.jit
def step(s, m_i, BASE2: tl.constexpr):
    """One block of scores s [rows, positions] (masked ones -inf) against the running maximum m_i [rows] -> (p, alpha,
    m_new): the block's probabilities relative to the new maximum, the factor the running sum and output take, and
    the new maximum. A row with no position seen yet keeps a maximum of -inf and gets p = 0, alpha = 0 (no NaN)."""
    m_new = tl.maximum(m_i, tl.max(s, axis=1))
    m_sub = tl.where(m_new == float("-inf"), 0.0, m_new)
    p = _exp(s - m_sub[:, None], BASE2)
    alpha = _exp(m_i - m_sub, BASE2)
    return p, alpha, m_new


@triton.jit
def row_sum(p, ones, FIXED: tl.constexpr):
    """Each row's sum of p [rows, positions]: with FIXED on the tensor cores (p in ones' dtype times ones [positions,
    16]: the MMA's k-order, whatever the tile's layout), else tl.sum in FP32."""
    if FIXED:
        total = tl.max(tl.dot(p.to(ones.dtype), ones), axis=1)
    else:
        total = tl.sum(p, axis=1)
    return total


@triton.jit
def fold(m_run, l_run, m, l, BASE2: tl.constexpr):
    """A partial state's (maximum m, sum l) into the running (m_run, l_run), in the caller's order -> (maximum, sum,
    the running output's factor, the partial output's factor): output = output * keep + partial * take (with an FMA,
    as the sum). Folding the same partials in the same order gives the same bits wherever it runs."""
    m_new = tl.maximum(m_run, m)
    m_sub = tl.where(m_new == float("-inf"), 0.0, m_new)
    keep = _exp(m_run - m_sub, BASE2)
    take = _exp(m - m_sub, BASE2)
    return m_new, tl.fma(l_run, keep, l * take), keep, take


@triton.jit
def sink(m_run, l_run, s, BASE2: tl.constexpr):
    """A sink logit s (natural log, in the scaled scores' units; -inf: none) into a row's final state as one more
    position that carries no value -> (maximum, sum, the output's factor): the output is acc * keep, then finish(),
    and lse() of the returned state counts the sink. Folded once, after every partial state, so a split launch's merge
    and a fused launch take the same steps."""
    if BASE2:
        s = s * 1.4426950408889634
    m_new, l_new, keep, take = fold(m_run, l_run, s, 1.0, BASE2)
    return m_new, l_new, keep


@triton.jit
def finish(acc, l):
    """The normalized output acc / l per row, 0 where a row saw no position (l = 0); a NaN sum (a NaN score) stays
    NaN."""
    ok = l != 0
    return tl.where(ok[:, None], acc / tl.where(ok, l, 1.0)[:, None], 0.0)


@triton.jit
def lse(m, l, BASE2: tl.constexpr):
    """Natural-log log-sum-exp of a row's scores from its maximum and sum (-inf where it saw no position). Base 2: one
    FMA, written out (the compiler may contract a multiply and an add or not, by the code around them: an explicit one
    gives every caller the same bits)."""
    if BASE2:
        found = tl.fma(m, 0.6931471805599453, tl.log(l))
    else:
        found = m + tl.log(l)
    return tl.where(l != 0, found, float("-inf"))
