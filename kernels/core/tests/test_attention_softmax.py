"""crit_kcore.attention_softmax on the device: an online softmax over blocks (step, row_sum) and partial states folded
in order (fold, finish, lse) equal float64 softmax attention, in base 2 and in natural exponents, rows that see
nothing give zeros and -inf, and the tensor-core row sum's bits do not follow the tile's rows."""
import math

import pytest
import torch
import triton
import triton.language as tl

from crit_kcore import attention_softmax as asm

dev = torch.device("cuda")


@triton.jit
def _attend(s_ptr, v_ptr, out_ptr, lse_ptr, n, split, M: tl.constexpr, N: tl.constexpr, DV: tl.constexpr,
            BASE2: tl.constexpr, FIXED: tl.constexpr):
    """Rows of scores s [M][n] (-inf: not seen) over values v [n][DV]: blocks of N positions in an online softmax,
    the positions before `split` and from it on as two partial states folded in order."""
    rows = tl.arange(0, M)
    dv = tl.arange(0, DV)
    ones = tl.full([N, 16], 1.0, tl.bfloat16)
    m_run = tl.full([M], float("-inf"), tl.float32)
    l_run = tl.zeros([M], tl.float32)
    out = tl.zeros([M, DV], tl.float32)
    for part in tl.static_range(2):
        lo = 0 if part == 0 else split
        hi = split if part == 0 else n
        m_i = tl.full([M], float("-inf"), tl.float32)
        l_i = tl.zeros([M], tl.float32)
        acc = tl.zeros([M, DV], tl.float32)
        for n0 in range(lo, hi, N):
            c = n0 + tl.arange(0, N)
            cm = c < hi
            s = tl.load(s_ptr + rows[:, None] * n + c[None, :], mask=cm[None, :], other=float("-inf"))
            if BASE2:
                s = s * 1.4426950408889634     # asm.LOG2E: base-2 scores
            p, alpha, m_new = asm.step(s, m_i, BASE2)
            v = tl.load(v_ptr + c[:, None] * DV + dv[None, :], mask=cm[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            l_i = l_i * alpha + asm.row_sum(p, ones, FIXED)
            m_i = m_new
        m_run, l_run, keep, take = asm.fold(m_run, l_run, m_i, l_i, BASE2)
        out = tl.fma(out, keep[:, None], acc * take[:, None])
    tl.store(out_ptr + rows[:, None] * DV + dv[None, :], asm.finish(out, l_run))
    tl.store(lse_ptr + rows, asm.lse(m_run, l_run, BASE2))


def _run(s, v, split, base2, fixed, block=32):
    m, n = s.shape
    out = torch.empty(m, v.shape[1], device=dev)
    lse = torch.empty(m, device=dev)
    _attend[(1,)](s, v, out, lse, n, split, M=m, N=block, DV=v.shape[1], BASE2=base2, FIXED=fixed)
    return out, lse


@pytest.mark.kernels_ci
@pytest.mark.parametrize("base2", [True, False])
@pytest.mark.parametrize("fixed", [True, False])
def test_an_online_softmax_with_folded_states_is_softmax_attention(base2, fixed):
    g = torch.Generator().manual_seed(1)
    s = (torch.randn(32, 300, generator=g) * 4).to(dev)
    s[3, 200:] = float("-inf")       # a causal edge
    s[5] = float("-inf")             # a row that sees nothing
    s[7, :150] = float("-inf")       # a row whose first partial state is empty
    v = torch.randn(300, 64, generator=g).to(dev).to(torch.bfloat16)
    out, lse = _run(s.contiguous(), v.contiguous(), 128, base2, fixed)
    seen = torch.isfinite(s).any(1)
    probs = torch.softmax(s.double(), 1).nan_to_num(0.0)
    want = probs @ v.double()
    torch.testing.assert_close(out[seen].double(), want[seen], atol=8e-3, rtol=2e-2)
    # (the kernels' log-sum-exp bound: exp2 and log run on the GPU's approximate units)
    torch.testing.assert_close(lse[seen].double(), torch.logsumexp(s.double(), 1)[seen], atol=1e-3, rtol=1e-4)
    assert bool((out[5] == 0).all()) and math.isinf(float(lse[5])) and float(lse[5]) < 0


@pytest.mark.kernels_ci
def test_the_tensor_core_row_sum_keeps_a_rows_bits_whatever_the_tile():
    g = torch.Generator().manual_seed(2)
    s = torch.randn(64, 200, generator=g).to(dev)
    v = torch.randn(200, 32, generator=g).to(dev).to(torch.bfloat16)
    wide, wide_lse = _run(s.contiguous(), v, 96, True, True)
    narrow, narrow_lse = _run(s[:16].contiguous(), v, 96, True, True)
    assert torch.equal(wide[:16], narrow) and torch.equal(wide_lse[:16], narrow_lse)
