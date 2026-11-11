"""Against a float64 token-by-token reference: the chunked prefills (FP32 and on tensor cores), the scans, column
invariance and checkpoints."""
import pytest
import torch

from crit_kernels import build as kbuild

rs = kbuild.load("recurrent-scans")   # built for this GPU with its default knobs (cache: CRIT_KERNEL_CACHE)
dev = torch.device("cuda")


def delta_reference(q, k, v, g, b, state, scale):
    q, k, v, g, b, S = (t.double() for t in (q, k, v, g, b, state))
    T, kh, dk = q.shape
    vh = v.shape[1]
    out = torch.empty(T, vh, v.shape[2], dtype=torch.float64, device=q.device)
    for t in range(T):
        for h in range(vh):
            j = h // (vh // kh)
            e = torch.exp(g[t, h])
            delta = b[t, h] * (v[t, h] - e * S[h].T @ k[t, j])
            S[h] = e * S[h] + torch.outer(k[t, j], delta)
            out[t, h] = scale * S[h].T @ q[t, j]
    return out, S


def delta_inputs(T, kh=2, vh=4, dk=64, dv=64, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(T, kh, dk, generator=g), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(T, kh, dk, generator=g), dim=-1)
    v = torch.randn(T, vh, dv, generator=g)
    logd = -torch.rand(T, vh, generator=g) * 0.2
    beta = torch.rand(T, vh, generator=g)
    s0 = torch.randn(vh, dk, dv, generator=g) * 0.1
    return [t.to(dev) for t in (q, k, v, logd, beta, s0)]


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dk", [64, 128])
@pytest.mark.parametrize("T", [1, 37, 200])
def test_gated_delta_prefill_and_scan_match_the_recurrence(dk, T):
    q, k, v, logd, beta, s0 = delta_inputs(T, dk=dk, dv=128)
    ref_out, ref_state = delta_reference(q, k, v, logd, beta, s0.clone(), 0.5)
    for fn in (rs.gated_delta_prefill, rs.gated_delta_scan):
        state = s0.clone()
        out = fn(q, k, v, logd, beta, state, scale=0.5)
        torch.testing.assert_close(out.double(), ref_out, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(state.double(), ref_state, rtol=1e-4, atol=1e-4)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dk,dv,kh,vh", [(128, 128, 2, 6), (64, 48, 1, 3), (128, 16, 3, 3)])
@pytest.mark.parametrize("T", [1, 37, 200, 700])
def test_gated_delta_prefill_on_tensor_cores_matches_the_recurrence(dk, dv, kh, vh, T):
    """delta-prefill-tc (FP16 operands, FP32 accumulation and state) within 1e-3 + 1e-3|x| of the float64 recurrence
    (the bench's tolerance) at the plugins' query scale, over partial chunks, an odd number of value heads a key head
    and value columns that leave a block's warps idle; empty prompts leave the state."""
    q, k, v, logd, beta, s0 = delta_inputs(T, kh=kh, vh=vh, dk=dk, dv=dv, seed=T + dv)
    ref_out, ref_state = delta_reference(q, k, v, logd, beta, s0.clone(), dk ** -0.5)
    state = s0.clone()
    out = rs.gated_delta_prefill(q, k, v, logd, beta, state, scale=dk ** -0.5, tensor_cores=True)
    torch.testing.assert_close(out.double(), ref_out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(state.double(), ref_state, rtol=1e-3, atol=1e-3)
    state = s0.clone()
    rs.gated_delta_prefill(q[:0], k[:0], v[:0], logd[:0], beta[:0], state, tensor_cores=True)
    assert torch.equal(state, s0)


@pytest.mark.kernels_ci
def test_gated_delta_scan_is_column_invariant_and_checkpoints_each_token():
    q, k, v, logd, beta, s0 = delta_inputs(6)
    state, cps = s0.clone(), torch.empty(6, *s0.shape, device=dev)
    together = rs.gated_delta_scan(q, k, v, logd, beta, state, checkpoints=cps)
    alone, s = [], s0.clone()
    for t in range(6):
        alone.append(rs.gated_delta_scan(q[t:t + 1], k[t:t + 1], v[t:t + 1], logd[t:t + 1], beta[t:t + 1], s))
        assert torch.equal(cps[t], s)
    assert torch.equal(torch.cat(alone), together)


def ssd_reference(x, dt, B, C, A, D, state):
    x, dt, B, C, A, D, H = (t.double() for t in (x, dt, B, C, A, D, state))
    T, heads, p = x.shape
    groups = B.shape[1]
    out = torch.empty_like(x)
    for t in range(T):
        for h in range(heads):
            gi = h // (heads // groups)
            H[h] = torch.exp(dt[t, h] * A[h]) * H[h] + dt[t, h] * torch.outer(x[t, h], B[t, gi])
            out[t, h] = H[h] @ C[t, gi] + D[h] * x[t, h]
    return out, H


@pytest.mark.kernels_ci
@pytest.mark.parametrize("n", [64, 128])
def test_ssd_prefill_and_scan_match_the_recurrence(n):
    g = torch.Generator(device="cpu").manual_seed(1)
    T, heads, groups, p = 70, 4, 2, 64
    x = torch.randn(T, heads, p, generator=g).to(dev)
    dt = (torch.rand(T, heads, generator=g) * 0.1).to(dev)
    B = torch.randn(T, groups, n, generator=g).to(dev)
    C = torch.randn(T, groups, n, generator=g).to(dev)
    A = (-torch.rand(heads, generator=g) - 0.1).to(dev)
    D = torch.randn(heads, generator=g).to(dev)
    s0 = (torch.randn(heads, p, n, generator=g) * 0.1).to(dev)
    ref_out, ref_state = ssd_reference(x, dt, B, C, A, D, s0.clone())
    for fn in (rs.ssd_prefill, rs.ssd_scan):
        state = s0.clone()
        out = fn(x, dt, B, C, A, state, D=D)
        torch.testing.assert_close(out.double(), ref_out, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(state.double(), ref_state, rtol=1e-4, atol=1e-4)


@pytest.mark.kernels_ci
def test_causal_conv1d_carries_its_window():
    g = torch.Generator(device="cpu").manual_seed(2)
    T, ch, w = 50, 96, 4
    x = torch.randn(T, ch, generator=g).to(dev)
    weight = torch.randn(ch, w, generator=g).to(dev)
    bias = torch.randn(ch, generator=g).to(dev)
    s0 = torch.randn(ch, w - 1, generator=g).to(dev)
    padded = torch.cat([s0.T, x]).double()
    ref = torch.stack([(padded[t:t + w].T * weight.double()).sum(1) + bias.double() for t in range(T)])
    state = s0.clone()
    out = rs.causal_conv1d(x, weight, state, bias=bias, silu=True)
    torch.testing.assert_close(out.double(), torch.nn.functional.silu(ref), rtol=1e-5, atol=1e-5)
    assert torch.equal(state, x[-(w - 1):].T.contiguous())
