"""Attention sinks: each query head's sink logit joins every row's softmax as one more position with no value. Every
structure (the split launch and merge, fused, flash, flash fused, the CUDA tile split and fused, and the three prompt
chunk paths) against a float64 reference with the sink as an extra column, its log-sum-exp counting the sink; a fused
launch keeps its split launch's bits with sinks; sinks of -inf keep every bit of a launch without them."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_paged_kv_attention import Cache, _q, dev, pk  # noqa: E402

NEG = float("-inf")
FAMILIES = [{}, {"flash": True}, {"flash": "cuda"}]


def reference(q, keys, values, seen, scale, sink):
    """q [hq][d] float64 over keys [n][hkv][d] where seen [n], each head's sink one more score -> (out, lse)."""
    hq, hkv = q.shape[0], keys.shape[1]
    out, lse = [], []
    for h in range(hq):
        k, v = keys[:, h // (hq // hkv)], values[:, h // (hq // hkv)]
        s = torch.where(seen, (k @ q[h]) * scale, torch.tensor(NEG, dtype=torch.float64, device=dev))
        s = torch.cat([s, torch.tensor([float(sink[h])], dtype=torch.float64, device=dev)])
        p = torch.softmax(s, 0)
        out.append(p[:-1] @ v)
        lse.append(torch.logsumexp(s, 0))
    return torch.stack(out), torch.stack(lse)


def sinks_for(hq, seed=7):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(hq, generator=g) * 2.0
    s[1] = 8.0      # a sink above every score: most of the row's weight
    s[2] = NEG      # a head without one
    return s.to(dev)


@pytest.mark.parametrize("family", FAMILIES, ids=["split", "flash", "cuda"])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("kind", ["bf16", "int8"])
def test_windows_with_sinks_match_float64_attention(family, fused, kind):
    hq, hkv, d = 8, 2, 128
    cache = Cache(kind, hkv, d, d, 16, [130, 64, 1, 517])
    rows, starts, firsts, counts = [0, 1, 2, 3], [0, 5, 10, 15], [125, 59, 0, 400], [5, 5, 1, 3]
    q = _q(20, hq, d, 3)
    sinks = sinks_for(hq)
    lse = torch.empty(20, hq, device=dev)
    out = cache.attend(q, rows, starts, firsts, 5, lse=lse, sinks=sinks, fused=fused,
                       counts=torch.tensor(counts, dtype=torch.int32, device=dev), **family)
    qd = q.to(torch.bfloat16).double()
    for s, start, first, count in zip(rows, starts, firsts, counts):
        keys, values = cache.rows(s, first + count)
        for c in range(count):
            seen = torch.arange(first + count, device=dev) <= first + c
            want, want_lse = reference(qd[start + c], keys, values, seen, d ** -0.5, sinks)
            torch.testing.assert_close(out[start + c].double(), want, atol=8e-3, rtol=2e-2)
            torch.testing.assert_close(lse[start + c].double(), want_lse, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("family", FAMILIES, ids=["split", "flash", "cuda"])
def test_a_fused_launch_keeps_the_split_launchs_bits_with_sinks_and_no_sink_is_no_change(family):
    hq, hkv, d = 8, 2, 128
    for length in (40, 700, 5000):
        cache = Cache("bf16", hkv, d, d, 64, [length], seed=1)
        T = min(130, length)
        pos0 = length - T
        q = _q(T, hq, d, 2)
        starts = list(range(0, T, 64))
        counts = torch.tensor([min(64, T - s) for s in starts], dtype=torch.int32, device=dev)
        sinks = sinks_for(hq)
        window = dict(counts=counts, **family)
        args = ([0] * len(starts), starts, [pos0 + s for s in starts], 64)
        lse, split_lse = torch.empty(T, hq, device=dev), torch.empty(T, hq, device=dev)
        fused = cache.attend(q, *args, fused=True, lse=lse, sinks=sinks, **window)
        split = cache.attend(q, *args, lse=split_lse, sinks=sinks, **window)
        assert torch.equal(fused, split) and torch.equal(lse, split_lse), length
        none = torch.full((hq,), NEG, device=dev)
        for f in (False, True):
            assert torch.equal(cache.attend(q, *args, fused=f, sinks=none, **window),
                               cache.attend(q, *args, fused=f, **window)), (length, f)


@pytest.mark.parametrize("flash", [False, True, "cuda"])
def test_prompt_chunks_with_sinks_match_float64_attention(flash):
    cache = Cache("bf16", 2, 128, 128, 16, [300])
    q = _q(100, 8, 128, 4)
    sinks = sinks_for(8)
    lse = torch.empty(100, 8, device=dev)
    out = pk.prefill(q.to(torch.bfloat16), cache.k, cache.v, "bf16", cache.tables[0], 200, 16, lse=lse, sinks=sinks,
                     flash=flash)
    keys, values = cache.rows(0, 300)
    qd = q.to(torch.bfloat16).double()
    for t in (0, 37, 99):
        seen = torch.arange(300, device=dev) <= 200 + t
        want, want_lse = reference(qd[t], keys, values, seen, 128 ** -0.5, sinks)
        torch.testing.assert_close(out[t].double(), want, atol=8e-3, rtol=2e-2)
        torch.testing.assert_close(lse[t].double(), want_lse, atol=1e-3, rtol=1e-4)
    none = torch.full((8,), NEG, device=dev)
    plain = pk.prefill(q.to(torch.bfloat16), cache.k, cache.v, "bf16", cache.tables[0], 200, 16, flash=flash)
    assert torch.equal(pk.prefill(q.to(torch.bfloat16), cache.k, cache.v, "bf16", cache.tables[0], 200, 16,
                                  sinks=none, flash=flash), plain)


def test_sinks_other_than_fp32_a_head_are_refused():
    cache = Cache("bf16", 2, 64, 64, 16, [20])
    q = _q(1, 4, 64, 1)
    for bad in (torch.zeros(4, device=dev, dtype=torch.bfloat16), torch.zeros(3, device=dev), torch.zeros(4)):
        with pytest.raises(ValueError, match="sinks are contiguous FP32"):
            cache.attend(q, [0], [0], [10], 1, sinks=bad)
