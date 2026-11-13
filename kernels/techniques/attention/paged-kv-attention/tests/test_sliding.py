"""Sliding windows: with sliding=W a token at position t sees positions t - W + 1 .. t. Every structure (the split
launch and merge, fused, flash, flash fused, the CUDA tile split and fused, and the three prompt-chunk paths) against a
float64 reference with that mask, with and without sinks; a fused launch keeps its split launch's bits; a window at
least the row's length keeps every bit of a launch without one (the flash prompt chunk within rounding: a window
masks every block); a long cache under a short window reads only the window's blocks."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_paged_kv_attention import Cache, _q, dev, pk  # noqa: E402
from test_sinks import reference as sink_reference  # noqa: E402

FAMILIES = [{}, {"flash": True}, {"flash": "cuda"}]
IDS = ["split", "flash", "cuda"]


def reference(q, keys, values, position, window, scale, sinks=None):
    n = keys.shape[0]
    at = torch.arange(n, device=dev)
    seen = (at <= position) & (at > position - window if window else True)
    if sinks is None:
        sinks = torch.full((q.shape[0],), float("-inf"), device=dev)
    return sink_reference(q, keys, values, seen, scale, sinks)


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("kind", ["bf16", "int8"])
def test_windows_with_a_sliding_window_match_float64_attention(family, fused, kind):
    hq, hkv, d = 8, 2, 128
    cache = Cache(kind, hkv, d, d, 16, [130, 64, 1, 2517])
    rows, starts, firsts, counts = [0, 1, 2, 3], [0, 5, 10, 15], [125, 59, 0, 2400], [5, 5, 1, 3]
    q = _q(20, hq, d, 3)
    sinks = torch.randn(hq, generator=torch.Generator().manual_seed(4)).to(dev)
    qd = q.to(torch.bfloat16).double()
    for window, s in ((7, None), (100, sinks), (1, None)):
        lse = torch.empty(20, hq, device=dev)
        out = cache.attend(q, rows, starts, firsts, 5, lse=lse, sliding=window, sinks=s, fused=fused,
                           counts=torch.tensor(counts, dtype=torch.int32, device=dev), **family)
        for r, start, first, count in zip(rows, starts, firsts, counts):
            keys, values = cache.rows(r, first + count)
            for c in range(count):
                want, want_lse = reference(qd[start + c], keys, values, first + c, window, d ** -0.5, s)
                torch.testing.assert_close(out[start + c].double(), want, atol=8e-3, rtol=2e-2)
                # The split reader sums a row's BF16 probabilities on the tensor cores: over a few positions their
                # rounding does not average out (up to about 2^-9 of the sum).
                torch.testing.assert_close(lse[start + c].double(), want_lse, atol=2e-3, rtol=1e-4)


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_a_fused_launch_keeps_the_split_launchs_bits_and_a_wide_window_is_none(family):
    hq, hkv, d = 8, 2, 128
    for length in (40, 700, 5000):
        cache = Cache("bf16", hkv, d, d, 64, [length], seed=1)
        T = min(130, length)
        pos0 = length - T
        q = _q(T, hq, d, 2)
        starts = list(range(0, T, 64))
        counts = torch.tensor([min(64, T - s) for s in starts], dtype=torch.int32, device=dev)
        args = ([0] * len(starts), starts, [pos0 + s for s in starts], 64)
        for window in (33, 500):
            fused = cache.attend(q, *args, counts=counts, fused=True, sliding=window, **family)
            split = cache.attend(q, *args, counts=counts, sliding=window, **family)
            assert torch.equal(fused, split), (length, window)
        for f in (False, True):
            assert torch.equal(cache.attend(q, *args, counts=counts, fused=f, sliding=length + 1, **family),
                               cache.attend(q, *args, counts=counts, fused=f, **family)), (length, f)


@pytest.mark.parametrize("flash", [False, True, "cuda"])
def test_prompt_chunks_with_a_sliding_window_match_float64_attention(flash):
    cache = Cache("bf16", 2, 128, 128, 16, [700])
    q = _q(100, 8, 128, 4).to(torch.bfloat16)
    keys, values = cache.rows(0, 700)
    for window in (1, 64, 300):
        lse = torch.empty(100, 8, device=dev)
        out = pk.prefill(q, cache.k, cache.v, "bf16", cache.tables[0], 600, 16, lse=lse, sliding=window, flash=flash)
        for t in (0, 37, 99):
            want, want_lse = reference(q[t].double(), keys, values, 600 + t, window, 128 ** -0.5)
            torch.testing.assert_close(out[t].double(), want, atol=8e-3, rtol=2e-2)
            torch.testing.assert_close(lse[t].double(), want_lse, atol=1e-3, rtol=1e-4)
    plain = pk.prefill(q, cache.k, cache.v, "bf16", cache.tables[0], 600, 16, flash=flash)
    wide = pk.prefill(q, cache.k, cache.v, "bf16", cache.tables[0], 600, 16, sliding=701, flash=flash)
    if flash is True:   # a window masks every block (no unmasked fast path), so its rounding differs
        torch.testing.assert_close(wide.float(), plain.float(), atol=8e-3, rtol=2e-2)
    else:
        assert torch.equal(wide, plain)


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_a_long_cache_under_a_short_window_reads_only_the_window(family):
    cache = Cache("bf16", 2, 128, 128, 64, [60000], seed=5)
    q = _q(1, 8, 128, 6).to(torch.bfloat16)
    zero = torch.zeros(1, dtype=torch.int32, device=dev)
    first = torch.tensor([59999], dtype=torch.int64, device=dev)
    out = torch.empty(1, 8, 128, dtype=torch.bfloat16, device=dev)

    def timed(**kw):   # ten launches in a CUDA graph: the kernels' time, not the host's
        def launch():
            pk.attend(q, cache.k, cache.v, "bf16", cache.tables, zero, zero, first, 1, 64, out=out, **kw, **family)
        launch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(10):
                launch()
        graph.replay()
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(5):   # the best of five replays: a GPU shared with other work times high at times
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            graph.replay()
            b.record()
            torch.cuda.synchronize()
            best = min(best, a.elapsed_time(b))
        return best
    assert timed(sliding=128) < 0.5 * timed()


def test_a_sliding_window_with_block_lists_is_refused():
    cache = Cache("bf16", 2, 64, 64, 16, [40])
    q = _q(1, 4, 64, 1)
    blocks = torch.tensor([[0, 1]], dtype=torch.int32, device=dev)
    with pytest.raises(ValueError, match="without block lists"):
        cache.attend(q, [0], [0], [30], 1, blocks=blocks, block_size=16, sliding=8)
    with pytest.raises(ValueError, match="0 \\(none\\) or a number"):
        cache.attend(q, [0], [0], [30], 1, sliding=-1)
