"""latent-attention against attention-state's float64 reference (latent_attention.py) over the cached rows: decode
tokens and verify windows over shuffled pages, ranks and heads of the models that use it, sinks and log-sum-exps; a
token's bits the same alone, in a verify window and beside other rows; CUDA graph replay at other lengths; what it
refuses."""
import importlib.util
from pathlib import Path

import pytest
import torch
import triton

from crit_kernels.build import load

la = load("latent-attention")
dev = torch.device("cuda")
_spec = importlib.util.spec_from_file_location(
    "latent_reference", Path(__file__).resolve().parents[5] / "inference" / "attention-state" / "code" /
    "latent_attention.py")
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)


class Cache:
    """Sequences of compressed rows (latent R, rotary key P) over shuffled pages."""

    def __init__(self, rank, rope, page, lengths, seed=0, dtype=torch.bfloat16):
        g = torch.Generator().manual_seed(seed)
        self.rank, self.rope, self.page = rank, rope, page
        width = max(-(-n // page) for n in lengths) + 1
        n_pages = width * len(lengths) + 2
        self.cache = torch.zeros(n_pages, page, rank + rope, dtype=dtype, device=dev)
        order = torch.randperm(n_pages, generator=g)
        self.tables = order[:width * len(lengths)].view(len(lengths), width).to(torch.int32).to(dev)
        for s, n in enumerate(lengths):
            latent = torch.randn(n, rank, generator=g).to(dev)
            rope_k = torch.randn(n, rope, generator=g).to(dev)
            la.write(latent, rope_k, self.slots(s, torch.arange(n, device=dev)), self.cache)

    def slots(self, s, positions):
        return self.tables[s][positions // self.page].long() * self.page + positions % self.page

    def rows(self, s, n):
        found = self.cache.view(-1, self.rank + self.rope)[self.slots(s, torch.arange(n, device=dev))].double()
        return found[:, :self.rank], found[:, self.rank:]

    def attend(self, q, rows, starts, firsts, columns, **kw):
        return la.attend(q, self.cache, self.tables, torch.tensor(rows, dtype=torch.int32, device=dev),
                         torch.tensor(starts, dtype=torch.int32, device=dev),
                         torch.tensor(firsts, dtype=torch.int64, device=dev), columns, self.page, rank=self.rank, **kw)


def _q(tokens, heads, width, seed, dtype=torch.bfloat16):
    return (torch.randn(tokens, heads, width, generator=torch.Generator().manual_seed(seed)) * 0.5).to(dev, dtype)


@pytest.mark.parametrize("rank,rope,heads,page", [(512, 64, 16, 64), (512, 64, 128, 16), (256, 64, 8, 32),
                                                 (128, 32, 20, 16)])
def test_windows_match_float64_latent_attention(rank, rope, heads, page):
    cache = Cache(rank, rope, page, [130, 64, 1, 517])
    rows, starts, firsts, counts = [0, 1, 2, 3], [0, 5, 10, 15], [125, 59, 0, 400], [5, 5, 1, 3]
    q = _q(20, heads, rank + rope, 3)
    scale = 192 ** -0.5
    sinks = torch.randn(heads, generator=torch.Generator().manual_seed(9)).to(dev)
    sinks[0] = float("-inf")
    for with_sinks in (False, True):
        lse = torch.empty(20, heads, device=dev)
        out = cache.attend(q, rows, starts, firsts, 5, scale=scale, lse=lse,
                           counts=torch.tensor(counts, dtype=torch.int32, device=dev),
                           sinks=sinks if with_sinks else None)
        for s, start, first, count in zip(rows, starts, firsts, counts):
            latent, rope_k = cache.rows(s, first + count)
            seen = torch.arange(first + count, device=dev)[None, :] <= first + torch.arange(count, device=dev)[:, None]
            span = slice(start, start + count)
            want, want_lse = ref.latent_attention(q[span, :, :rank], q[span, :, rank:], latent, rope_k, seen, scale,
                                                  sinks if with_sinks else None)
            torch.testing.assert_close(out[span].double(), want, atol=8e-3, rtol=2e-2)
            torch.testing.assert_close(lse[span].double(), want_lse, atol=1e-3, rtol=1e-4)


def test_a_tokens_bits_are_its_own_alone_in_a_window_and_beside_other_rows():
    cache = Cache(512, 64, 64, [700, 300, 2100], seed=2)
    q = _q(24, 16, 576, 5)
    together = cache.attend(q, [0, 1, 2], [0, 8, 16], [690, 290, 2090], 8)
    for r, (start, first) in enumerate(zip([0, 8, 16], [690, 290, 2090])):
        for c in range(8):
            alone = cache.attend(q[start + c:start + c + 1], [r], [0], [first + c], 1)
            assert torch.equal(alone[0], together[start + c]), (r, c)
        window = cache.attend(q[start:start + 8], [r], [0], [first], 8)
        assert torch.equal(window, together[start:start + 8]), r


def test_the_speed_knobs_keep_every_bit():
    global la
    default = la
    cache = Cache(512, 64, 64, [3000], seed=4)
    q = _q(4, 16, 576, 6)
    base = cache.attend(q, [0], [0], [2996], 4)
    ran = 0
    try:
        for knobs in ({"BLOCK_M": 32}, {"BLOCK_M": 32, "VSPLIT": 1}, {"STAGES": 2}, {"WARPS": 8}, {"VSPLIT": 4},
                      {"WARPS": 2}, {"PIECE": 32, "BLOCK_M": 32, "STAGES": 2, "VSPLIT": 1}):
            la = load("latent-attention", knobs={"window": knobs})
            try:
                got = cache.attend(q, [0], [0], [2996], 4)
            except triton.runtime.errors.OutOfResources:   # tiles this GPU's shared memory cannot hold
                continue
            ran += 1
            assert torch.equal(got, base) or "PIECE" in knobs, knobs
            torch.testing.assert_close(got.float(), base.float(), atol=8e-3, rtol=2e-2)
        assert ran >= 4
    finally:
        la = default


def test_one_cuda_graph_replays_at_every_length():
    cache = Cache(512, 64, 64, [4000], seed=3)
    q = _q(1, 16, 576, 7)
    first = torch.tensor([10], dtype=torch.int64, device=dev)
    args = (cache.cache, cache.tables, torch.zeros(1, dtype=torch.int32, device=dev),
            torch.zeros(1, dtype=torch.int32, device=dev), first, 1, 64)
    out = torch.empty(1, 16, 512, dtype=torch.bfloat16, device=dev)
    la.attend(q, *args, rank=512, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        la.attend(q, *args, rank=512, out=out)
    for at in (0, 63, 64, 1000, 3999):
        first.fill_(at)
        graph.replay()
        assert torch.equal(out, cache.attend(q, [0], [0], [at], 1)), at


def test_what_it_cannot_read_is_refused():
    cache = Cache(512, 64, 64, [10])
    q = _q(1, 16, 576, 1)
    with pytest.raises(ValueError, match="BF16 or FP16 in the cache's dtype"):
        cache.attend(q.float(), [0], [0], [5], 1)
    with pytest.raises(ValueError, match="powers of two"):
        la.attend(q, cache.cache, cache.tables, *(torch.zeros(1, dtype=t, device=dev) for t in
                                                 (torch.int32, torch.int32, torch.int64)), 1, 64, rank=500)
    with pytest.raises(ValueError, match="columns a row"):
        cache.attend(q, [0], [0], [5], 65)
    with pytest.raises(ValueError, match="sinks are contiguous FP32"):
        cache.attend(q, [0], [0], [5], 1, sinks=torch.zeros(3, device=dev))


class Fp8Cache(Cache):
    """The same sequences in an E4M3 cache with its scales (write()'s rule)."""

    def __init__(self, rank, rope, page, lengths, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.rank, self.rope, self.page = rank, rope, page
        width = max(-(-n // page) for n in lengths) + 1
        n_pages = width * len(lengths) + 2
        self.cache = torch.zeros(n_pages, page, rank + rope, dtype=torch.float8_e4m3fn, device=dev)
        self.scales = torch.zeros(n_pages, page, 2, device=dev)
        order = torch.randperm(n_pages, generator=g)
        self.tables = order[:width * len(lengths)].view(len(lengths), width).to(torch.int32).to(dev)
        for s, n in enumerate(lengths):
            latent = torch.randn(n, rank, generator=g).to(dev)
            rope_k = torch.randn(n, rope, generator=g).to(dev) * 3
            la.write(latent, rope_k, self.slots(s, torch.arange(n, device=dev)), self.cache, self.scales)

    def rows(self, s, n):
        slots = self.slots(s, torch.arange(n, device=dev))
        codes = self.cache.view(-1, self.rank + self.rope)[slots].double()
        sc = self.scales.view(-1, 2)[slots].double()
        return codes[:, :self.rank] * sc[:, :1], codes[:, self.rank:] * sc[:, 1:]

    def attend(self, q, rows, starts, firsts, columns, **kw):
        return super().attend(q, rows, starts, firsts, columns, scales=self.scales, **kw)


def test_an_e4m3_cache_matches_float64_over_its_stored_values_and_keeps_a_tokens_bits():
    cache = Fp8Cache(512, 64, 64, [130, 517], seed=6)
    q = _q(8, 16, 576, 8)
    lse = torch.empty(8, 16, device=dev)
    out = cache.attend(q, [0, 1], [0, 4], [126, 513], 4, scale=192 ** -0.5, lse=lse)
    for s, start, first in ((0, 0, 126), (1, 4, 513)):
        latent, rope_k = cache.rows(s, first + 4)
        seen = torch.arange(first + 4, device=dev)[None, :] <= first + torch.arange(4, device=dev)[:, None]
        span = slice(start, start + 4)
        want, want_lse = ref.latent_attention(q[span, :, :512], q[span, :, 512:], latent, rope_k, seen, 192 ** -0.5)
        torch.testing.assert_close(out[span].double(), want, atol=8e-3, rtol=2e-2)
        torch.testing.assert_close(lse[span].double(), want_lse, atol=1e-3, rtol=1e-4)
        for c in range(4):
            alone = cache.attend(q[start + c:start + c + 1], [s], [0], [first + c], 1, scale=192 ** -0.5)
            assert torch.equal(alone[0], out[start + c]), (s, c)
    with pytest.raises(ValueError, match="E4M3 cache takes its scales"):
        la.attend(q, cache.cache, cache.tables, *(torch.zeros(1, dtype=t, device=dev) for t in
                                                 (torch.int32, torch.int32, torch.int64)), 1, 64, rank=512)


@pytest.mark.parametrize("rank,rope,heads,page", [(512, 64, 16, 64), (512, 64, 128, 16), (128, 32, 20, 16)])
def test_prompt_chunks_match_float64_latent_attention(rank, rope, heads, page):
    cache = Cache(rank, rope, page, [900], seed=11)
    sinks = torch.randn(heads, generator=torch.Generator().manual_seed(12)).to(dev)
    for position, T in ((0, 70), (400, 133), (899 - 37, 38)):
        q = _q(T, heads, rank + rope, position + 1)
        for s in (None, sinks):
            lse = torch.empty(T, heads, device=dev)
            out = la.prefill(q, cache.cache, cache.tables[0], position, page, rank=rank, scale=192 ** -0.5, lse=lse,
                             sinks=s)
            latent, rope_k = cache.rows(0, position + T)
            seen = torch.arange(position + T, device=dev)[None, :] <= position + torch.arange(T, device=dev)[:, None]
            want, want_lse = ref.latent_attention(q[:, :, :rank], q[:, :, rank:], latent, rope_k, seen, 192 ** -0.5, s)
            torch.testing.assert_close(out.double(), want, atol=8e-3, rtol=2e-2)
            # BF16 probabilities summed on the tensor cores: over a chunk's first tokens' few positions their rounding
            # does not average out (up to about 2^-9 of the sum).
            torch.testing.assert_close(lse.double(), want_lse, atol=2e-3, rtol=1e-4)


def test_a_prompt_tokens_bits_do_not_follow_its_chunks_length_and_a_graph_replays_at_device_positions():
    cache = Cache(512, 64, 64, [3000], seed=13)
    q = _q(200, 16, 576, 14)
    full = la.prefill(q, cache.cache, cache.tables[0], 2000, 64, rank=512)
    for T in (1, 37, 64, 129):
        assert torch.equal(la.prefill(q[:T], cache.cache, cache.tables[0], 2000, 64, rank=512), full[:T]), T
    position = torch.tensor([0], dtype=torch.int64, device=dev)
    out = torch.empty(200, 16, 512, dtype=torch.bfloat16, device=dev)
    la.prefill(q, cache.cache, cache.tables[0], position, 64, rank=512, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        la.prefill(q, cache.cache, cache.tables[0], position, 64, rank=512, out=out)
    for at in (0, 700, 2799):
        position.fill_(at)
        graph.replay()
        assert torch.equal(out, la.prefill(q, cache.cache, cache.tables[0], at, 64, rank=512)), at


def test_a_prompt_chunk_over_an_e4m3_cache_matches_float64_over_its_stored_values():
    cache = Fp8Cache(512, 64, 64, [600], seed=15)
    q = _q(90, 16, 576, 16)
    out = la.prefill(q, cache.cache, cache.tables[0], 500, 64, rank=512, scale=192 ** -0.5, scales=cache.scales)
    latent, rope_k = cache.rows(0, 590)
    seen = torch.arange(590, device=dev)[None, :] <= 500 + torch.arange(90, device=dev)[:, None]
    want, _ = ref.latent_attention(q[:, :, :512], q[:, :, 512:], latent, rope_k, seen, 192 ** -0.5)
    torch.testing.assert_close(out.double(), want, atol=8e-3, rtol=2e-2)
