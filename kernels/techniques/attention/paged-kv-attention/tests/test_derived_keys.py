"""Derived keys (derive=): one cached row per position and KV head, read as its value and, normed and rotated, as its
key. Against the float64 reference of test_paged_kv_attention, the keys derived in float64 on the host from the rows as
the pages store them: every page kind, head widths 128 and 64, GQA, positions at page boundaries and a sliding window's
edges, a full context of 4,096 positions, partial RoPE and no norm weight, trees, sinks, log-sum-exps and fresh rows,
at the tolerances the plain-key tests take for the same call; a column's bits the same alone, in any window and fused
or split; the flash structures refuse derive=."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_paged_kv_attention import KINDS, STORE, _q, dev, pk, reference  # noqa: E402
from test_sinks import reference as sink_reference  # noqa: E402

EPS = 1e-6


class Rows:
    """One layer's value pages for several sequences over shuffled pages, written values only (write(None, ...))."""

    def __init__(self, kind, hkv, d, page, lengths, seed=0, multiplier=0.03):
        g = torch.Generator().manual_seed(seed)
        self.kind, self.hkv, self.d, self.page = kind, hkv, d, page
        self.multipliers = (1.0, multiplier) if kind == "fp8" else (1.0, 1.0)
        width = max(-(-n // page) for n in lengths) + 1
        n_pages = width * len(lengths) + 3
        self.v = torch.zeros(n_pages, page, hkv, d, dtype=STORE[kind], device=dev)
        self.vs = torch.zeros(n_pages, page, hkv, dtype=torch.float32, device=dev) \
            if kind in ("fp8:row", "int8") else None
        order = torch.randperm(n_pages, generator=g)
        self.tables = order[:width * len(lengths)].view(len(lengths), width).to(torch.int32).to(dev)
        for s, n in enumerate(lengths):
            # Rows of several magnitudes (the norm takes them to one), so the derived keys are not the rows.
            rows = torch.randn(n, hkv, d, generator=g) * (0.5 + 3 * torch.rand(n, hkv, 1, generator=g))
            pk.write(None, rows.to(dev), self.slots(s, torch.arange(n, device=dev)), None, self.v, kind, None,
                     self.vs, multipliers=self.multipliers)

    def slots(self, s, positions):
        positions = torch.as_tensor(positions, device=dev)
        return self.tables[s][positions // self.page].long() * self.page + positions % self.page

    def rows(self, s, n):
        """Positions 0..n-1 of sequence s as stored, float64 [n][hkv][d]."""
        slots = self.slots(s, torch.arange(n, device=dev))
        v = self.v.view(-1, self.hkv, self.d)[slots].double()
        if self.kind == "fp8":
            v = v * self.multipliers[1]
        if self.vs is not None:
            v = v * self.vs.view(-1, self.hkv)[slots].double()[..., None]
        return v

    def attend(self, q, rows, starts, firsts, columns, derive, **kw):
        q_dtype = torch.float16 if self.kind == "fp16" else torch.bfloat16
        return pk.attend(q.to(q_dtype), None, self.v, self.kind, self.tables,
                         torch.tensor(rows, dtype=torch.int32, device=dev),
                         torch.tensor(starts, dtype=torch.int32, device=dev),
                         torch.tensor(firsts, dtype=torch.int64, device=dev), columns, self.page,
                         value_scales=self.vs, multipliers=self.multipliers, derive=derive, **kw)


def derive_for(d, rotary_dim, weighted=True, seed=5, packed=True):
    g = torch.Generator().manual_seed(seed)
    freqs = (10000.0 ** (-torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim)).float().to(dev)
    weight = (1.0 + 0.3 * torch.randn(d, generator=g)).to(dev) if weighted else None
    return pk.derive(weight, EPS, freqs, rotary_dim) if packed else (weight, EPS, freqs, rotary_dim)


def derived_keys(rows, derive, start=0):
    """float64 keys of rows [n][hkv][d] at positions start..: rope(rms_norm(r) * w, p), the half rotation, the angle
    fp32(p) * freq as the kernel takes it."""
    weight, eps, freqs, rot = derive[:4]
    y = rows / torch.sqrt((rows * rows).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.double()
    pos = torch.arange(start, start + rows.shape[0], device=dev, dtype=torch.float32)
    angle = (pos[:, None] * freqs[None, :]).double()[:, None, :]
    half = rot // 2
    a, b = y[..., :half], y[..., half:rot]
    turned = torch.cat([a * angle.cos() - b * angle.sin(), b * angle.cos() + a * angle.sin()], -1)
    return torch.cat([turned, y[..., rot:]], -1)


def q_double(q, kind):
    return q.to(torch.float16 if kind == "fp16" else torch.bfloat16).double()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("hq,hkv,d,rot,weighted", [(8, 2, 128, 128, True), (12, 2, 64, 64, True),
                                                   (8, 4, 128, 64, False), (16, 2, 64, 32, True)])
@pytest.mark.parametrize("fused", [False, True])
def test_windows_with_derived_keys_match_float64_attention(kind, hq, hkv, d, rot, weighted, fused):
    page = 16
    # Rows whose windows start and end at page boundaries (15/16, 31/32) and one of a single position.
    cache = Rows(kind, hkv, d, page, [140, 64, 1, 517])
    derive = derive_for(d, rot, weighted)
    cols = 5
    rows, starts, firsts, counts = [0, 1, 2, 3], [0, 5, 10, 15], [127, 31, 0, 400], [5, 5, 1, 3]
    q = _q(20, hq, d, 3)
    lse = torch.empty(20, hq, device=dev)
    out = cache.attend(q, rows, starts, firsts, cols, derive, lse=lse, fused=fused,
                       counts=torch.tensor(counts, dtype=torch.int32, device=dev))
    scale = d ** -0.5
    qd = q_double(q, kind)
    for r, (s, start, first, count) in enumerate(zip(rows, starts, firsts, counts)):
        values = cache.rows(s, first + count)
        keys = derived_keys(values, derive)
        for c in range(count):
            seen = torch.arange(first + count, device=dev) <= first + c
            want, want_lse = reference(qd[start + c], keys, values, seen, scale)
            torch.testing.assert_close(out[start + c].double(), want, atol=8e-3, rtol=2e-2)
            torch.testing.assert_close(lse[start + c].double(), want_lse, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("kind", ["bf16", "fp8:row", "int8", "fp8"])
def test_a_full_context_and_a_sliding_windows_edges_with_sinks(kind):
    hq, hkv, d = 8, 2, 128
    n = 4096
    cache = Rows(kind, hkv, d, 64, [n + 4], seed=2)
    derive = derive_for(d, 128)
    q = _q(4, hq, d, 9)
    qd = q_double(q, kind)
    values = cache.rows(0, n + 4)
    keys = derived_keys(values, derive)
    sinks = torch.randn(hq, generator=torch.Generator().manual_seed(4)).to(dev)
    for window, sk in ((0, None), (64, None), (65, sinks), (1000, sinks)):
        lse = torch.empty(4, hq, device=dev)
        out = cache.attend(q, [0], [0], [n], 4, derive, lse=lse, sliding=window, fused=True, sinks=sk)
        split = cache.attend(q, [0], [0], [n], 4, derive, sliding=window, sinks=sk)
        assert torch.equal(out, split), window
        at = torch.arange(n + 4, device=dev)
        for c in range(4):
            seen = (at <= n + c) & ((at > n + c - window) if window else True)
            want, want_lse = sink_reference(qd[c], keys, values, seen, d ** -0.5,
                                            sk if sk is not None else torch.full((hq,), float("-inf"), device=dev))
            torch.testing.assert_close(out[c].double(), want, atol=8e-3, rtol=2e-2, msg=(window, c))
            torch.testing.assert_close(lse[c].double(), want_lse, atol=1e-3, rtol=1e-4, msg=(window, c))


@pytest.mark.parametrize("kind", ["bf16", "fp8:row", "int8", "fp8"])
@pytest.mark.parametrize("hq,hkv,d", [(24, 4, 128), (12, 2, 64)])
def test_a_columns_bits_are_its_own_alone_in_any_window_and_fused_or_split(kind, hq, hkv, d):
    cache = Rows(kind, hkv, d, 16, [700, 1500])
    derive = derive_for(d, d // 2)
    q = _q(16, hq, d, 13)
    lse = torch.empty(16, hq, device=dev)
    for cols in (2, 5, 8):
        window = cache.attend(q, [0, 1], [0, 8], [690, 1490], cols, derive, lse=lse)
        fused = cache.attend(q, [0, 1], [0, 8], [690, 1490], cols, derive, fused=True)
        for start in (0, 8):   # a launch writes its rows' tokens only
            assert torch.equal(window[start:start + cols], fused[start:start + cols]), cols
        for r, (s, start, first) in enumerate(zip([0, 1], [0, 8], [690, 1490])):
            for c in range(cols):
                one_lse = torch.empty(1, hq, device=dev)
                one = cache.attend(q[start + c:start + c + 1], [s], [0], [first + c], 1, derive, lse=one_lse)
                assert torch.equal(one[0], window[start + c]), (cols, r, c)
                assert torch.equal(one_lse[0], lse[start + c]), (cols, r, c)
    parents = torch.tensor([-1, 0, 0, 1, 1, 2, 5, 3], dtype=torch.int64, device=dev)
    tree = cache.attend(q[:8], [0], [0], [690], 8, derive, parents=parents)
    assert torch.equal(cache.attend(q[:8], [0], [0], [690], 8, derive, parents=parents, fused=True), tree)
    # A plain tuple derives the same keys as a packed Derive.
    plain = derive_for(d, d // 2, packed=False)
    assert torch.equal(cache.attend(q, [0, 1], [0, 8], [690, 1490], 8, plain),
                       cache.attend(q, [0, 1], [0, 8], [690, 1490], 8, derive))


@pytest.mark.parametrize("kind", ["bf16", "fp8"])
def test_a_tree_sees_its_ancestors_through_derived_keys(kind):
    cache = Rows(kind, 2, 128, 16, [400])
    derive = derive_for(128, 64)
    parents = torch.tensor([-1, 0, 0, 1, 1, 2, 5, 3], dtype=torch.int64, device=dev)
    q = _q(8, 8, 128, 6)
    out = cache.attend(q, [0], [0], [392], 8, derive, parents=parents)
    values = cache.rows(0, 400)
    keys = derived_keys(values, derive)
    anc = []
    for c in range(8):
        found, p = {c}, int(parents[c])
        while p >= 0:
            found.add(p)
            p = int(parents[p])
        anc.append(found)
    qd = q_double(q, kind)
    for c in range(8):
        seen = torch.tensor([n < 392 or (n - 392) in anc[c] for n in range(400)], device=dev)
        want, _ = reference(qd[c], keys, values, seen, 128 ** -0.5)
        torch.testing.assert_close(out[c].double(), want, atol=8e-3, rtol=2e-2)


@pytest.mark.parametrize("kind", ["fp8:row", "int8", "fp8"])
def test_a_fused_chunk_derives_its_fresh_rows_keys(kind):
    cache = Rows(kind, 2, 128, 16, [700], multiplier=0.04)
    derive = derive_for(128, 128)
    g = torch.Generator().manual_seed(21)
    T = 100
    q = _q(T, 8, 128, 22).to(torch.bfloat16)
    fv = torch.randn(T, 2, 128, generator=g).to(dev).to(torch.bfloat16)
    frontier = torch.zeros(1, dtype=torch.int64, device=dev)
    first = torch.zeros(1, dtype=torch.int64, device=dev)
    starts = torch.tensor([0, 64], dtype=torch.int32, device=dev)
    counts = torch.tensor([64, T - 64], dtype=torch.int32, device=dev)
    rows = torch.zeros(2, dtype=torch.int32, device=dev)
    qd = q.double()
    for at in (0, 200, 600):
        for split in (0, 37, 100):
            frontier.fill_(at)
            first.fill_(at + split)
            lse = torch.empty(T, 8, device=dev)
            out = pk.attend(q, None, cache.v, kind, cache.tables, rows, starts,
                            torch.tensor([at, at + 64], dtype=torch.int64, device=dev), 64, 16, counts=counts,
                            value_scales=cache.vs, multipliers=cache.multipliers, lse=lse, fused=True,
                            fresh=(None, fv, frontier, first), derive=derive)
            values = torch.cat([cache.rows(0, at + split), fv[split:].double()])
            keys = derived_keys(values, derive)
            for t in (0, 41, 99):
                seen = torch.arange(at + T, device=dev) <= at + t
                want, want_lse = reference(qd[t], keys, values, seen, 128 ** -0.5)
                torch.testing.assert_close(out[t].double(), want, atol=8e-3, rtol=2e-2, msg=(at, split, t))
                torch.testing.assert_close(lse[t].double(), want_lse, atol=1e-3, rtol=1e-4, msg=(at, split, t))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("d,rot,weighted", [(128, 128, True), (64, 32, False)])
def test_prompt_chunks_with_derived_keys_match_float64_attention(kind, d, rot, weighted):
    cache = Rows(kind, 2, d, 16, [300])
    derive = derive_for(d, rot, weighted)
    q = _q(100, 8, d, 4)
    q_dtype = torch.float16 if kind == "fp16" else torch.bfloat16
    lse = torch.empty(100, 8, device=dev)
    out = pk.prefill(q.to(q_dtype), None, cache.v, kind, cache.tables[0], 200, 16, value_scales=cache.vs,
                     multipliers=cache.multipliers, lse=lse, derive=derive)
    values = cache.rows(0, 300)
    keys = derived_keys(values, derive)
    qd = q.to(q_dtype).double()
    for t in (0, 37, 99):
        seen = torch.arange(300, device=dev) <= 200 + t
        want, want_lse = reference(qd[t], keys, values, seen, d ** -0.5)
        torch.testing.assert_close(out[t].double(), want, atol=8e-3, rtol=2e-2)
        torch.testing.assert_close(lse[t].double(), want_lse, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("kind", ["fp8:row", "bf16"])
def test_a_prompt_chunk_derives_its_fresh_rows_keys_with_a_sliding_window(kind):
    cache = Rows(kind, 2, 128, 16, [700])
    derive = derive_for(128, 64)
    g = torch.Generator().manual_seed(21)
    q = _q(100, 8, 128, 22).to(torch.bfloat16)
    fv = torch.randn(100, 2, 128, generator=g).to(dev).to(torch.bfloat16)
    qd = q.double()
    for at, split, window in ((0, 0, 0), (200, 37, 0), (600, 100, 0), (200, 37, 50)):
        lse = torch.empty(100, 8, device=dev)
        out = pk.prefill(q, None, cache.v, kind, cache.tables[0], at, 16, value_scales=cache.vs, lse=lse,
                         fresh=(None, fv, at + split), derive=derive, sliding=window)
        values = torch.cat([cache.rows(0, at + split), fv[split:].double()])
        keys = derived_keys(values, derive)
        n = torch.arange(at + 100, device=dev)
        for t in (0, 41, 99):
            seen = (n <= at + t) & ((n > at + t - window) if window else True)
            want, want_lse = reference(qd[t], keys, values, seen, 128 ** -0.5)
            torch.testing.assert_close(out[t].double(), want, atol=8e-3, rtol=2e-2, msg=(at, split, window, t))
            torch.testing.assert_close(lse[t].double(), want_lse, atol=1e-3, rtol=1e-4, msg=(at, split, window, t))


def test_without_derive_nothing_changes_and_values_only_writes_are_key_only_writes():
    g = torch.Generator().manual_seed(3)
    values = torch.randn(20, 2, 64, generator=g).to(dev)
    slots = torch.arange(20, device=dev) + 5
    for kind in ("bf16", "int8", "fp8"):
        v = torch.zeros(32, 2, 64, dtype=STORE[kind], device=dev)
        vs = torch.zeros(32, 2, device=dev) if kind == "int8" else None
        pk.write(None, values, slots, None, v, kind, None, vs, multipliers=(0.5, 0.03))
        k = torch.zeros_like(v)
        ks = torch.zeros_like(vs) if vs is not None else None
        pk.write(values, None, slots, k, None, kind, ks, None, multipliers=(0.03, 0.5))
        view = (lambda t: t.view(torch.uint8)) if kind == "fp8" else (lambda t: t)
        assert torch.equal(view(v), view(k))
        if vs is not None:
            assert torch.equal(vs, ks)


@pytest.mark.parametrize("flash", [True, "cuda"])
def test_the_flash_structures_refuse_derived_keys(flash):
    cache = Rows("bf16", 2, 128, 16, [300])
    derive = derive_for(128, 128)
    q = _q(4, 8, 128, 2)
    with pytest.raises(ValueError, match="flash structures"):
        cache.attend(q, [0], [0], [296], 4, derive, flash=flash)
    with pytest.raises(ValueError, match="flash structures"):
        pk.prefill(q.to(torch.bfloat16), None, cache.v, "bf16", cache.tables[0], 200, 16, derive=derive, flash=flash)


def test_a_derive_is_checked():
    cache = Rows("bf16", 2, 128, 16, [300])
    q = _q(1, 8, 128, 2)
    freqs = torch.ones(64, device=dev)
    for bad, match in (((None, EPS, freqs, 127), "rotary_dim"), ((None, EPS, freqs, 256), "rotary_dim"),
                       ((None, EPS, freqs.double(), 128), "freqs"), ((None, EPS, freqs[:32], 128), "freqs"),
                       ((torch.ones(64, device=dev), EPS, freqs, 128), "weight")):
        with pytest.raises(ValueError, match=match):
            cache.attend(q, [0], [0], [296], 1, bad)
    with pytest.raises(ValueError, match="key_pages None"):
        pk.attend(q.to(torch.bfloat16), torch.zeros_like(cache.v), cache.v, "bf16", cache.tables,
                  torch.zeros(1, dtype=torch.int32, device=dev), torch.zeros(1, dtype=torch.int32, device=dev),
                  torch.tensor([296], device=dev), 1, 16, derive=(None, EPS, freqs, 128))
